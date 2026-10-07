# Shared, memoizing feature extraction for models under model_store/.
#
# Centralizes only the expensive, model-agnostic computation each predictor
# used to redo independently: the ESM2 tokenize+forward-pass, and the
# modlAMP/propy descriptor passes. Pooling, scaling, PCA, and every other
# per-model transform stay exactly where they already live, inside each
# predictor's own code -- this class never pools, scales, or interprets a
# feature, it only computes and caches raw per-sequence representations.
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
# propy3 is OPTIONAL and is GPL-2.0-only. This project is Apache-2.0 and does not require it
# (see docs/LICENSING.md). It is NOT a cosmetic dependency: _propy_features() reproduces
# aggregation_predictor_v1's feature layout exactly, so without propy3 that predictor has no
# inputs and MUST NOT be run. Callers check PROPY_AVAILABLE and report the screen as not run --
# an unrun safety screen is never reported as a pass.
try:
    from propy.PyPro import GetProDes

    PROPY_AVAILABLE = True
    PROPY_UNAVAILABLE_REASON = None
except (
    Exception
) as problem:  # noqa: BLE001 - any import failure means "not installed here"
    GetProDes = None  # type: ignore[assignment]
    PROPY_AVAILABLE = False
    PROPY_UNAVAILABLE_REASON = (
        "propy3 is not installed, so the aggregation predictor could not run. It is optional and"
        " GPL-2.0-only; this project does not require it. The candidate has NOT been screened for"
        " aggregation -- this is not a pass. See docs/LICENSING.md."
        f" ({type(problem).__name__})"
    )
from transformers import AutoTokenizer, EsmModel

from common import storage
from common.model_sync import sync_model_weights, weights_dir_for
from model_store.hemolysis_predictor_v1.predictor import (
    Extractor as _HemolysisV1Extractor,
    normalize_sequences as _normalize_hemolysis_v1_sequences,
)
from model_store.hemolysis_predictor_v1.property_tables import (
    PROPERTY_TABLES as _HEMOLYSIS_V1_PROPERTY_TABLES,
)

_ESM2_EMBEDDINGS_FILE = "esm2_embeddings.npz"
_ESM2_MANIFEST_FILE = "esm2_manifest.json"
_DESCRIPTORS_FILE = "descriptors.json"

_ESM2_CODE_DIR = Path(__file__).resolve().parents[2] / "model_store" / "esm2_t30_150M"
DEFAULT_ESM2_MODEL_PATH = weights_dir_for(_ESM2_CODE_DIR)


_AROMATIC = set("FWY")
_HYDROPHOBIC = set("AVLIMFW")
_POLAR = set("STNQCY")
_POSITIVE = set("KRH")
_NEGATIVE = set("DE")
_SMALL = set("AGS")

_SOCN_QSO_LAG = 4
_PAAC_LAMBDA = 4
_PAAC_KEYS = [f"PAAC{i}" for i in range(1, 21 + _PAAC_LAMBDA)]
_APAAC_KEYS = [f"APAAC{i}" for i in range(1, 21 + 2 * _PAAC_LAMBDA)]


def _biopython_features(seq: str) -> dict[str, float]:
    """Matches aggregation_predictor_v1's biopython_features exactly."""
    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    ext_reduced, ext_oxidized = pa.molar_extinction_coefficient()
    n = len(seq)
    return {
        "bp_length": n,
        "bp_molecular_weight": pa.molecular_weight(),
        "bp_aromaticity": pa.aromaticity(),
        "bp_instability_index": pa.instability_index(),
        "bp_gravy": pa.gravy(),
        "bp_isoelectric_point": pa.isoelectric_point(),
        "bp_charge_at_pH7": pa.charge_at_pH(7.0),
        "bp_ss_helix_frac": helix,
        "bp_ss_turn_frac": turn,
        "bp_ss_sheet_frac": sheet,
        "bp_molar_ext_reduced": ext_reduced,
        "bp_molar_ext_oxidized": ext_oxidized,
        "bp_aromatic_frac": sum(c in _AROMATIC for c in seq) / n,
        "bp_hydrophobic_frac": sum(c in _HYDROPHOBIC for c in seq) / n,
        "bp_polar_frac": sum(c in _POLAR for c in seq) / n,
        "bp_positive_frac": sum(c in _POSITIVE for c in seq) / n,
        "bp_negative_frac": sum(c in _NEGATIVE for c in seq) / n,
        "bp_small_frac": sum(c in _SMALL for c in seq) / n,
        "bp_net_charge_frac": (
            sum(c in _POSITIVE for c in seq) - sum(c in _NEGATIVE for c in seq)
        )
        / n,
    }


def _safe_descriptor(fn, fallback_keys, *args, **kwargs) -> dict[str, float]:
    try:
        return fn(*args, **kwargs)
    except Exception:
        return {k: np.nan for k in fallback_keys}


class PropyUnavailable(RuntimeError):
    """Raised when a propy3-derived feature set is requested but propy3 is not installed.

    Deliberately an exception rather than a default or an empty dict: the aggregation predictor's
    feature vector has a fixed layout, and handing it zeros would produce a confident-looking
    score computed from nothing.
    """


def _propy_features(seq: str) -> dict[str, float]:
    """Matches aggregation_predictor_v1's pybiomed_features exactly."""
    if not PROPY_AVAILABLE:
        raise PropyUnavailable(PROPY_UNAVAILABLE_REASON)
    gp = GetProDes(seq)
    feats: dict[str, float] = {}
    feats.update(gp.GetAAComp())
    feats.update(gp.GetDPComp())
    feats.update(gp.GetCTD())
    feats.update(gp.GetMoranAuto())
    feats.update(gp.GetGearyAuto())
    feats.update(gp.GetMoreauBrotoAuto())
    feats.update(gp.GetQSO(maxlag=_SOCN_QSO_LAG))
    feats.update(gp.GetSOCN(maxlag=_SOCN_QSO_LAG))
    feats.update(_safe_descriptor(gp.GetPAAC, _PAAC_KEYS, lamda=_PAAC_LAMBDA))
    feats.update(_safe_descriptor(gp.GetAPAAC, _APAAC_KEYS, lamda=_PAAC_LAMBDA))
    return feats


@dataclass(frozen=True)
class ESM2Embedding:
    """One sequence's raw ESM2 output; `hidden_states` includes BOS/EOS and is trimmed to this sequence's length."""

    hidden_states: np.ndarray  # (seq_len_with_specials, hidden_dim), float32
    input_ids: np.ndarray  # (seq_len_with_specials,), token ids incl. BOS/EOS
    cls_token_id: int
    eos_token_id: int


def _sequence_cache_key(sequence: str, fingerprint: str) -> str:
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    return f"{digest}:{fingerprint}"


class FeatureExtractor:
    """Run-scoped cache of raw per-sequence features, keyed by sequence hash plus an extractor/checkpoint fingerprint."""

    def __init__(
        self,
        esm2_model_path: Path = DEFAULT_ESM2_MODEL_PATH,
        esm2_batch_size: int = 256,
    ) -> None:
        self.esm2_model_path = Path(esm2_model_path)
        self.esm2_batch_size = esm2_batch_size
        self._esm2_fingerprint = f"esm2:{self.esm2_model_path.name}"
        self._esm2_cache: dict[str, ESM2Embedding] = {}
        self._modlamp_cache: dict[str, dict[str, float]] = {}
        self._propy_biopython_cache: dict[str, dict[str, float]] = {}
        self._hemolysis_v1_cache: dict[str, dict[str, float]] = {}

        self._esm2_tokenizer: Any = None
        self._esm2_model: Any = None
        self._esm2_device: str | None = None
        self._hemolysis_v1_extractor: _HemolysisV1Extractor | None = None

    # Persistence: round-trips the four cache dicts through feature_cache/.

    def save(self, dir: str) -> None:
        storage.ensure_dir(dir)

        if self._esm2_cache:
            arrays: dict[str, np.ndarray] = {}
            for key, embedding in self._esm2_cache.items():
                arrays[f"{key}__hidden"] = embedding.hidden_states
                arrays[f"{key}__ids"] = embedding.input_ids
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            storage.write_bytes(
                storage.join(dir, _ESM2_EMBEDDINGS_FILE), buffer.getvalue()
            )

            first = next(iter(self._esm2_cache.values()))
            manifest = {
                "keys": list(self._esm2_cache.keys()),
                "cls_token_id": first.cls_token_id,
                "eos_token_id": first.eos_token_id,
            }
            storage.write_text(
                storage.join(dir, _ESM2_MANIFEST_FILE), json.dumps(manifest)
            )

        descriptors = {
            "modlamp": self._modlamp_cache,
            "propy_biopython": self._propy_biopython_cache,
            "hemolysis_v1": self._hemolysis_v1_cache,
        }
        storage.write_text(
            storage.join(dir, _DESCRIPTORS_FILE), json.dumps(descriptors)
        )

    @classmethod
    def load(
        cls,
        dir: str,
        esm2_model_path: Path = DEFAULT_ESM2_MODEL_PATH,
        esm2_batch_size: int = 256,
    ) -> "FeatureExtractor":
        """Missing files leave the corresponding dict empty, same as a fresh instance."""
        extractor = cls(
            esm2_model_path=esm2_model_path, esm2_batch_size=esm2_batch_size
        )

        embeddings_path = storage.join(dir, _ESM2_EMBEDDINGS_FILE)
        manifest_path = storage.join(dir, _ESM2_MANIFEST_FILE)
        if storage.exists(embeddings_path) and storage.exists(manifest_path):
            manifest = json.loads(storage.read_text(manifest_path))
            with np.load(io.BytesIO(storage.read_bytes(embeddings_path))) as npz:
                for key in manifest["keys"]:
                    extractor._esm2_cache[key] = ESM2Embedding(
                        hidden_states=npz[f"{key}__hidden"],
                        input_ids=npz[f"{key}__ids"],
                        cls_token_id=manifest["cls_token_id"],
                        eos_token_id=manifest["eos_token_id"],
                    )

        descriptors_path = storage.join(dir, _DESCRIPTORS_FILE)
        if storage.exists(descriptors_path):
            descriptors = json.loads(storage.read_text(descriptors_path))
            extractor._modlamp_cache = descriptors.get("modlamp", {})
            extractor._propy_biopython_cache = descriptors.get("propy_biopython", {})
            extractor._hemolysis_v1_cache = descriptors.get("hemolysis_v1", {})

        return extractor

    # ESM2

    def _load_esm2(self) -> None:
        if self._esm2_model is not None:
            return
        sync_model_weights(_ESM2_CODE_DIR)
        self._esm2_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._esm2_tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_path)
        self._esm2_model = (
            EsmModel.from_pretrained(self.esm2_model_path).to(self._esm2_device).eval()
        )

    def get_esm2_embedding(self, sequence: str) -> ESM2Embedding:
        """Raw per-token ESM2 hidden states for one sequence, memoized."""
        return self.get_esm2_embedding_batch([sequence])[0]

    def get_esm2_embedding_batch(self, sequences: list[str]) -> list[ESM2Embedding]:
        """Raw per-token ESM2 hidden states for a batch, memoized per sequence and returned in input order."""
        keys = [_sequence_cache_key(s, self._esm2_fingerprint) for s in sequences]
        missing_indices = [i for i, k in enumerate(keys) if k not in self._esm2_cache]

        if missing_indices:
            self._load_esm2()
            cls_id = self._esm2_tokenizer.cls_token_id
            eos_id = self._esm2_tokenizer.eos_token_id
            for chunk_start in range(0, len(missing_indices), self.esm2_batch_size):
                chunk_indices = missing_indices[
                    chunk_start : chunk_start + self.esm2_batch_size
                ]
                chunk_sequences = [sequences[i] for i in chunk_indices]
                with torch.no_grad():
                    enc = self._esm2_tokenizer(
                        chunk_sequences, return_tensors="pt", padding=True
                    ).to(self._esm2_device)
                    out = self._esm2_model(**enc)
                    hidden = out.last_hidden_state
                    attention_mask = enc["attention_mask"]

                for row, index in enumerate(chunk_indices):
                    n_valid = int(attention_mask[row].sum().item())
                    trimmed_hidden = (
                        hidden[row, :n_valid].cpu().numpy().astype(np.float32)
                    )
                    trimmed_ids = enc["input_ids"][row, :n_valid].cpu().numpy()
                    self._esm2_cache[keys[index]] = ESM2Embedding(
                        hidden_states=trimmed_hidden,
                        input_ids=trimmed_ids,
                        cls_token_id=cls_id,
                        eos_token_id=eos_id,
                    )

        return [self._esm2_cache[k] for k in keys]

    # modlAMP physicochemical descriptors

    def get_modlamp_descriptors(self, sequence: str) -> dict[str, float]:
        """GlobalDescriptor plus per-scale PeptideDescriptor features in the layout amp_classifier_v1/mbic_predictor_v1 expect; memoized per sequence."""
        return self.get_modlamp_descriptors_batch([sequence])[0]

    def get_modlamp_descriptors_batch(
        self, sequences: list[str]
    ) -> list[dict[str, float]]:
        fingerprint = "modlamp"
        keys = [_sequence_cache_key(s, fingerprint) for s in sequences]
        missing_indices = [
            i for i, k in enumerate(keys) if k not in self._modlamp_cache
        ]

        if missing_indices:
            missing_sequences = [sequences[i].upper() for i in missing_indices]

            gd = GlobalDescriptor(missing_sequences)
            gd.calculate_all(amide=True)
            global_names = [f"modlamp_{name}" for name in gd.featurenames]

            scalar_scales = ["eisenberg", "gravy", "flexibility", "aasi"]
            multidim_scales = ["z3", "z5", "abhprk"]
            scale_columns: dict[str, np.ndarray] = {}
            for scale in scalar_scales:
                pdesc = PeptideDescriptor(missing_sequences, scale)
                pdesc.calculate_global()
                scale_columns[f"modlamp_{scale}_global"] = pdesc.descriptor.flatten()
                pdesc_m = PeptideDescriptor(missing_sequences, scale)
                pdesc_m.calculate_moment()
                scale_columns[f"modlamp_{scale}_moment"] = pdesc_m.descriptor.flatten()
            for scale in multidim_scales:
                pdesc = PeptideDescriptor(missing_sequences, scale)
                pdesc.calculate_global()
                scale_columns[f"modlamp_{scale}_global"] = pdesc.descriptor.flatten()

            for row, index in enumerate(missing_indices):
                row_feats = {
                    name: float(gd.descriptor[row, col])
                    for col, name in enumerate(global_names)
                }
                for name, values in scale_columns.items():
                    row_feats[name] = float(values[row])
                self._modlamp_cache[keys[index]] = row_feats

        return [self._modlamp_cache[k] for k in keys]

    # biopython / propy descriptors

    def get_propy_biopython_descriptors(self, sequence: str) -> dict[str, float]:
        """biopython ProteinAnalysis + propy GetProDes features in aggregation_predictor_v1's layout (excluding its own AAindex1 pass); memoized per sequence."""
        return self.get_propy_biopython_descriptors_batch([sequence])[0]

    def get_propy_biopython_descriptors_batch(
        self, sequences: list[str]
    ) -> list[dict[str, float]]:
        fingerprint = "propy_biopython"
        keys = [_sequence_cache_key(s, fingerprint) for s in sequences]
        missing_indices = [
            i for i, k in enumerate(keys) if k not in self._propy_biopython_cache
        ]

        for index in missing_indices:
            seq = sequences[index].upper()
            feats: dict[str, float] = {}
            feats.update(_biopython_features(seq))
            feats.update(_propy_features(seq))
            self._propy_biopython_cache[keys[index]] = feats

        return [self._propy_biopython_cache[k] for k in keys]

    def _get_hemolysis_v1_extractor(self) -> _HemolysisV1Extractor:
        if self._hemolysis_v1_extractor is None:
            self._hemolysis_v1_extractor = _HemolysisV1Extractor(
                _HEMOLYSIS_V1_PROPERTY_TABLES
            )
        return self._hemolysis_v1_extractor

    def get_hemolysis_v1_descriptors(self, sequence: str) -> dict[str, float]:
        """hemolysis_predictor_v1's full descriptor row in Extractor.extract's column layout; memoized per sequence."""
        return self.get_hemolysis_v1_descriptors_batch([sequence])[0]

    def get_hemolysis_v1_descriptors_batch(
        self, sequences: list[str]
    ) -> list[dict[str, float]]:
        fingerprint = "hemolysis_v1"
        keys = [_sequence_cache_key(s, fingerprint) for s in sequences]
        missing_indices = [
            i for i, k in enumerate(keys) if k not in self._hemolysis_v1_cache
        ]

        if missing_indices:
            missing_sequences = [sequences[i] for i in missing_indices]
            normalized = list(_normalize_hemolysis_v1_sequences(missing_sequences))
            extractor = self._get_hemolysis_v1_extractor()
            rows = extractor.extract(normalized)
            for row_position, index in enumerate(missing_indices):
                self._hemolysis_v1_cache[keys[index]] = rows.iloc[
                    row_position
                ].to_dict()

        return [self._hemolysis_v1_cache[k] for k in keys]
