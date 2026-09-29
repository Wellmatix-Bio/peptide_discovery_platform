"""Architecture + loader/inference wrapper for the MIC prediction ensemble."""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
ESM_DIM = 640
MAX_SEQ_LEN = 50
GENOME_DIM = 84

SUPPORTED_ORGANISMS = ["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if len(sequence) > MAX_SEQ_LEN:
        raise ValueError(f"sequence length {len(sequence)} exceeds max supported length {MAX_SEQ_LEN}.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def _validate_organism(organism: str) -> None:
    if organism not in SUPPORTED_ORGANISMS:
        raise ValueError(
            f"organism {organism!r} not supported. Must be one of: {SUPPORTED_ORGANISMS}"
        )


# ---------------------------------------------------------------------------
# Genomic nucleotide-composition features (NAC + DNC + TNC, 84-dim)
# Precomputed per organism at export time; see genome_vectors.json.
# ---------------------------------------------------------------------------

class GenomicANN(nn.Module):
    def __init__(self, in_dim=GENOME_DIM, out_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BiLSTMModel(nn.Module):
    def __init__(self, esm_dim=ESM_DIM, lstm_hidden=128, genome_dim=GENOME_DIM, genome_out=32):
        super().__init__()
        self.lstm = nn.LSTM(esm_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.genomic_ann = GenomicANN(genome_dim, genome_out)
        merged_dim = lstm_hidden * 2 + genome_out
        self.bn = nn.BatchNorm1d(merged_dim)
        self.regressor = nn.Sequential(
            nn.Linear(merged_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, esm2_residues, esm2_mask, genome):
        lengths = esm2_mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            esm2_residues, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        lstm_out = torch.cat([h_n[0], h_n[1]], dim=1)
        genome_out = self.genomic_ann(genome)
        merged = torch.cat([lstm_out, genome_out], dim=1)
        merged = self.bn(merged)
        return self.regressor(merged).squeeze(-1)


class CNNModel(nn.Module):
    def __init__(self, esm_dim=ESM_DIM, genome_dim=GENOME_DIM, genome_out=32, seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(esm_dim, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_out_len = seq_len // 4
        conv_out_dim = 64 * conv_out_len
        self.genomic_ann = GenomicANN(genome_dim, genome_out)
        merged_dim = conv_out_dim + genome_out
        self.bn = nn.BatchNorm1d(merged_dim)
        self.regressor = nn.Sequential(
            nn.Linear(merged_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, esm2_residues, esm2_mask, genome):
        x = esm2_residues * esm2_mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        genome_out = self.genomic_ann(genome)
        merged = torch.cat([x, genome_out], dim=1)
        merged = self.bn(merged)
        return self.regressor(merged).squeeze(-1)


# ---------------------------------------------------------------------------
# iFeature protein descriptors (AAC, GAAC, CTDC, CTDT, CTDD, PAAC lambda=1)
# Vendored from iFeature (Chen et al.) so the RF branch needs no external
# iFeature install. Group definitions and PAAC property table copied verbatim.
# ---------------------------------------------------------------------------

_AAC_ORDER = "ACDEFGHIKLMNPQRSTVWY"

_GAAC_GROUPS = {
    "alphatic": "GAVLMI",
    "aromatic": "FYW",
    "postivecharge": "KRH",
    "negativecharge": "DE",
    "uncharge": "STCPNQ",
}

_CTD_GROUP1 = {
    "hydrophobicity_PRAM900101": "RKEDQN", "hydrophobicity_ARGP820101": "QSTNGDE",
    "hydrophobicity_ZIMJ680101": "QNGSWTDERA", "hydrophobicity_PONP930101": "KPDESNQT",
    "hydrophobicity_CASG920101": "KDEQPSRNTG", "hydrophobicity_ENGD860101": "RDKENQHYP",
    "hydrophobicity_FASG890101": "KERSQD", "normwaalsvolume": "GASTPDC",
    "polarity": "LIFWCMVY", "polarizability": "GASDT", "charge": "KR",
    "secondarystruct": "EALMQKRH", "solventaccess": "ALFCGIVW",
}
_CTD_GROUP2 = {
    "hydrophobicity_PRAM900101": "GASTPHY", "hydrophobicity_ARGP820101": "RAHCKMV",
    "hydrophobicity_ZIMJ680101": "HMCKV", "hydrophobicity_PONP930101": "GRHA",
    "hydrophobicity_CASG920101": "AHYMLV", "hydrophobicity_ENGD860101": "SGTAW",
    "hydrophobicity_FASG890101": "NTPG", "normwaalsvolume": "NVEQIL",
    "polarity": "PATGS", "polarizability": "CPNVEQIL", "charge": "ANCQGHILMFPSTWYV",
    "secondarystruct": "VIYCWFT", "solventaccess": "RKQEND",
}
_CTD_GROUP3 = {
    "hydrophobicity_PRAM900101": "CLVIMFW", "hydrophobicity_ARGP820101": "LYPFIW",
    "hydrophobicity_ZIMJ680101": "LPFYI", "hydrophobicity_PONP930101": "YMFWLCVI",
    "hydrophobicity_CASG920101": "FIWC", "hydrophobicity_ENGD860101": "CVLIMF",
    "hydrophobicity_FASG890101": "AYHWVMFLIC", "normwaalsvolume": "MHKFRYW",
    "polarity": "HQRKNED", "polarizability": "KMHFRYW", "charge": "DE",
    "secondarystruct": "GNPSD", "solventaccess": "MSPTHY",
}
_CTD_PROPERTIES = (
    "hydrophobicity_PRAM900101", "hydrophobicity_ARGP820101", "hydrophobicity_ZIMJ680101",
    "hydrophobicity_PONP930101", "hydrophobicity_CASG920101", "hydrophobicity_ENGD860101",
    "hydrophobicity_FASG890101", "normwaalsvolume", "polarity", "polarizability",
    "charge", "secondarystruct", "solventaccess",
)

# PAAC.txt property table (Hydrophobicity, Hydrophilicity, SideChainMass), AA order below
_PAAC_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_PAAC_RAW_PROPERTIES = [
    [0.62, -2.53, -0.78, -0.9, 0.29, -0.85, -0.74, 0.48, -0.4, 1.38, 1.06, -1.5, 0.64, 1.19, 0.12, -0.18, -0.05, 0.81, 0.26, 1.08],
    [-0.5, 3, 0.2, 3, -1, 0.2, 3, 0, -0.5, -1.8, -1.8, 3, -1.3, -2.5, 0, 0.3, -0.4, -3.4, -2.3, -1.5],
    [15, 101, 58, 59, 47, 72, 73, 1, 82, 57, 57, 73, 75, 91, 42, 31, 45, 130, 107, 43],
]


def _ifeature_aac(sequence: str) -> dict:
    count = Counter(sequence)
    n = len(sequence)
    return {f"aac_{aa}": count.get(aa, 0) / n for aa in _AAC_ORDER}


def _ifeature_gaac(sequence: str) -> dict:
    count = Counter(sequence)
    n = len(sequence)
    out = {}
    for key, members in _GAAC_GROUPS.items():
        out[f"gaac_{key}"] = sum(count.get(aa, 0) for aa in members) / n
    return out


def _ctd_count(group: str, sequence: str) -> int:
    return sum(sequence.count(aa) for aa in group)


def _ifeature_ctdc(sequence: str) -> dict:
    n = len(sequence)
    out = {}
    for p in _CTD_PROPERTIES:
        c1 = _ctd_count(_CTD_GROUP1[p], sequence) / n
        c2 = _ctd_count(_CTD_GROUP2[p], sequence) / n
        c3 = 1 - c1 - c2
        out[f"ctdc_{p}.G1"] = c1
        out[f"ctdc_{p}.G2"] = c2
        out[f"ctdc_{p}.G3"] = c3
    return out


def _ifeature_ctdt(sequence: str) -> dict:
    aa_pairs = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
    n_pairs = max(len(aa_pairs), 1)
    out = {}
    for p in _CTD_PROPERTIES:
        g1, g2, g3 = _CTD_GROUP1[p], _CTD_GROUP2[p], _CTD_GROUP3[p]
        c1221 = c1331 = c2332 = 0
        for pair in aa_pairs:
            if (pair[0] in g1 and pair[1] in g2) or (pair[0] in g2 and pair[1] in g1):
                c1221 += 1
                continue
            if (pair[0] in g1 and pair[1] in g3) or (pair[0] in g3 and pair[1] in g1):
                c1331 += 1
                continue
            if (pair[0] in g2 and pair[1] in g3) or (pair[0] in g3 and pair[1] in g2):
                c2332 += 1
        out[f"ctdt_{p}.Tr1221"] = c1221 / n_pairs
        out[f"ctdt_{p}.Tr1331"] = c1331 / n_pairs
        out[f"ctdt_{p}.Tr2332"] = c2332 / n_pairs
    return out


def _ctd_distribution(group: str, sequence: str) -> list:
    number = sum(1 for aa in sequence if aa in group)
    cutoffs = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffs = [c if c >= 1 else 1 for c in cutoffs]
    code = []
    for cutoff in cutoffs:
        my_count = 0
        found = False
        for i, aa in enumerate(sequence):
            if aa in group:
                my_count += 1
                if my_count == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    found = True
                    break
        if my_count == 0:
            code.append(0)
        elif not found:
            code.append(0)
    return code


def _ifeature_ctdd(sequence: str) -> dict:
    out = {}
    for p in _CTD_PROPERTIES:
        for g_idx, group in enumerate((_CTD_GROUP1[p], _CTD_GROUP2[p], _CTD_GROUP3[p]), start=1):
            values = _ctd_distribution(group, sequence)
            for d, v in zip(("0", "25", "50", "75", "100"), values):
                out[f"ctdd_{p}.{g_idx}.residue{d}"] = v
    return out


def _paac_rvalue(aa1: str, aa2: str, aa_index: dict, normalized_props: list) -> float:
    return sum((row[aa_index[aa1]] - row[aa_index[aa2]]) ** 2 for row in normalized_props) / len(normalized_props)


def _ifeature_paac(sequence: str, lambda_value: int = 1, w: float = 0.05) -> dict:
    aa_index = {aa: i for i, aa in enumerate(_PAAC_AA_ORDER)}
    normalized_props = []
    for row in _PAAC_RAW_PROPERTIES:
        mean_i = sum(row) / 20
        denom = math.sqrt(sum((v - mean_i) ** 2 for v in row) / 20)
        normalized_props.append([(v - mean_i) / denom for v in row])

    theta = []
    for n in range(1, lambda_value + 1):
        pairs = range(len(sequence) - n)
        theta.append(
            sum(_paac_rvalue(sequence[j], sequence[j + n], aa_index, normalized_props) for j in pairs)
            / max(len(sequence) - n, 1)
        )
    aa_count = Counter(sequence)
    denom = 1 + w * sum(theta)
    out = {f"paac_Xc1.{aa}": aa_count.get(aa, 0) / denom for aa in _PAAC_AA_ORDER}
    for n, t in enumerate(theta, start=1):
        out[f"paac_Xc2.lambda{n}"] = (w * t) / denom
    return out


def ifeature_descriptors(sequence: str) -> dict:
    out = {}
    out.update(_ifeature_aac(sequence))
    out.update(_ifeature_gaac(sequence))
    out.update(_ifeature_ctdc(sequence))
    out.update(_ifeature_ctdt(sequence))
    out.update(_ifeature_ctdd(sequence))
    out.update(_ifeature_paac(sequence, lambda_value=1))
    return out


class MICPredictorEnsemble:
    """Lazy-loaded ensemble of BiLSTM, CNN (per-residue ESM2 + genome features)
    and Random Forest (iFeature descriptors + genome features). Predicts
    log10(MIC, uM) for one of 3 ATCC reference organisms. See README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bilstm_model = BiLSTMModel().to(self.device)
        self.bilstm_model.load_state_dict(
            torch.load(self.model_dir / "bilstm_model.pt", map_location=self.device, weights_only=False)
        )
        self.bilstm_model.eval()

        self.cnn_model = CNNModel().to(self.device)
        self.cnn_model.load_state_dict(
            torch.load(self.model_dir / "cnn_model.pt", map_location=self.device, weights_only=False)
        )
        self.cnn_model.eval()

        self.rf_model = joblib.load(self.model_dir / "rf_model.joblib")
        self.rf_feature_cols = list(self.rf_model.feature_names_in_)

        with open(self.model_dir / "genome_vectors.json", "r", encoding="utf-8") as f:
            self.genome_vectors = json.load(f)

        self.loaded = True

    def _residue_embedding(self, sequence: str, feature_extractor: FeatureExtractor):
        batch_arr, batch_mask = self._residue_embedding_batch(
            [sequence], feature_extractor
        )
        return batch_arr, batch_mask

    def _residue_embedding_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ):
        n = len(sequences)
        embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
        with torch.no_grad():
            # Cached hidden_states are already trimmed to each sequence's
            # own valid length (BOS + residues + EOS, no padding) --
            # stripping BOS/EOS (positions 0 and -1) leaves exactly the
            # same valid residues the original CLS/EOS-id masking selected.
            batch_arr = torch.zeros((n, MAX_SEQ_LEN, ESM_DIM), dtype=torch.float32, device=self.device)
            batch_mask = torch.zeros((n, MAX_SEQ_LEN), dtype=torch.float32, device=self.device)
            for row, embedding in enumerate(embeddings):
                residues = torch.from_numpy(embedding.hidden_states[1:-1]).to(self.device)
                n_valid = min(residues.shape[0], MAX_SEQ_LEN)
                batch_arr[row, :n_valid] = residues[:n_valid]
                batch_mask[row, :n_valid] = 1.0
        return batch_arr, batch_mask

    def _predict_bilstm_cnn(
        self,
        sequence: str,
        genome_vec: np.ndarray,
        feature_extractor: FeatureExtractor,
    ) -> tuple:
        residue_emb, residue_mask = self._residue_embedding(sequence, feature_extractor)
        genome_t = torch.tensor(genome_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            bilstm_pred = self.bilstm_model(residue_emb, residue_mask, genome_t).item()
            cnn_pred = self.cnn_model(residue_emb, residue_mask, genome_t).item()
        return bilstm_pred, cnn_pred

    def _predict_bilstm_cnn_batch(
        self,
        sequences: list[str],
        genome_vec: np.ndarray,
        feature_extractor: FeatureExtractor,
    ) -> tuple:
        n = len(sequences)
        residue_emb, residue_mask = self._residue_embedding_batch(sequences, feature_extractor)
        genome_t = torch.tensor(genome_vec, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        with torch.no_grad():
            bilstm_pred = self.bilstm_model(residue_emb, residue_mask, genome_t).cpu().numpy()
            cnn_pred = self.cnn_model(residue_emb, residue_mask, genome_t).cpu().numpy()
        return bilstm_pred, cnn_pred

    def _predict_rf(self, sequence: str, genome_vec: np.ndarray) -> float:
        return self._predict_rf_batch([sequence], genome_vec)[0]

    def _predict_rf_batch(self, sequences: list[str], genome_vec: np.ndarray) -> list[float]:
        rows = []
        for sequence in sequences:
            features = ifeature_descriptors(sequence)
            for i, v in enumerate(genome_vec):
                features[f"genome_{i}"] = float(v)
            rows.append([features[col] for col in self.rf_feature_cols])
        batch_df = pd.DataFrame(rows, columns=self.rf_feature_cols)
        return [float(v) for v in self.rf_model.predict(batch_df)]

    def predict_log_mic(
        self,
        sequence: str,
        organism: str,
        feature_extractor: FeatureExtractor,
    ) -> float:
        """Predicted log10(MIC, uM) against `organism`, averaged over the
        3-model ensemble. Convert with MIC_uM = 10 ** result."""
        _validate_sequence(sequence)
        _validate_organism(organism)
        self._load()
        sequence = sequence.upper()
        genome_vec = np.array(self.genome_vectors[organism], dtype=np.float32)
        bilstm_pred, cnn_pred = self._predict_bilstm_cnn(sequence, genome_vec, feature_extractor)
        rf_pred = self._predict_rf(sequence, genome_vec)
        return float(np.mean([bilstm_pred, cnn_pred, rf_pred]))

    def predict_log_mic_batch(
        self,
        sequences: list[str],
        organism: str,
        feature_extractor: FeatureExtractor,
    ) -> list[float]:
        """Batched predict_log_mic: one ESM2 forward pass for the whole batch,
        same genome vector (per `organism`) applied to every row."""
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        _validate_organism(organism)
        if not sequences:
            return []
        self._load()
        upper_sequences = [s.upper() for s in sequences]
        genome_vec = np.array(self.genome_vectors[organism], dtype=np.float32)
        bilstm_pred, cnn_pred = self._predict_bilstm_cnn_batch(
            upper_sequences, genome_vec, feature_extractor
        )
        rf_pred = self._predict_rf_batch(upper_sequences, genome_vec)
        return [
            float(np.mean([bilstm_pred[i], cnn_pred[i], rf_pred[i]]))
            for i in range(len(sequences))
        ]

    def predict_mic_um(
        self,
        sequence: str,
        organism: str,
        feature_extractor: FeatureExtractor,
    ) -> float:
        """Predicted MIC in micromolar (inverse of predict_log_mic)."""
        return float(10 ** self.predict_log_mic(sequence, organism, feature_extractor))
