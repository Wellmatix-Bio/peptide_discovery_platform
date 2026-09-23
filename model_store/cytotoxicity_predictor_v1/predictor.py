"""Architecture + loader/inference wrapper for the cytotoxicity classifier ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM_DIM = 1280

# Fixed architecture hyperparameters — identical across all 4 fold checkpoints
# (verified against each checkpoint's saved "config" dict), so hardcoded here
# rather than re-read from each checkpoint at load time.
HIDDEN_DIM = 600
D_MODEL = 120
N_HEADS = 5
DROPOUT = 0.2
LSTM_LAYERS = 1
MAX_LEN = 50
CELL_TYPE_DIM = 16

# Default cell_type context for a sequence with no known assay cell line.
# "DRAMP_aggregate" represents an unspecified/aggregate cytotoxicity source,
# as opposed to a specific cell line or the DBAASP negative-control bucket.
DEFAULT_CELL_TYPE = "DRAMP_aggregate"


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


class CytotoxicityPredictor(nn.Module):
    def __init__(
        self,
        esm_dim=ESM_DIM,
        hidden_dim=HIDDEN_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        lstm_layers=LSTM_LAYERS,
        max_len=MAX_LEN,
        n_cell_types=7,
        cell_type_dim=CELL_TYPE_DIM,
    ):
        super().__init__()
        self.max_len = max_len
        self.local_proj = nn.Linear(esm_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.local_out = nn.Linear(hidden_dim, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_mlp = nn.Sequential(
            nn.Linear(esm_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cell_type_embedding = nn.Embedding(n_cell_types, cell_type_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model + 1 + cell_type_dim, 1)

    def forward(self, residue_emb, residue_mask, seq_emb, lengths, cell_type_id, feature_mask=None):
        x = residue_emb.masked_fill(feature_mask, 0.0) if feature_mask is not None else residue_emb
        x = self.local_proj(x)
        x, _ = self.lstm(x)
        x = self.local_out(x)
        h_seq, _ = self.self_attn(x, x, x, key_padding_mask=residue_mask)
        h_seq = self.dropout(h_seq)
        g = self.global_mlp(seq_emb)
        attended, attn_weights = self.cross_attn(g.unsqueeze(1), h_seq, h_seq, key_padding_mask=residue_mask)
        y = self.dropout(g + attended.squeeze(1))
        length_feat = lengths.float().unsqueeze(1) / self.max_len
        cell_type_feat = self.cell_type_embedding(cell_type_id)
        logit = self.head(torch.cat([y, length_feat, cell_type_feat], dim=1)).squeeze(-1)
        return logit, attn_weights.squeeze(1)


class CytotoxicityClassifier:
    """Lazy-loaded 4-fold ensemble. Returns a mammalian-cell cytotoxicity
    probability in [0, 1] (sigmoid already applied) — see README.md for the
    cell_type context requirement."""

    def __init__(self, model_dir: Path = MODEL_DIR, use_feature_cache: bool = False):
        self.model_dir = Path(model_dir)
        self.use_feature_cache = use_feature_cache
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.use_feature_cache:
            self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
            self.esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(self.device).eval()
        self.models = []
        self.cell_type_to_id = None
        for checkpoint_path in sorted(self.model_dir.glob("cytotoxicity_predictor_fold*.pt")):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if self.cell_type_to_id is None:
                self.cell_type_to_id = checkpoint["cell_type_to_id"]
            model = CytotoxicityPredictor(n_cell_types=len(checkpoint["cell_type_to_id"])).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models.append(model)
        if not self.models:
            raise FileNotFoundError(f"No cytotoxicity checkpoints found in {self.model_dir}")
        self.loaded = True

    def _embed(self, sequence: str, feature_extractor: FeatureExtractor | None = None):
        if self.use_feature_cache:
            embedding = feature_extractor.get_esm2_embedding(sequence)
            with torch.no_grad():
                # Cached hidden_states are already trimmed to this sequence's
                # own valid length (BOS + residues + EOS, no padding) --
                # strip BOS/EOS the same way the original per-call tokenize did.
                residues = torch.from_numpy(embedding.hidden_states[1:-1]).to(self.device)
        else:
            with torch.no_grad():
                enc = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
                hidden = self.esm_model(**enc).last_hidden_state[0]
                n_tok = int(enc["attention_mask"][0].sum().item())
                residues = hidden[1 : n_tok - 1]
        with torch.no_grad():
            lengths = torch.tensor([residues.shape[0]], device=self.device)
            residue_emb = residues.unsqueeze(0)
            residue_mask = torch.zeros(1, residues.shape[0], dtype=torch.bool, device=self.device)
            seq_emb = residues.mean(dim=0, keepdim=True)
        return residue_emb, residue_mask, seq_emb, lengths

    def predict_cytotoxicity(
        self,
        sequence: str,
        feature_extractor: FeatureExtractor | None = None,
        cell_type: str = DEFAULT_CELL_TYPE,
    ) -> float:
        _validate_sequence(sequence)
        self._load()
        if cell_type not in self.cell_type_to_id:
            raise ValueError(
                f"Unknown cell_type {cell_type!r}. Known: {sorted(self.cell_type_to_id)}"
            )
        cell_type_id = torch.tensor(
            [self.cell_type_to_id[cell_type]], dtype=torch.long, device=self.device
        )
        residue_emb, residue_mask, seq_emb, lengths = self._embed(sequence, feature_extractor)
        with torch.no_grad():
            probs = [
                torch.sigmoid(model(residue_emb, residue_mask, seq_emb, lengths, cell_type_id)[0]).item()
                for model in self.models
            ]
        return float(np.clip(np.mean(probs), 0.0, 1.0))
