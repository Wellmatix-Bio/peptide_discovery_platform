"""Loader/inference wrapper for the anti-inflammatory peptide classifier v2: frozen ESM2-150M, three branches and a fusion head."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model_sync import sync_model_weights, weights_dir_for

if TYPE_CHECKING:  # annotation only; a runtime import would be circular with the Stage 6 import of this package
    from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
ESM_DIM = 640
WIN_LEN = 20  # longest window (terminal-20)
N_SLOTS = 10  # 4 terminal windows + up to 6 cleavage-centred windows
N_WINDOWS = (10, 20)
C_WINDOWS = (10, 20)
CLEAVAGE_RADIUS = 5
MAX_CLEAVAGE_WINDOWS = 6
MOTIFS = ("KR", "RR", "RK", "KK")
WINDOW_TYPES = (0,) * len(N_WINDOWS) + (1,) * len(C_WINDOWS) + (2,) * MAX_CLEAVAGE_WINDOWS  # 0 N-term, 1 C-term, 2 cleavage
BATCH = 64

# Observed training length range (8 is the shortest augmented crop); outside it the score is extrapolation, flagged but not rejected.
MIN_LENGTH, MAX_LENGTH = 8, 30


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    bad = sorted(set(sequence) - set(AA))
    if bad:
        raise ValueError(f"sequence contains residues outside the 20 uppercase standard amino acids: {bad}")


def in_trained_length_range(sequence: str) -> bool:
    return MIN_LENGTH <= len(sequence) <= MAX_LENGTH


# ---------------------------------------------------------------- windows

def find_cleavage_positions(seq: str) -> list[int]:
    """Every K/R position plus the centre of each KR/RR/RK/KK motif, sorted."""
    pos = {i for i, aa in enumerate(seq) if aa in "KR"}
    for motif in MOTIFS:
        for m in re.finditer(motif, seq):
            pos.add(m.start() + (len(motif) - 1) // 2)
    return sorted(pos)


def get_windows(seq: str) -> list[str]:
    """N-terminal (10, 20), C-terminal (10, 20), then up to 6 cleavage-centred windows (radius 5); slots beyond these stay empty."""
    wins = [seq[: min(k, len(seq))] for k in N_WINDOWS]
    wins += [seq[max(0, len(seq) - k):] for k in C_WINDOWS]
    for p in find_cleavage_positions(seq)[:MAX_CLEAVAGE_WINDOWS]:
        wins.append(seq[max(0, p - CLEAVAGE_RADIUS): min(len(seq), p + CLEAVAGE_RADIUS + 1)])
    return wins


# ------------------------------------------------------ handcrafted features

_CTD_PROPERTIES = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "normwaalsvolume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondarystruct": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solventaccess": ("ALFCGIVW", "RKQEND", "MPSTHY"),
}
_HYDRO = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                       1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
_MW = dict(zip(AA, [89.09, 121.15, 133.10, 147.13, 165.19, 75.07, 155.16, 131.17, 146.19, 131.17,
                    149.21, 132.12, 115.13, 146.15, 174.20, 105.09, 119.12, 117.15, 204.23, 181.19]))
_PKA_POS = {"K": 10.5, "R": 12.5, "H": 6.0}
_PKA_NEG = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
_PKA_NTERM, _PKA_CTERM = 9.69, 2.34


def _aac(seq):
    v = np.zeros(20, dtype=np.float32)
    for aa in seq:
        v[AA_IDX[aa]] += 1
    return v / max(len(seq), 1)


def _dpc(seq):
    v = np.zeros(400, dtype=np.float32)
    if len(seq) < 2:
        return v
    for i in range(len(seq) - 1):
        v[AA_IDX[seq[i]] * 20 + AA_IDX[seq[i + 1]]] += 1
    return v / (len(seq) - 1)


def _paac(seq, lamda=3, w=0.05):
    freq = _aac(seq)
    if len(seq) < 2:
        return np.concatenate([freq, np.zeros(lamda, dtype=np.float32)])
    theta = np.array([np.mean([(_HYDRO[seq[i]] - _HYDRO[seq[i + lam]]) ** 2 for i in range(len(seq) - lam)])
                      if len(seq) > lam else 0.0 for lam in range(1, lamda + 1)], dtype=np.float32)
    denom = 1.0 + w * theta.sum()
    return np.concatenate([freq / denom, (w * theta) / denom]).astype(np.float32)


def _ctd(seq):
    """Composition, transition and distribution: 21 values for each of 7 property groupings."""
    L = len(seq)
    feats = []
    for groups in _CTD_PROPERTIES.values():
        mapped = ["123"[next((g for g, mem in enumerate(groups) if aa in mem), 0)] for aa in seq]
        feats += [mapped.count(g) / L for g in "123"]
        counts = {"12": 0, "13": 0, "23": 0}
        for i in range(L - 1):
            pair = "".join(sorted(mapped[i] + mapped[i + 1]))
            if pair in counts:
                counts[pair] += 1
        feats += [counts[k] / max(L - 1, 1) for k in ("12", "13", "23")]
        for g in "123":
            positions = [i + 1 for i, x in enumerate(mapped) if x == g]
            if not positions:
                feats += [0.0] * 5
            else:
                feats += [positions[0] / L] + [
                    positions[max(math.ceil(len(positions) * p / 100.0) - 1, 0)] / L for p in (25, 50, 75, 100)]
    return feats


def _net_charge(seq, pH=7.0):
    pos = 1 / (1 + 10 ** (pH - _PKA_NTERM))
    neg = 1 / (1 + 10 ** (_PKA_CTERM - pH))
    for aa in seq:
        if aa in _PKA_POS:
            pos += 1 / (1 + 10 ** (pH - _PKA_POS[aa]))
        elif aa in _PKA_NEG:
            neg += 1 / (1 + 10 ** (_PKA_NEG[aa] - pH))
    return pos - neg


def _estimate_pi(seq):
    low, high = 0.0, 14.0
    for _ in range(60):
        mid = (low + high) / 2
        if _net_charge(seq, mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _gravy(seq):
    return float(np.mean([_HYDRO[aa] for aa in seq])) if seq else 0.0


def _physicochemical(seq):
    L = len(seq)
    hydros = np.array([_HYDRO[aa] for aa in seq], dtype=np.float32)
    counts = {aa: seq.count(aa) for aa in "AVIL"}
    return np.array([
        L, sum(_MW[aa] for aa in seq), _estimate_pi(seq), _net_charge(seq, 7.0), float(hydros.mean()), float(hydros.std()), _gravy(seq),
        sum(aa in "FWY" for aa in seq) / L,
        (counts["A"] + 2.9 * counts["V"] + 3.9 * (counts["I"] + counts["L"])) / L * 100,
        sum(aa in "PGNQSEDKRH" for aa in seq) / L * 100,
        sum(aa in "DE" for aa in seq) / L, sum(aa in "KRH" for aa in seq) / L,
        sum(aa in "STNQCYWHKRED" for aa in seq) / L, sum(aa == "C" for aa in seq) / L,
    ], dtype=np.float32)


def _biological(seq):
    """22 cleavage-site and terminus features."""
    L = len(seq)
    cleav = find_cleavage_positions(seq)
    if len(cleav) >= 2:
        iv = np.diff(cleav)
        avg_iv, min_iv, max_iv = float(iv.mean()), float(iv.min()), float(iv.max())
    else:
        avg_iv = min_iv = max_iv = 0.0
    min_to_n = float(min(cleav)) if cleav else 0.0
    min_to_c = float(min(L - 1 - p for p in cleav)) if cleav else 0.0
    hydrophobic = set("AILMFWV")
    n10, n15, n20 = seq[:10], seq[:15], seq[:20]
    longest = cur = 0
    for aa in n20:
        cur = cur + 1 if aa in hydrophobic else 0
        longest = max(longest, cur)
    return np.array([
        seq.count("K") / L, seq.count("R") / L, (seq.count("K") + seq.count("R")) / L,
        len(re.findall("KR", seq)) / max(L - 1, 1), len(re.findall("RR", seq)) / max(L - 1, 1),
        len(re.findall("RK", seq)) / max(L - 1, 1), len(re.findall("KK", seq)) / max(L - 1, 1),
        len(cleav) / L, avg_iv / max(L, 1), min_iv / max(L, 1), max_iv / max(L, 1),
        min_to_n / max(L, 1), min_to_c / max(L, 1),
        1.0 if seq[-1] == "G" else 0.0, 1.0 if L >= 2 and seq[-2] == "G" else 0.0, 1.0 if seq[0] in "QE" else 0.0,
        sum(aa in "KR" for aa in seq[-3:]) / min(3, L),
        sum(aa in hydrophobic for aa in n10) / max(len(n10), 1), sum(aa in hydrophobic for aa in n15) / max(len(n15), 1),
        sum(aa in "KRH" for aa in n10) / max(len(n10), 1), _gravy(n15), longest / 20.0,
    ], dtype=np.float32)


def handcrafted_features(seq: str) -> np.ndarray:
    """626 descriptors: AAC(20) + DPC(400) + CTD(147) + PseAAC(23) + physicochemical(14) + biological(22)."""
    return np.concatenate([_aac(seq), _dpc(seq), _ctd(seq), _paac(seq), _physicochemical(seq), _biological(seq)]).astype(np.float32)


# ------------------------------------------------------------------ network

class AttnPool(nn.Module):
    """e_i = w^T tanh(W h_i + b); alpha = softmax(e) over valid positions; returns sum_i alpha_i h_i."""

    def __init__(self, dim, att_dim=128):
        super().__init__()
        self.W = nn.Linear(dim, att_dim)
        self.w = nn.Linear(att_dim, 1, bias=False)

    def forward(self, h, mask):
        e = self.w(torch.tanh(self.W(h))).squeeze(-1)
        alpha = torch.softmax(e.masked_fill(~mask, -1e4), dim=-1)
        return (alpha.unsqueeze(-1) * h).sum(1)


class GlobalBranch(nn.Module):
    """BOS + mean + max + attention pooling over residue embeddings."""

    def __init__(self, d_in, d_out, p, att_dim=128):
        super().__init__()
        self.att = AttnPool(d_in, att_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * d_in, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(p),
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, d_out))

    def forward(self, bos, h, mask):
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(~m, -1e4).max(1).values
        return self.net(torch.cat([bos, mean, mx, self.att(h, mask)], -1))


class LocalBranch(nn.Module):
    """Per-window CNN (k=3, 5) with attention pooling, window-type embedding, then a Transformer over the windows."""

    def __init__(self, d_in, d_out, p, ch=128, out_dim=256, type_dim=32, layers=2, heads=4, ffn=512):
        super().__init__()
        self.c3 = nn.Conv1d(d_in, ch, 3, padding=1)
        self.c5 = nn.Conv1d(d_in, ch, 5, padding=2)
        self.bn = nn.BatchNorm1d(2 * ch)
        self.res_att = AttnPool(2 * ch, 128)
        self.fc = nn.Sequential(nn.Linear(2 * ch, out_dim), nn.GELU(), nn.Dropout(p))
        self.type_emb = nn.Embedding(3, type_dim)
        self.pre_tf = nn.Linear(out_dim + type_dim, out_dim)
        layer = nn.TransformerEncoderLayer(out_dim, heads, ffn, p, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)  # pre-norm layers never use the nested path
        self.win_att = AttnPool(out_dim, 128)
        self.out = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU(), nn.Dropout(p), nn.Linear(out_dim, d_out))
        self.register_buffer("wtypes", torch.tensor(WINDOW_TYPES), persistent=False)

    def forward(self, w, rmask, wmask):
        B, S, L, D = w.shape
        x = w.reshape(B * S, L, D).transpose(1, 2)
        x = self.bn(torch.cat([F.gelu(self.c3(x)), F.gelu(self.c5(x))], 1)).transpose(1, 2)
        rm = rmask.reshape(B * S, L).clone()
        rm[:, 0] = True  # an empty slot must not give an all-masked softmax
        v = self.fc(self.res_att(x, rm)).reshape(B, S, -1)
        v = self.pre_tf(torch.cat([v, self.type_emb(self.wtypes).unsqueeze(0).expand(B, -1, -1)], -1))
        v = self.tr(v, src_key_padding_mask=~wmask)
        return self.out(self.win_att(v, wmask))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(self.norm(x))))
        return x + self.drop(self.fc2(h))


class HandcraftedBranch(nn.Module):
    def __init__(self, d_in, d_out, p, res_dim=256, hidden=128):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(d_in, res_dim), nn.LayerNorm(res_dim), nn.GELU(), nn.Dropout(p))
        self.res = ResidualMLPBlock(res_dim, p)
        self.out_proj = nn.Sequential(nn.Linear(res_dim, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.out_proj(self.res(self.in_proj(x)))


class FusionNet(nn.Module):
    """Global, local and handcrafted branches fused as [g, l, h, g*l, g*h, l*h, |g-l|, |g-h|, |l-h|] -> classifier logit."""

    def __init__(self, d_hand=80, fused_dim=128, p=0.35):
        super().__init__()
        self.g = GlobalBranch(ESM_DIM, fused_dim, p)
        self.l = LocalBranch(ESM_DIM, fused_dim, p)
        self.h = HandcraftedBranch(d_hand, fused_dim, p)
        self.classifier = nn.Sequential(
            nn.Linear(9 * fused_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(p),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(p), nn.Linear(128, 1))

    def forward(self, bos, hg, mg, hw, rmask, wmask, xh):
        g, l, h = self.g(bos, hg, mg), self.l(hw, rmask, wmask), self.h(xh)
        feats = [g, l, h, g * l, g * h, l * h, (g - l).abs(), (g - h).abs(), (l - h).abs()]
        return self.classifier(torch.cat(feats, -1)).squeeze(-1)


# ---------------------------------------------------------------- predictor

class AntiInflammatoryPredictor:
    """Lazy-loaded; returns P(anti-inflammatory) per peptide. Deterministic: dropout is off and nothing is sampled."""

    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        bundle = torch.load(self.model_dir / "weights.pt", map_location="cpu", weights_only=True)
        self.feature_index = bundle["feature_index"].numpy()
        self.feature_mean = bundle["feature_mean"].numpy()
        self.feature_scale = bundle["feature_scale"].numpy()
        self.model = FusionNet(d_hand=len(self.feature_index))
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device).eval()
        self.loaded = True

    def _batch(self, seqs: list[str], embeddings: dict[str, np.ndarray]):
        """Padded tensors for one chunk: BOS, residues, residue mask, windows, window residue mask, window mask, handcrafted."""
        B, L = len(seqs), max(len(s) for s in seqs)
        bos = torch.zeros(B, ESM_DIM)
        hg = torch.zeros(B, L, ESM_DIM)
        mg = torch.zeros(B, L, dtype=torch.bool)
        hw = torch.zeros(B, N_SLOTS, WIN_LEN, ESM_DIM)
        rmask = torch.zeros(B, N_SLOTS, WIN_LEN, dtype=torch.bool)
        wmask = torch.zeros(B, N_SLOTS, dtype=torch.bool)
        for i, s in enumerate(seqs):
            hidden = embeddings[s]
            bos[i] = torch.from_numpy(hidden[0])
            hg[i, : len(s)] = torch.from_numpy(hidden[1:-1])
            mg[i, : len(s)] = True
            for j, w in enumerate(get_windows(s)):
                wh = embeddings[w][1:-1]
                hw[i, j, : len(w)] = torch.from_numpy(wh)
                rmask[i, j, : len(w)] = True
                wmask[i, j] = True
        x = np.stack([handcrafted_features(s) for s in seqs])[:, self.feature_index]
        xh = torch.from_numpy(((x - self.feature_mean) / self.feature_scale).astype(np.float32))
        return [t.to(self.device) for t in (bos, hg, mg, hw, rmask, wmask, xh)]

    @torch.no_grad()
    def predict_proba_batch(self, sequences: list[str], feature_extractor: FeatureExtractor) -> list[float]:
        """P(anti-inflammatory) per sequence; sequences outside the trained length range are scored with a warning."""
        if not sequences:
            return []
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        out_of_range = [i for i, s in enumerate(sequences) if not in_trained_length_range(s)]
        if out_of_range:
            warnings.warn(
                f"{len(out_of_range)} sequence(s) outside the trained length range {MIN_LENGTH}-{MAX_LENGTH} "
                f"(indices {out_of_range[:10]}); scores are extrapolation.",
                stacklevel=2,
            )
        self._load()
        probabilities: list[float] = []
        for start in range(0, len(sequences), BATCH):
            chunk = sequences[start: start + BATCH]
            needed = sorted({t for s in chunk for t in [s, *get_windows(s)]})
            embeddings = dict(zip(needed, (e.hidden_states for e in feature_extractor.get_esm2_embedding_batch(needed))))
            probabilities += torch.sigmoid(self.model(*self._batch(chunk, embeddings))).cpu().tolist()
        return [float(p) for p in probabilities]

    def predict_batch(self, sequences: list[str], feature_extractor: FeatureExtractor) -> list[dict]:
        """P(anti-inflammatory) per sequence with `in_length_range`, so callers can carry the extrapolation flag."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probabilities = self.predict_proba_batch(sequences, feature_extractor)
        return [{"probability": p, "in_length_range": in_trained_length_range(s)} for p, s in zip(probabilities, sequences)]

    def predict_proba(self, sequence: str, feature_extractor: FeatureExtractor) -> float:
        return self.predict_proba_batch([sequence], feature_extractor)[0]
