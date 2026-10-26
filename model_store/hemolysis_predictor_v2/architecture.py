"""The multi-task network of the hemolysis ensemble, copied verbatim from the training project's models.py.

Training-only methods (fit, _loss, save) are kept so the class stays identical to what produced network.pt.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

ESM_NAME = "facebook/esm2_t30_150M_UR50D"
SEED = 42
HEMOLYTIC_PHC50 = 4.0  # HC50 = 100 uM


def hemolytic_label(phc50):
    """1 where HC50 < 100 uM (pHC50 > 4), else 0."""
    return (np.asarray(phc50, dtype=np.float64) > HEMOLYTIC_PHC50).astype(np.float32)


class LocalGlobalMultiTaskPredictor(nn.Module):
    """Input: list of (L, 640) residue embeddings. Output heads: pHC50 (Huber) and hemolytic (BCE), loss 0.5 + 0.5.

        projection 640 -> d  |  local: BiLSTM (d/2 per direction)  |  global: masked mean -> MLP
        fusion: 2-head cross-attention (global query, local keys/values) -> concat(global, attended) -> 32 -> 2

    `predict` -> pHC50, `predict_proba` -> P(hemolytic), `predict_both` -> (n, 2) [pHC50, P(hemolytic)].
    The seed is set before the layers are built, so equal seeds give equal initial weights.
    """

    TASK_WEIGHTS = (0.5, 0.5)  # (pHC50, hemolytic)
    LEGACY_NAMES = ('LocalGlobalMultiTask',)  # class name written by earlier saved files

    def __init__(self, d_in=640, d=32, heads=2, dropout=0.3, head_hidden=32, lr=1e-3, weight_decay=1e-2,
                 epochs=200, batch_size=64, patience=50, huber_delta=1.0, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.config = dict(d_in=d_in, d=d, heads=heads, dropout=dropout, head_hidden=head_hidden, lr=lr,
                           weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, patience=patience,
                           huber_delta=huber_delta, seed=seed)
        self.lr, self.weight_decay, self.epochs = lr, weight_decay, epochs
        self.batch_size, self.patience, self.huber_delta, self.seed = batch_size, patience, huber_delta, seed
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.Dropout(dropout))
        self.lstm = nn.LSTM(d, d // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.global_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, head_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, 2))

    @staticmethod
    def _collate(embs):
        return pad_sequence(list(embs), batch_first=True), torch.tensor([len(e) for e in embs])

    def forward(self, x, lengths):
        """x: (B, Lmax, 640) zero-padded; lengths: (B,). Returns (B, 2): [pHC50, hemolysis logit]."""
        mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]  # True = real residue
        m = mask.unsqueeze(-1).to(x.dtype)
        z = self.proj(x)
        packed = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        local, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=x.shape[1])
        g = self.global_mlp((z * m).sum(1) / m.sum(1))
        attended, _ = self.attn(g.unsqueeze(1), local, local, key_padding_mask=~mask)
        return self.head(torch.cat([g, attended.squeeze(1)], dim=-1))

    # ---- training
    @staticmethod
    def _targets(phc50):
        phc50 = np.asarray(phc50, dtype=np.float32)
        return torch.tensor(np.stack([phc50, hemolytic_label(phc50)], axis=1), dtype=torch.float32)

    def _loss(self, out, target, weight=None):
        huber = nn.functional.huber_loss(out[:, 0], target[:, 0], delta=self.huber_delta, reduction='none')
        bce = nn.functional.binary_cross_entropy_with_logits(out[:, 1], target[:, 1], reduction='none')
        loss = self.TASK_WEIGHTS[0] * huber + self.TASK_WEIGHTS[1] * bce
        return loss.mean() if weight is None else (loss * weight).sum() / weight.sum()

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, val_weight=None):
        """Train on residue embeddings `X` and pHC50 values `y` (the hemolytic label is derived from `y`).
        With a validation set, the best epoch is kept and training stops after `patience` epochs without
        improvement. `sample_weight` / `val_weight` weight each row's loss (e.g. 1/sqrt(cluster size))."""
        rng = np.random.default_rng(self.seed)
        target = self._targets(y)
        target_val = self._targets(y_val) if X_val is not None else None
        tensor = lambda a: None if a is None else torch.tensor(np.asarray(a), dtype=torch.float32)
        w, w_val = tensor(sample_weight), tensor(val_weight)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss, best_state, bad_epochs, self.best_epoch_ = np.inf, None, 0, 0
        for epoch in range(self.epochs):
            self.train()
            order = rng.permutation(len(X))
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) < 2:  # ignore a trailing single-row batch
                    continue
                optimizer.zero_grad()
                loss = self._loss(self(*self._collate([X[j] for j in idx])), target[idx], None if w is None else w[idx])
                loss.backward()
                optimizer.step()
            if X_val is None:
                continue
            val_loss = self._loss(self._forward_batches(X_val), target_val, w_val).item()
            if val_loss < best_loss - 1e-4:
                best_loss, best_state, bad_epochs, self.best_epoch_ = val_loss, copy.deepcopy(self.state_dict()), 0, epoch
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        return self

    # ---- prediction
    @torch.no_grad()
    def _forward_batches(self, X):
        self.eval()
        return torch.cat([self(*self._collate([X[j] for j in range(s, min(s + self.batch_size, len(X)))]))
                          for s in range(0, len(X), self.batch_size)])

    def predict(self, X) -> np.ndarray:
        """pHC50."""
        return self._forward_batches(X)[:, 0].numpy()

    def predict_proba(self, X) -> np.ndarray:
        """P(HC50 < 100 uM)."""
        return torch.sigmoid(self._forward_batches(X)[:, 1]).numpy()

    def predict_both(self, X) -> np.ndarray:
        """(n, 2): [pHC50, P(hemolytic)]."""
        out = self._forward_batches(X)
        return torch.stack([out[:, 0], torch.sigmoid(out[:, 1])], dim=1).numpy()

    # ---- persistence
    def save(self, path, meta=None) -> Path:
        """Weights + constructor config (+ plain-Python metadata) in one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'class': type(self).__name__, 'config': self.config, 'state_dict': self.state_dict(),
                    'meta': dict(meta or {}, esm_model=ESM_NAME)}, path)
        return path

    @classmethod
    def load(cls, path) -> 'LocalGlobalMultiTaskPredictor':
        ckpt = torch.load(Path(path), map_location='cpu', weights_only=True)
        if ckpt['class'] not in (cls.__name__, *cls.LEGACY_NAMES):
            raise ValueError(f"{path} holds a {ckpt['class']}, not a {cls.__name__}")
        model = cls(**ckpt['config'])
        model.load_state_dict(ckpt['state_dict'])
        model.meta = ckpt['meta']
        return model.eval()
