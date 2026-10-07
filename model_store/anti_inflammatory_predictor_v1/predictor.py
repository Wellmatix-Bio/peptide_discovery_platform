"""Architecture + loader/inference wrapper for the DAC-AIPs anti-inflammatory classifier."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model_sync import sync_model_weights, weights_dir_for

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AMINO_ACIDS = len(AMINO_ACIDS)

# Fixed architecture/preprocessing hyperparameters, taken from the checkpoint's config dict
SEQ_LEN = 25
LATENT_DIM = 208
CLS_HIDDEN_DIM = 64
NUM_CLASSES = 2
K_VALUES = (1, 2, 3)


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def encode_sequence(
    sequence: str, max_len: int = SEQ_LEN, k_values=K_VALUES
) -> torch.Tensor:
    """Fused one-hot (k=1) + multi-hot (k=2, k=3) sliding-window encoding; truncated to max_len, zero-padded on the right."""
    seq = sequence[:max_len]
    length = len(seq)

    channels = []
    for k in k_values:
        vec = torch.zeros(max_len, NUM_AMINO_ACIDS)
        half = k // 2
        for pos in range(length):
            start = max(0, pos - half)
            end = min(length, pos + (k - half))
            window = seq[start:end]
            for aa in window:
                idx = AA_TO_IDX.get(aa)
                if idx is not None:
                    vec[pos, idx] = 1.0
        channels.append(vec)

    fused = torch.cat(channels, dim=1)
    return fused.transpose(0, 1).contiguous()


def encode_batch(sequences, max_len: int = SEQ_LEN, k_values=K_VALUES) -> torch.Tensor:
    return torch.stack(
        [encode_sequence(s, max_len, k_values) for s in sequences], dim=0
    )


class Encoder(nn.Module):
    """Conv1d -> MaxPool1d -> Conv1d -> MaxPool1d -> Linear -> (mu, logvar)."""

    def __init__(self, in_channels: int, seq_len: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            flat_dim = self._conv_forward(dummy).shape[1]

        self.flat_dim = flat_dim
        self.fc1 = nn.Linear(flat_dim, flat_dim // 2)
        self.fc_mu = nn.Linear(flat_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim // 2, latent_dim)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor):
        h = self._conv_forward(x)
        h = F.relu(self.fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h.shape[1]


class Decoder(nn.Module):
    """Linear -> Linear -> Upsample -> ConvTranspose1d -> Upsample -> ConvTranspose1d. Outputs raw logits."""

    def __init__(self, latent_dim: int, flat_dim: int, out_channels: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.out_channels = out_channels

        self.fc1 = nn.Linear(latent_dim, flat_dim // 2)
        self.fc2 = nn.Linear(flat_dim // 2, flat_dim)

        self.pre_channels = 64
        self.pre_len = flat_dim // self.pre_channels

        self.upsample1 = nn.Upsample(size=self.pre_len + 1, mode="nearest")
        self.deconv1 = nn.ConvTranspose1d(64, 128, kernel_size=2)
        self.upsample2 = nn.Upsample(size=self.pre_len + 3, mode="nearest")
        self.deconv2 = nn.ConvTranspose1d(128, out_channels, kernel_size=2)

        self.output_resize = nn.Upsample(size=seq_len, mode="nearest")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = h.view(-1, self.pre_channels, self.pre_len)

        h = self.upsample1(h)
        h = F.relu(self.deconv1(h))
        h = self.upsample2(h)
        h = self.deconv2(h)

        h = self.output_resize(h)
        return h


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ClassificationHead(nn.Module):
    """Two fully connected layers, applied to latent features. Softmax applied by the caller."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = CLS_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        logits = self.fc2(h)
        return logits


class DACAIPs(nn.Module):
    """Deep variational Autoencoder + Contrastive learning for anti-inflammatory peptide identification."""

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        in_channels: int = NUM_AMINO_ACIDS * 3,
        latent_dim: int = LATENT_DIM,
        cls_hidden_dim: int = CLS_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, seq_len, latent_dim)
        self.decoder = Decoder(latent_dim, self.encoder.flat_dim, in_channels, seq_len)
        self.classifier = ClassificationHead(latent_dim, cls_hidden_dim, num_classes)

    def encode(self, x: torch.Tensor):
        mu, logvar, _ = self.encoder(x)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encode(x)
        x_rec_logits = self.decoder(z)
        logits = self.classifier(z)
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "x_rec": torch.sigmoid(x_rec_logits),
            "x_rec_logits": x_rec_logits,
            "logits": logits,
        }


class AntiInflammatoryPredictor:
    """Lazy-loaded classifier returning the calibrated anti-inflammatory probability"""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = self.model_dir / "dac_aips_final.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.config = checkpoint["config"]
        self.model = DACAIPs(seq_len=int(self.config["seq_len"])).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.loaded = True

    def predict_proba(self, sequence: str) -> float:
        _validate_sequence(sequence)
        self._load()
        x = encode_batch([sequence], max_len=int(self.config["seq_len"])).to(
            self.device
        )
        with torch.no_grad():
            logits = self.model(x)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
        return float(probs.item())
