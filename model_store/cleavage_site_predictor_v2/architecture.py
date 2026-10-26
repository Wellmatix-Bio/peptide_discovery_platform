"""UniZyme architecture (replicated from code/predict.py so state-dict keys match UniZyme.pt)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Exact (erf) gelu, as used by the original feed-forward block."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GaussianParameters(nn.Module):
    def __init__(self, K=10):
        super().__init__()
        self.K = K
        self.mu_D_Enzyme = nn.Parameter(torch.randn(K))
        self.sigma_D_Enzyme = nn.Parameter(torch.ones(K))
        self.b_D_Enzyme = nn.Parameter(torch.zeros(K))

        self.mu_D_Sub = nn.Parameter(torch.randn(K))
        self.sigma_D_Sub = nn.Parameter(torch.ones(K))
        self.b_D_Sub = nn.Parameter(torch.zeros(K))

        self.mu_E = nn.Parameter(torch.randn(K))
        self.sigma_E = nn.Parameter(torch.ones(K))
        self.b_E = nn.Parameter(torch.zeros(K))

        self.W_D1 = nn.Linear(K, K)
        self.W_D2 = nn.Linear(K, 1)

        self.mu_activation = nn.Parameter(torch.randn(K))
        self.sigma_activation = nn.Parameter(torch.ones(K))
        self.b_activation = nn.Parameter(torch.zeros(K))

        self.W_activation1 = nn.Linear(K, K)
        self.W_activation2 = nn.Linear(K, 1)


def _gaussian_kernel(x, mu, sigma):
    psi = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return psi / (torch.sqrt(torch.tensor(2 * math.pi, device=x.device)) * sigma)


class Pool_Cat_MLP(nn.Module):
    def __init__(self, d, gaussian_params):
        super().__init__()
        self.d = d
        # Unused in forward, kept so the checkpoint's `mlp.*` keys load.
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(d, 64),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.conv1d = nn.Conv1d(in_channels=2 * d, out_channels=128, kernel_size=31, stride=1, padding=15)
        self.conv_activation = nn.SELU()
        self.conv_dropout = nn.Dropout(p=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.gaussian_params = gaussian_params

    def forward(self, Enzyme, Substrate, Activate_Site, Activate_Site_f, Enzymes_mask):
        gp = self.gaussian_params
        Enzymes_mask = Enzymes_mask[:, 0, :, 0]
        Enzymes_mask = Enzymes_mask.unsqueeze(-1).repeat(1, 1, self.d)

        Activate_Site_f = Activate_Site_f.repeat(1, 1, gp.K) + gp.b_activation
        psi = _gaussian_kernel(Activate_Site_f, gp.mu_activation, gp.sigma_activation)
        psi = F.gelu(gp.W_activation1(psi))
        psi = gp.W_activation2(psi)

        flag = psi * Enzymes_mask
        flag = flag.masked_fill(flag == 0.0, -1e9)
        weights = F.softmax(flag, dim=-2)

        pooled_Enzyme = (weights * Enzyme).sum(dim=-2)
        pooled_Enzyme = pooled_Enzyme.unsqueeze(1).repeat(1, Substrate.size(1), 1)

        combined = torch.cat((Substrate, pooled_Enzyme), dim=-1).permute(0, 2, 1)
        out = self.conv_dropout(self.conv_activation(self.conv1d(combined)))
        return self.fc(out.permute(0, 2, 1))  # (batch, length_sub, 1)


class ScaledDotProductAttention_Gauss(nn.Module):
    def __init__(self, d_model, gaussian_params, K=10):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.gaussian_params = gaussian_params

    def _phi(self, matrix, mu, sigma, bias):
        gp = self.gaussian_params
        matrix = matrix.unsqueeze(-1).repeat(1, 1, 1, 1, self.K) + bias
        psi = _gaussian_kernel(matrix, mu, sigma)
        return gp.W_D2(F.gelu(gp.W_D1(psi))).squeeze(-1)

    def forward(self, Q, K, V, distance_matrix, energy_matrix, mask):
        gp = self.gaussian_params
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model * 2)

        if energy_matrix is None:
            scores = scores + self._phi(distance_matrix, gp.mu_D_Sub, gp.sigma_D_Sub, gp.b_D_Sub)
        else:
            scores = (
                scores
                + self._phi(distance_matrix, gp.mu_D_Enzyme, gp.sigma_D_Enzyme, gp.b_D_Enzyme)
                + self._phi(energy_matrix, gp.mu_E, gp.sigma_E, gp.b_E)
            )

        scores = scores.masked_fill(mask == 0, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        return torch.matmul(attn, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, gaussian_params):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model * 2 * n_heads)
        self.W_K = nn.Linear(d_model, d_model * 2 * n_heads)
        self.W_V = nn.Linear(d_model, d_model * 2 * n_heads)
        self.d_model = d_model
        self.n_heads = n_heads
        self.liner = nn.Linear(n_heads * self.d_model * 2, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.gaussian_params = gaussian_params

    def forward(self, Q, K, V, distance_matrix, energy_matrix, mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1, 2)

        # built per call (not registered) so the checkpoint's shared gaussian_params keys stay unduplicated
        attention = ScaledDotProductAttention_Gauss(self.d_model, self.gaussian_params, self.gaussian_params.K)
        context = attention(q_s, k_s, v_s, distance_matrix, energy_matrix, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model * 2)
        return self.layernorm(self.liner(context) + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, gaussian_params):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, gaussian_params=gaussian_params)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model)

    def forward(self, input_feature, distance_matrix, energy_matrix, mask):
        out = self.enc_self_attn(input_feature, input_feature, input_feature, distance_matrix, energy_matrix, mask)
        return self.pos_ffn(out)


class Transformer_DE(nn.Module):
    def __init__(self, n_layers=1, d_model=480, n_heads=2, K=10):
        super().__init__()
        self.K = K
        self.gaussian_params = GaussianParameters(K=K)
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, self.gaussian_params) for _ in range(n_layers)]
        )
        self.layers_substrate = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, self.gaussian_params) for _ in range(n_layers)]
        )
        self.Pool_Cat_MLP = Pool_Cat_MLP(d_model, self.gaussian_params)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(p=0.3),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.SELU(),
            nn.Linear(64, 1),
        )

    def encode_enzyme(self, Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask):
        """The enzyme half of forward(): encoded residue features and active-site logits, independent of the substrate."""
        for layer in self.layers:
            Enzymes_features = layer(Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask)
        return Enzymes_features, self.classifier(Enzymes_features)

    def cleave_substrates(
        self, Substrate_features, Substrate_distances, Substrate_mask, Enzymes_features, Activate_Site_f, Enzymes_mask
    ):
        """The substrate half of forward() for a batch of substrates against one encoded enzyme (batch of 1, broadcast)."""
        for layer in self.layers_substrate:
            Substrate_features = layer(Substrate_features, Substrate_distances, None, Substrate_mask)
        batch = Substrate_features.size(0)
        return self.Pool_Cat_MLP(
            Enzymes_features.expand(batch, -1, -1),
            Substrate_features,
            None,
            Activate_Site_f.expand(batch, -1, -1),
            Enzymes_mask.expand(batch, -1, -1, -1),
        )

    def forward(
        self,
        Substrate_features,
        Substrate_distances,
        Substrate_mask,
        Enzymes_features,
        Enzymes_distances,
        Enzymes_energy,
        Enzymes_mask,
    ):
        for layer in self.layers:
            Enzymes_features = layer(Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask)
        for layer in self.layers_substrate:
            Substrate_features = layer(Substrate_features, Substrate_distances, None, Substrate_mask)

        Activate_Site_f = self.classifier(Enzymes_features)
        Activate_Site = torch.sigmoid(Activate_Site_f)
        Cleavage_Site = self.Pool_Cat_MLP(
            Enzymes_features, Substrate_features, Activate_Site, Activate_Site_f, Enzymes_mask
        )
        return Activate_Site, Cleavage_Site
