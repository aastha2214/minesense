"""Neural network architectures.

All models take the raw dataset inputs, a float tensor ``x`` with columns
[V, H] and an integer soil index, so that feature engineering, augmentation and
the physics-consistency loss all act on the same physical quantities, and the
exported ONNX graph needs no separate preprocessing.
"""
import math

import numpy as np
import torch
from torch import nn

from .config import N_CLASSES, N_SOILS, NO_MINE

SATURATION = 0.999  # normalised V at or above this is the sensor's upper limit
OFFSET_GRID = np.geomspace(0.05, 20.0, 400)

BASE_FEATURES = ["V", "H"]
PHYSICS_FEATURES = ["log V", "V*r^3", "V/r", "dV*r^3", "saturated"]


def soil_factors(soil):
    """Split soil index 0..5 into moisture (0 dry, 1 humid) and texture
    (0 sandy, 1 humus, 2 limy), following the order in config.SOIL_NAMES."""
    return soil // 3, soil % 3


def fit_offset(v: np.ndarray, h: np.ndarray, soil: np.ndarray, y: np.ndarray) -> float:
    """Fit the height offset d0 in V = a + b / (H + d0)^3.

    One d0 is shared by every (class, soil) series; a and b are free per
    series. Saturated readings are excluded because their true value is
    unknown. Uses only the rows passed in, which are always training rows.
    """
    keep = v < SATURATION
    v, h, soil, y = v[keep], h[keep], soil[keep], y[keep]
    sse = np.zeros(len(OFFSET_GRID))
    for key in np.unique(y * N_SOILS + soil):
        idx = np.flatnonzero(y * N_SOILS + soil == key)
        if len(idx) < 3:
            continue
        x = (h[idx][None, :] + OFFSET_GRID[:, None]) ** -3.0
        xc = x - x.mean(1, keepdims=True)
        yc = v[idx] - v[idx].mean()
        sxx = (xc ** 2).sum(1)
        sxy = (xc * yc).sum(1)
        sse += (yc ** 2).sum() - np.where(sxx > 0, sxy ** 2 / np.maximum(sxx, 1e-30), 0.0)
    return float(OFFSET_GRID[int(sse.argmin())])


class InputFeatures(nn.Module):
    """Builds and standardises the numeric inputs.

    With ``physics=True`` it adds features motivated by the dipole law, under
    which a buried magnetic object's anomaly falls off as 1/r^3:
      r          = H + d0, an effective sensor-to-object distance
      dV         = V - b(soil), the anomaly above the soil's no-mine background
      V*r^3, dV*r^3   distance-compensated signal strength
      log V, V/r       alternative compressions of the signal
      saturated  1 if the reading is at the sensor limit
    d0, b(soil), and the standardisation statistics are fitted on training rows.
    """

    def __init__(self, physics: bool):
        super().__init__()
        self.physics = physics
        self.names = BASE_FEATURES + (PHYSICS_FEATURES if physics else [])
        self.n_out = len(self.names)
        self.register_buffer("offset", torch.tensor(1.0))
        self.register_buffer("background", torch.zeros(N_SOILS))
        self.register_buffer("mean", torch.zeros(self.n_out))
        self.register_buffer("std", torch.ones(self.n_out))

    def raw(self, v, h, soil):
        cols = [v, h]
        if self.physics:
            r = h + self.offset
            dv = v - self.background[soil]
            cols += [torch.log(v.clamp_min(1e-3)), v * r ** 3, v / r, dv * r ** 3,
                     (v >= SATURATION).to(v.dtype)]
        return torch.stack(cols, 1)

    def forward(self, v, h, soil):
        return (self.raw(v, h, soil) - self.mean) / self.std

    @torch.no_grad()
    def fit(self, v: np.ndarray, h: np.ndarray, soil: np.ndarray, y: np.ndarray) -> None:
        # d0 and b(soil) are fitted even without physics features because the
        # physics-consistency loss needs them (ablation: loss without features).
        self.offset.fill_(fit_offset(v, h, soil, y))
        no_mine = y == NO_MINE
        fallback = v[no_mine].mean()
        for s in range(N_SOILS):
            rows = no_mine & (soil == s)
            # A soil absent from training (leave-one-soil-out) gets the mean background.
            self.background[s] = float(v[rows].mean()) if rows.any() else float(fallback)
        raw = self.raw(torch.as_tensor(v), torch.as_tensor(h), torch.as_tensor(soil))
        std = raw.std(0)
        self.mean.copy_(raw.mean(0))
        self.std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))

    def shift_height(self, v, h, soil, delta):
        """Where would this reading be if the sensor were moved by ``delta``?

        The anomaly above background is rescaled by the inverse-cube law:
        V' = b + (V - b) * ((H + d0) / (H' + d0))^3, clipped to the sensor range.
        """
        h2 = (h + delta).clamp(0.0, 1.0)
        b = self.background[soil]
        v2 = b + (v - b) * ((h + self.offset) / (h2 + self.offset)) ** 3
        return v2.clamp(0.0, 1.0), h2


class SoilEncoder(nn.Module):
    """Soil type as a learned embedding.

    embed   one vector per soil type
    factor  moisture vector + texture vector, so an unseen combination of a
            seen moisture and a seen texture still has a meaningful encoding
    onehot  fixed one-hot (used to ablate the embedding)
    none    soil is ignored
    """

    def __init__(self, mode: str, dim: int):
        super().__init__()
        self.mode = mode
        if mode == "embed":
            self.table = nn.Embedding(N_SOILS, dim)
            self.out_dim = dim
        elif mode == "factor":
            self.moisture = nn.Embedding(2, dim)
            self.texture = nn.Embedding(3, dim)
            self.out_dim = dim
        elif mode == "onehot":
            self.register_buffer("eye", torch.eye(N_SOILS))
            self.out_dim = N_SOILS
        elif mode == "none":
            self.out_dim = 0
        else:
            raise ValueError(f"unknown soil mode {mode!r}")

    def forward(self, soil):
        if self.mode == "embed":
            return self.table(soil)
        if self.mode == "factor":
            moisture, texture = soil_factors(soil)
            return self.moisture(moisture) + self.texture(texture)
        if self.mode == "onehot":
            return self.eye[soil]
        return torch.zeros(soil.shape[0], 0, device=soil.device)

    @torch.no_grad()
    def fill_unseen(self, seen: list[int]) -> None:
        """Give soils that never appeared in training the mean seen embedding."""
        if self.mode == "embed":
            mean = self.table.weight[seen].mean(0)
            for s in range(N_SOILS):
                if s not in seen:
                    self.table.weight[s] = mean


class MLP(nn.Module):
    def __init__(self, physics, soil_mode, emb_dim, hidden, n_layers, dropout, batchnorm):
        super().__init__()
        self.features = InputFeatures(physics)
        self.soil = SoilEncoder(soil_mode, emb_dim)
        width = self.features.n_out + self.soil.out_dim
        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.Linear(width, hidden))
            if batchnorm:
                blocks.append(nn.BatchNorm1d(hidden))
            blocks += [nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, N_CLASSES)

    def forward(self, x, soil):
        z = torch.cat([self.features(x[:, 0], x[:, 1], soil), self.soil(soil)], 1)
        return self.head(self.body(z))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d, n_heads, attn_dropout, ffn_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(ffn_dropout),
                                 nn.Linear(2 * d, d))
        self.drop = nn.Dropout(ffn_dropout)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ffn(self.norm2(x)))


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al., NeurIPS 2021).

    Every numeric feature becomes a token x_i * w_i + b_i, soil becomes an
    embedding token, and a [CLS] token's final state is classified.
    """

    def __init__(self, physics, soil_mode, d_token, n_blocks, n_heads, attn_dropout, ffn_dropout):
        super().__init__()
        self.features = InputFeatures(physics)
        n = self.features.n_out
        bound = 1 / math.sqrt(d_token)
        self.num_weight = nn.Parameter(torch.empty(n, d_token).uniform_(-bound, bound))
        self.num_bias = nn.Parameter(torch.empty(n, d_token).uniform_(-bound, bound))
        if soil_mode not in ("embed", "factor", "none"):
            raise ValueError("FT-Transformer supports soil modes embed, factor and none")
        self.soil = SoilEncoder(soil_mode, d_token)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token).uniform_(-bound, bound))
        self.blocks = nn.Sequential(*[TransformerBlock(d_token, n_heads, attn_dropout, ffn_dropout)
                                      for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, N_CLASSES))

    def forward(self, x, soil):
        num = self.features(x[:, 0], x[:, 1], soil)
        tokens = [self.cls.expand(x.shape[0], -1, -1),
                  num[:, :, None] * self.num_weight + self.num_bias]
        if self.soil.out_dim:
            tokens.append(self.soil(soil)[:, None, :])
        z = self.blocks(torch.cat(tokens, 1))
        return self.head(z[:, 0])
