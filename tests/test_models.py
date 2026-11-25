"""Unit tests for the physics features and model plumbing (synthetic data only)."""
import numpy as np
import torch

import minesense  # noqa: F401  (sets library load order)
from minesense.models import InputFeatures, SoilEncoder, fit_offset, soil_factors
from minesense.train import TrainConfig, build_model, compute_loss

H = np.arange(12) / 11


def synthetic_series(d0=1.5, n_series=10, seed=0):
    rng = np.random.default_rng(seed)
    v, h, soil, y = [], [], [], []
    for k in range(n_series):
        a, b = rng.uniform(0.2, 0.3), rng.uniform(0.5, 5.0)
        v.append(a + b / (H + d0) ** 3)
        h.append(H)
        soil.append(np.full(12, k % 6))
        y.append(np.full(12, k // 6 + 1))
    return [np.concatenate(z) for z in (v, h, soil, y)]


def test_fit_offset_recovers_true_offset():
    v, h, soil, y = synthetic_series(d0=1.5)
    assert abs(fit_offset(v, h, soil, y) - 1.5) < 0.05


def test_soil_factors_match_names():
    moisture, texture = soil_factors(np.arange(6))
    assert moisture.tolist() == [0, 0, 0, 1, 1, 1]  # dry x3, humid x3
    assert texture.tolist() == [0, 1, 2, 0, 1, 2]   # sandy, humus, limy


def test_height_shift_follows_inverse_cube_and_round_trips():
    feats = InputFeatures(physics=True)
    feats.offset.fill_(1.0)
    feats.background.fill_(0.3)
    v, h, s = torch.tensor([0.5]), torch.tensor([0.2]), torch.tensor([0])
    v2, h2 = feats.shift_height(v, h, s, torch.tensor([0.3]))
    expected = 0.3 + 0.2 * (1.2 / 1.5) ** 3
    assert torch.allclose(v2, torch.tensor([expected]))
    v3, h3 = feats.shift_height(v2, h2, s, torch.tensor([-0.3]))
    assert torch.allclose(v3, v, atol=1e-6) and torch.allclose(h3, h)


def test_unseen_soil_gets_mean_embedding():
    enc = SoilEncoder("embed", 3)
    enc.fill_unseen([0, 1, 2, 3, 4])
    w = enc.table.weight
    assert torch.allclose(w[5], w[:5].mean(0))


def test_every_model_variant_runs_forward_and_backward():
    x = torch.rand(8, 2)
    soil = torch.arange(8) % 6
    y = torch.arange(8) % 5
    gen = torch.Generator().manual_seed(0)
    for arch in ("mlp", "ft"):
        for physics in (False, True):
            for soil_mode in ("embed", "factor", "none"):
                cfg = TrainConfig(arch=arch, physics=physics, soil_mode=soil_mode,
                                  phys_weight=0.5 if physics else 0.0)
                model = build_model(cfg)
                model.features.fit(x[:, 0].numpy(), x[:, 1].numpy(), soil.numpy(), y.numpy())
                assert model(x, soil).shape == (8, 5)
                compute_loss(model, x, soil, y, cfg, gen).backward()
