"""Training loop shared by all neural models."""
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import NO_MINE
from .models import MLP, SATURATION, FTTransformer


@dataclass
class TrainConfig:
    arch: str = "mlp"            # "mlp" or "ft"
    physics: bool = False        # physics-informed input features
    soil_mode: str = "embed"     # embed | factor | onehot | none
    # MLP
    emb_dim: int = 4
    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    batchnorm: bool = True
    # FT-Transformer
    d_token: int = 32
    n_blocks: int = 2
    n_heads: int = 4
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    # Optimisation
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 400
    patience: int = 40           # early stopping on validation cross-entropy
    # Regularisers
    noise: float = 0.1           # Gaussian augmentation, std as a fraction of each input's std
    miss_weight: float = 2.0     # safety penalty on p(no mine) for true mines
    phys_weight: float = 0.0     # physics-consistency loss weight
    phys_delta: float = 0.25     # max simulated height change (normalised units)


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.arch == "mlp":
        return MLP(cfg.physics, cfg.soil_mode, cfg.emb_dim, cfg.hidden, cfg.n_layers,
                   cfg.dropout, cfg.batchnorm)
    if cfg.arch == "ft":
        return FTTransformer(cfg.physics, cfg.soil_mode, cfg.d_token, cfg.n_blocks, cfg.n_heads,
                             cfg.attn_dropout, cfg.ffn_dropout)
    raise ValueError(f"unknown architecture {cfg.arch!r}")


def compute_loss(model, xb, sb, yb, cfg: TrainConfig, gen) -> torch.Tensor:
    """Cross-entropy + safety penalty + physics-consistency term."""
    n = len(yb)
    if cfg.phys_weight > 0:
        delta = (torch.rand(n, generator=gen) * 2 - 1).to(xb.device) * cfg.phys_delta
        v2, h2 = model.features.shift_height(xb[:, 0], xb[:, 1], sb, delta)
        logits_all = model(torch.cat([xb, torch.stack([v2, h2], 1)]), torch.cat([sb, sb]))
        logits, logits_shifted = logits_all[:n], logits_all[n:]
    else:
        logits = model(xb, sb)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp, yb)

    if cfg.miss_weight > 0:
        # -log(1 - p(no mine)) on rows that hold a mine: grows without bound as
        # the model becomes confident that a real mine is empty ground.
        p_none = logp[:, NO_MINE].exp().clamp(max=1 - 1e-6)
        is_mine = (yb != NO_MINE).to(logp.dtype)
        loss = loss + cfg.miss_weight * (-torch.log1p(-p_none) * is_mine).sum() / n

    if cfg.phys_weight > 0:
        # The class must not change when the same object is seen from another height.
        target = logp.detach()
        kl = (target.exp() * (target - F.log_softmax(logits_shifted, 1))).sum(1)
        valid = (xb[:, 0] < SATURATION).to(kl.dtype)  # a clipped reading cannot be rescaled
        loss = loss + cfg.phys_weight * (kl * valid).sum() / valid.sum().clamp_min(1.0)
    return loss


def _tensors(x, soil, y, idx, device):
    return (torch.as_tensor(x[idx], device=device), torch.as_tensor(soil[idx], device=device),
            torch.as_tensor(y[idx], device=device))


@torch.no_grad()
def _eval_ce(model, xt, st, yt) -> float:
    model.eval()
    return float(F.cross_entropy(model(xt, st), yt))


def fit(cfg: TrainConfig, x: np.ndarray, soil: np.ndarray, y: np.ndarray,
        train_idx: np.ndarray, val_idx: np.ndarray, seed: int, device: str = "cpu"):
    """Train with early stopping; return (best model, history)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_model(cfg)
    model.features.fit(x[train_idx, 0], x[train_idx, 1], soil[train_idx], y[train_idx])
    model.to(device)
    xt, st, yt = _tensors(x, soil, y, train_idx, device)
    xv, sv, yv = _tensors(x, soil, y, val_idx, device)
    noise_scale = torch.as_tensor(cfg.noise * x[train_idx].std(0), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    n = len(train_idx)
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if len(idx) < 2:  # batch norm needs at least two rows
                continue
            xb = xt[idx]
            if cfg.noise > 0:
                eps = torch.randn(xb.shape, generator=gen).to(device)
                xb = (xb + eps * noise_scale).clamp(0.0, 1.0)
            loss = compute_loss(model, xb, st[idx], yt[idx], cfg, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_ce, val_ce = _eval_ce(model, xt, st, yt), _eval_ce(model, xv, sv, yv)
        history["train_loss"].append(train_ce)
        history["val_loss"].append(val_ce)
        if val_ce < best - 1e-5:
            best, best_epoch, wait = val_ce, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"] = best_epoch
    return model, history


@torch.no_grad()
def predict_proba(model, x: np.ndarray, soil: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    logits = model(torch.as_tensor(x, device=device), torch.as_tensor(soil, device=device))
    return torch.softmax(logits, 1).cpu().numpy()


def save_model(model, cfg: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, path)


def load_model(path: Path) -> tuple[torch.nn.Module, TrainConfig]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg = TrainConfig(**blob["config"])
    model = build_model(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg
