"""Time training and inference on CPU and Apple MPS.

    python -m minesense.bench_device

Writes results/tables/device_benchmark.json. The pipeline uses whichever
device is faster for each job.
"""
import json
import platform
import time
from dataclasses import replace

import numpy as np
import torch

from .config import TABLES_DIR
from .dataset import load_data
from .seed import set_seed
from .splits import outer_splits
from .train import TrainConfig, fit, predict_proba

CONFIGS = {
    "MLP": TrainConfig(arch="mlp"),
    "PI-MLP": TrainConfig(arch="mlp", physics=True, phys_weight=0.5),
    "FT-Transformer": TrainConfig(arch="ft"),
}
EPOCHS = 100
REPEATS = 3
BATCH_ROWS = 100_000


def devices() -> list[str]:
    return ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def main() -> None:
    set_seed()
    torch.set_num_threads(1)
    x, soil, y = load_data()
    sp = next(outer_splits(y))
    rng = np.random.default_rng(0)
    big_x = rng.uniform(0, 1, (BATCH_ROWS, 2)).astype(np.float32)
    big_s = rng.integers(0, 6, BATCH_ROWS)
    out = {"machine": platform.machine(), "processor": platform.processor(),
           "torch": torch.__version__, "cpu_threads": 1,
           "train_epochs": EPOCHS, "train_rows": len(sp.train), "repeats": REPEATS, "results": {}}
    for name, base in CONFIGS.items():
        cfg = replace(base, max_epochs=EPOCHS, patience=EPOCHS)
        for device in devices():
            fit(cfg, x, soil, y, sp.train, sp.val, seed=0, device=device)  # warm-up
            times = []
            for r in range(REPEATS):
                t = time.perf_counter()
                model, _ = fit(cfg, x, soil, y, sp.train, sp.val, seed=r, device=device)
                sync(device)
                times.append(time.perf_counter() - t)
            predict_proba(model, big_x, big_s, device)
            sync(device)
            t = time.perf_counter()
            predict_proba(model, big_x, big_s, device)
            sync(device)
            infer = time.perf_counter() - t
            out["results"][f"{name} / {device}"] = {
                "train_seconds_per_100_epochs": float(np.median(times)),
                f"inference_seconds_for_{BATCH_ROWS}_rows": infer,
            }
            print(name, device, out["results"][f"{name} / {device}"])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "device_benchmark.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
