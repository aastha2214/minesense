"""Ablation study: remove one component at a time and retrain.

    python -m minesense.ablation            # run (about 10 minutes on 10 cores)
    python -m minesense.ablation --report   # tables and figure only

For every one of the 25 outer splits, the configuration that nested CV tuned
for that split (Stage 2) is reused, one component is switched off, and the
model is retrained with the same seed, then scored on the same test fold. The
"full" variant therefore reproduces the Stage 2 model exactly, and every
difference is paired by split.

Removing an input feature means replacing it with its training mean in the
train, validation and test rows, so that the architecture stays unchanged.
"""
import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .config import RESULTS_DIR, TABLES_DIR
from .dataset import load_data
from .metrics import PROB_COLS, per_split
from .registry import MODELS
from .stage2 import split_name, splits, tuned_params
from .stats import compare_paired
from .train import fit, predict_proba

ABLATION_DIR = RESULTS_DIR / "ablation"
BASES = ("pi_mlp", "pi_ft")
SAFETY_SWEEP = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)  # 2.0 is the value fixed before any experiment


def variants(arch: str) -> dict:
    """name -> (config change, input column replaced by its mean or None)."""
    v = {
        "full model": ({}, None),
        "no V input": ({}, 0),
        "no H input": ({}, 1),
        "no soil input": ({"soil_mode": "none"}, None),
        "no physics features": ({"physics": False}, None),
        "no physics loss": ({"phys_weight": 0.0}, None),
        "no physics at all": ({"physics": False, "phys_weight": 0.0}, None),
        "no noise augmentation": ({"noise": 0.0}, None),
    }
    if arch == "mlp":
        v["no batch norm"] = ({"batchnorm": False}, None)
        v["no dropout"] = ({"dropout": 0.0}, None)
        v["one-hot soil (no embedding)"] = ({"soil_mode": "onehot"}, None)
    else:
        v["no dropout"] = ({"attn_dropout": 0.0, "ffn_dropout": 0.0}, None)
    for lam in SAFETY_SWEEP:
        if lam != 2.0:
            v[f"safety weight {lam:g}"] = ({"miss_weight": lam}, None)
    return v


def run_one(key: str, variant: str, seed: int, fold: int):
    x, soil, y = load_data()
    sp = next(s for s in splits() if (s.seed, s.fold) == (seed, fold))
    cfg = tuned_params(key)[split_name(sp)]
    change, drop = variants(cfg.arch)[variant]
    cfg = replace(cfg, **change)
    if drop is not None:
        x = x.copy()
        x[:, drop] = x[sp.train, drop].mean()
    model, history = fit(cfg, x, soil, y, sp.train, sp.val, seed=1000 * seed + fold)
    model.soil.fill_unseen(sorted(set(soil[sp.train].tolist())))
    df = pd.DataFrame(predict_proba(model, x[sp.test], soil[sp.test]), columns=PROB_COLS)
    df.insert(0, "y", y[sp.test])
    df.insert(0, "row", sp.test)
    for name, value in (("fold", fold), ("seed", seed), ("variant", variant), ("model", key)):
        df.insert(0, name, value)
    return df, history["best_epoch"]


def worker_init() -> None:
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", category=UserWarning)


def run(workers: int) -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(key, name, sp.seed, sp.fold) for key in BASES
            for name in variants(MODELS[key].base.arch) for sp in splits()]
    start = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=worker_init) as pool:
        results = list(pool.map(run_one, *zip(*jobs), chunksize=1))
    preds = pd.concat([r[0] for r in results], ignore_index=True)
    preds.to_csv(ABLATION_DIR / "predictions.csv", index=False, float_format="%.6f")
    print(f"{len(jobs)} fits in {time.perf_counter() - start:.0f}s")


def report() -> None:
    preds = pd.read_csv(ABLATION_DIR / "predictions.csv")
    stage2 = {k: pd.read_csv(RESULTS_DIR / "cv" / k / "predictions.csv") for k in BASES}
    rows, sweep = [], []
    for key in BASES:
        sub = preds[preds["model"] == key]
        full = sub[sub["variant"] == "full model"].sort_values(["seed", "fold", "row"])
        ref = stage2[key].sort_values(["seed", "fold", "row"])
        repro = np.abs(full[PROB_COLS].to_numpy() - ref[PROB_COLS].to_numpy()).max()
        print(f"{key}: max |p| difference between re-fitted full model and Stage 2: {repro:.2e}")
        base = per_split(full)
        for name in variants(MODELS[key].base.arch):
            s = per_split(sub[sub["variant"] == name])
            row = {"model": MODELS[key].label, "variant": name}
            for m in ("accuracy", "macro_f1", "mine_missed", "false_alarm"):
                row[m] = s[m].mean()
            if name != "full model":
                for m in ("accuracy", "mine_missed"):
                    test = compare_paired(s[m].to_numpy(), base[m].to_numpy())
                    row[f"delta_{m}"] = test["mean_diff"]
                    row[f"delta_{m}_std"] = test["std_diff"]
                    row[f"p_{m}_wilcoxon"] = test["p_wilcoxon"]
                    row[f"p_{m}_corrected_t"] = test["p_corrected_t"]
            rows.append(row)
            lam = 2.0 if name == "full model" else (
                float(name.split()[-1]) if name.startswith("safety weight") else None)
            if lam is not None:
                sweep.append({"key": key, "lambda": lam, **row})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "ablation.csv", index=False, float_format="%.4f")
    sweep = pd.DataFrame(sweep).sort_values(["key", "lambda"])
    sweep.drop(columns="key").to_csv(TABLES_DIR / "safety_weight_sweep.csv", index=False,
                                     float_format="%.4f")
    fig_ablation(table)
    fig_sweep(sweep)
    pd.set_option("display.width", 250)
    cols = ["model", "variant", "accuracy", "delta_accuracy", "p_accuracy_corrected_t",
            "mine_missed", "delta_mine_missed", "false_alarm"]
    print(table[cols].round(4).to_string(index=False))


def fig_ablation(table: pd.DataFrame) -> None:
    t = table[(table["variant"] != "full model") & ~table["variant"].str.startswith("safety")]
    labels = list(dict.fromkeys(t["variant"]))
    fig, axes = plt.subplots(1, 2, figsize=(P.FULL_WIDTH, 3.2), sharey=True,
                             constrained_layout=True)
    y = np.arange(len(labels))
    for ax, metric, title in ((axes[0], "accuracy", "(a) Change in accuracy"),
                              (axes[1], "mine_missed", "(b) Change in mine-missed rate")):
        for j, (model, color) in enumerate(zip(t["model"].unique(), (P.BLUE, P.ORANGE))):
            m = t[t["model"] == model].set_index("variant").reindex(labels)
            off = (j - 0.5) * 0.36
            ax.barh(y + off, m[f"delta_{metric}"], height=0.34, color=color, label=model)
            ax.errorbar(m[f"delta_{metric}"], y + off,
                        xerr=m[f"delta_{metric}_std"] / np.sqrt(25), fmt="none",
                        ecolor=P.INK, elinewidth=0.8, capsize=1.5)
        ax.axvline(0, color=P.INK, linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Ablated minus full model (mean over 25 splits ± s.e.)")
    axes[1].set_xlabel("Positive = more real mines missed")
    axes[0].legend(loc="lower left", frameon=False, fontsize=7)
    P.save(fig, "fig16_ablation")


def fig_sweep(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for key, color, offset in zip(BASES, (P.BLUE, P.ORANGE), ((-4, -9), (4, 3))):
        s = sweep[sweep["key"] == key]
        ax.plot(s["false_alarm"], s["mine_missed"], "-o", color=color, markersize=4,
                linewidth=1.5, label=MODELS[key].label.split(" (")[0])
        for _, r in s.iterrows():
            ax.annotate(f"λ={r['lambda']:g}", (r["false_alarm"], r["mine_missed"]),
                        textcoords="offset points", xytext=offset, fontsize=6, color=color,
                        ha="right" if offset[0] < 0 else "left")
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Mine-missed rate")
    ax.set_title("Safety-loss weight trade-off", loc="left")
    ax.legend(frameon=False, fontsize=7)
    P.save(fig, "fig17_safety_tradeoff")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    P.apply_style()
    if not args.report:
        run(args.workers)
    report()


if __name__ == "__main__":
    main()
