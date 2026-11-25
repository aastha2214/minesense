"""Robustness to sensor noise at test time.

    python -m minesense.robustness

Every Stage 2 model is scored on its own test fold after Gaussian noise is added
to the voltage and height readings. The noise standard deviation is a multiple
sigma of each input's training standard deviation; readings are clipped to the
sensor range [0, 1], as in training-time augmentation. Soil type is unchanged.
Five noise draws per split and level.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .config import NO_MINE, TABLES_DIR
from .dataset import load_data
from .registry import MODELS
from .stage2 import BEST_DL, fitted_model, proba, splits
from .stats import compare_paired

SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
DRAWS = 5


def run_model(key: str) -> pd.DataFrame:
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    x, soil, y = load_data()
    rows = []
    for sp in splits():
        model = fitted_model(key, sp)
        scale = x[sp.train].std(0)
        rng = np.random.default_rng(1000 * sp.seed + sp.fold)
        yt, mine = y[sp.test], y[sp.test] != NO_MINE
        for sigma in SIGMAS:
            for draw in range(DRAWS if sigma > 0 else 1):
                xn = np.clip(x[sp.test] + rng.normal(size=x[sp.test].shape) * sigma * scale, 0, 1)
                pred = proba(key, model, xn.astype(np.float32), soil[sp.test]).argmax(1)
                rows.append({"model": key, "seed": sp.seed, "fold": sp.fold, "sigma": sigma,
                             "draw": draw, "accuracy": (pred == yt).mean(),
                             "mine_missed": (pred[mine] == NO_MINE).mean()})
    return pd.DataFrame(rows)


def main() -> None:
    P.apply_style()
    with ProcessPoolExecutor(10) as pool:
        df = pd.concat(pool.map(run_model, list(MODELS)), ignore_index=True)
    # Average draws within a split, then mean and std over the 25 splits.
    per_split = df.groupby(["model", "sigma", "seed", "fold"])[["accuracy", "mine_missed"]].mean()
    table = per_split.groupby(["model", "sigma"]).agg(["mean", "std"]).reset_index()
    table.columns = ["_".join(c).strip("_") for c in table.columns]
    table["model"] = table["model"].map(lambda k: MODELS[k].label)
    table.to_csv(TABLES_DIR / "noise_robustness.csv", index=False, float_format="%.4f")
    # Paired by split: best neural model vs SVM at every noise level.
    tests = []
    for sigma in SIGMAS:
        for metric in ("accuracy", "mine_missed"):
            a = per_split.loc[(BEST_DL, sigma), metric].to_numpy()
            b = per_split.loc[("svm", sigma), metric].to_numpy()
            tests.append({"sigma": sigma, "metric": metric, **compare_paired(a, b)})
    tests = pd.DataFrame(tests)
    tests.to_csv(TABLES_DIR / "noise_robustness_tests.csv", index=False, float_format="%.4g")
    print(f"{MODELS[BEST_DL].label} minus SVM, paired over 25 splits")
    print(tests[["sigma", "metric", "mean_diff", "wins", "losses", "p_wilcoxon", "p_corrected_t"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    wide = table.pivot(index="model", columns="sigma", values="accuracy_mean")
    print(wide.round(3).to_string())
    print(table.pivot(index="model", columns="sigma", values="mine_missed_mean").round(3).to_string())
    fig(table)


def fig(table: pd.DataFrame) -> None:
    order = ["pi_ft", "ft", "pi_mlp", "mlp", "svm", "rf", "xgb", "knn"]
    styles = {"pi_ft": (P.BLUE, "-", "o"), "ft": (P.BLUE, "--", "o"),
              "pi_mlp": (P.ORANGE, "-", "s"), "mlp": (P.ORANGE, "--", "s"),
              "svm": (P.CLASS_COLORS[2], "-", "^"), "rf": (P.CLASS_COLORS[3], "-", "D"),
              "xgb": (P.CLASS_COLORS[4], "-", "v"), "knn": (P.INK_2, "-", "x")}
    fig, axes = plt.subplots(1, 2, figsize=(P.FULL_WIDTH, 2.8), constrained_layout=True)
    for key in order:
        t = table[table["model"] == MODELS[key].label]
        color, ls, marker = styles[key]
        for ax, metric in zip(axes, ("accuracy", "mine_missed")):
            ax.plot(t["sigma"], t[f"{metric}_mean"], color=color, linestyle=ls, marker=marker,
                    markersize=3.5, linewidth=1.3,
                    label=MODELS[key].label.replace(" (physics features + loss)", ""))
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Mine-missed rate")
    axes[0].set_title("(a) Accuracy under input noise", loc="left")
    axes[1].set_title("(b) Real mines predicted as 'no mine'", loc="left")
    for ax in axes:
        ax.set_xlabel("Noise std (multiples of each input's training std)")
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    P.save(fig, "fig14_noise_robustness")


if __name__ == "__main__":
    main()
