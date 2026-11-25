"""SHAP explanations, global and per prediction.

    python -m minesense.explain

With only three inputs (V, H, soil) Shapley values can be computed exactly
(2^3 coalitions), so no sampling approximation is needed. The explained output
is each class probability. We use the seed-0 models of Stage 2: their five test
folds partition the data, so every reading is explained once, by a model that
never saw it. Missing features are filled from 100 training rows of the same
split (interventional SHAP with an independent masker).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import plotting as P
from .config import CLASS_NAMES, NO_MINE, SOIL_NAMES, TABLES_DIR
from .dataset import load_data
from .registry import MODELS
from .stage2 import BEST_DL, fitted_model, proba, splits

FEATURES = ["V (voltage)", "H (height)", "S (soil)"]
EXPLAINED = (BEST_DL, "svm")


def explain(key: str):
    x, soil, y = load_data()
    values, base, rows, probs = [], [], [], []
    for sp in [s for s in splits() if s.seed == 0]:
        model = fitted_model(key, sp)

        def f(z, model=model):
            return proba(key, model, z[:, :2].astype(np.float32), z[:, 2].round().astype(int))

        data = np.column_stack([x, soil]).astype(np.float64)
        rng = np.random.default_rng(sp.fold)
        background = data[rng.choice(sp.train, 100, replace=False)]
        explainer = shap.explainers.Exact(f, shap.maskers.Independent(background, max_samples=100))
        e = explainer(data[sp.test])
        values.append(e.values)        # (n, 3 features, 5 classes)
        base.append(e.base_values)     # (n, 5)
        rows.append(sp.test)
        probs.append(f(data[sp.test]))
    order = np.argsort(np.concatenate(rows))
    return (np.concatenate(values)[order], np.concatenate(base)[order],
            np.concatenate(probs)[order], x, soil, y)


def main() -> None:
    import torch

    # Multi-threaded torch inside SHAP's batched calls segfaults on macOS
    # (OpenMP runtimes clash); these models are tiny, so one thread is enough.
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    P.apply_style()
    results = {key: explain(key) for key in EXPLAINED}
    rows = []
    for key, (sv, base, p, *_ ) in results.items():
        assert np.allclose(base + sv.sum(1), p, atol=1e-5)  # Shapley efficiency
        for c in range(5):
            for j, name in enumerate(FEATURES):
                rows.append({"model": MODELS[key].label, "class": CLASS_NAMES[c], "feature": name,
                             "mean_abs_shap": np.abs(sv[:, j, c]).mean()})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "shap_global.csv", index=False, float_format="%.4f")
    print(table.pivot_table(index=["model", "feature"], columns="class",
                            values="mean_abs_shap")[CLASS_NAMES].round(3).to_string())
    fig_global(results, table)
    fig_local(results[BEST_DL])


def fig_global(results, table) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(P.FULL_WIDTH, 2.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, key in zip(axes[:2], EXPLAINED):
        t = table[table["model"] == MODELS[key].label]
        left = np.zeros(3)
        for c in range(5):
            v = t[t["class"] == CLASS_NAMES[c]].set_index("feature").loc[FEATURES, "mean_abs_shap"]
            ax.barh(range(3), v, left=left, color=P.CLASS_COLORS[c], height=0.6,
                    edgecolor="white", linewidth=1, label=CLASS_NAMES[c])
            left += v.to_numpy()
        ax.set_yticks(range(3), FEATURES)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(MODELS[key].label.split(" (")[0], loc="left", fontsize=8)
        ax.set_xlabel("Mean |SHAP| summed over classes")
    axes[1].set_yticklabels([])
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    sv, _, _, x, soil, _ = results[BEST_DL]
    sc = axes[2].scatter(x[:, 0], sv[:, 0, NO_MINE], c=x[:, 1], cmap=P.SEQUENTIAL, s=7,
                         linewidths=0)
    axes[2].axhline(0, color=P.AXIS, linewidth=0.8)
    axes[2].set_xlabel("V (normalised)")
    axes[2].set_ylabel("SHAP of V on p(no mine)")
    axes[2].set_title("Voltage effect, coloured by height", loc="left", fontsize=8)
    cb = fig.colorbar(sc, ax=axes[2], shrink=0.8)
    cb.set_label("H (normalised)", fontsize=7)
    cb.outline.set_visible(False)
    P.save(fig, "fig15_shap_global")


def fig_local(result) -> None:
    sv, base, p, x, soil, y = result
    pred = p.argmax(1)
    cases = []
    for title, mask in (("Correct: anti-tank mine", (y == 1) & (pred == 1)),
                        ("Missed mine (predicted empty)", (y != NO_MINE) & (pred == NO_MINE)),
                        ("False alarm on empty ground", (y == NO_MINE) & (pred != NO_MINE))):
        idx = np.flatnonzero(mask)
        if len(idx):
            cases.append((title, idx[len(idx) // 2]))
    fig, axes = plt.subplots(1, len(cases), figsize=(P.FULL_WIDTH, 2.1), constrained_layout=True)
    for ax, (title, i) in zip(np.atleast_1d(axes), cases):
        c = pred[i]
        contrib = sv[i, :, c]
        labels = [f"V = {x[i, 0]:.2f}", f"H = {x[i, 1]:.2f}", f"S = {SOIL_NAMES[soil[i]]}"]
        start = base[i, c]
        for j in range(3):
            ax.barh(j, contrib[j], left=start, color=P.BLUE if contrib[j] > 0 else P.ORANGE,
                    height=0.55)
            ax.text(start + contrib[j] + (0.01 if contrib[j] >= 0 else -0.01), j,
                    f"{contrib[j]:+.2f}", va="center", fontsize=6.5,
                    ha="left" if contrib[j] >= 0 else "right", zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
            start += contrib[j]
        ax.axvline(base[i, c], color=P.AXIS, linewidth=0.8, linestyle="--")
        ax.axvline(p[i, c], color=P.INK, linewidth=0.8)
        ax.set_yticks(range(3), labels, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{title}\ntrue {CLASS_NAMES[y[i]]}, predicted {CLASS_NAMES[c]}",
                     loc="left", fontsize=7)
        ax.set_xlabel(f"p({CLASS_NAMES[c]}): base {base[i, c]:.2f} → {p[i, c]:.2f}", fontsize=7)
    P.save(fig, "fig18_shap_local")


if __name__ == "__main__":
    main()
