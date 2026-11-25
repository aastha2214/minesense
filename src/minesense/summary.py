"""Summarise nested-CV and leave-one-soil-out results into tables and figures.

    python -m minesense.summary
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import plotting as P
from .config import SOIL_NAMES, TABLES_DIR
from .cv import CV_DIR, MODEL_DIR
from .dataset import load_data
from .loso import LOSO_DIR, LOSO_SPECS
from .metrics import PROB_COLS, basic_metrics, per_split
from .registry import BASELINE_SPECS, DL_SPECS, MODELS

METRICS = ["accuracy", "macro_f1", "mine_missed", "false_alarm"]
PHYSICS_PAIRS = [("mlp_phys", "mlp"), ("pi_mlp", "mlp"), ("pi_mlp", "mlp_phys"),
                 ("ft_phys", "ft"), ("pi_ft", "ft"), ("pi_ft", "ft_phys")]


def fmt(mu, sd):
    return f"{mu:.3f} ± {sd:.3f}"


def load_cv() -> dict[str, pd.DataFrame]:
    return {k: pd.read_csv(CV_DIR / k / "predictions.csv")
            for k in MODELS if (CV_DIR / k / "predictions.csv").exists()}


def cv_table(preds: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows, splits = [], {}
    for key, df in preds.items():
        s = per_split(df)
        splits[key] = s
        params = json.loads((CV_DIR / key / "params.json").read_text())
        row = {"key": key, "model": MODELS[key].label,
               "type": "deep learning" if MODELS[key].kind == "dl" else "baseline"}
        for m in METRICS:
            row[m] = fmt(s[m].mean(), s[m].std(ddof=1))
            row[f"{m}_mean"] = s[m].mean()
        row["inner_macro_f1"] = np.mean([p["inner_macro_f1"] for p in params.values()])
        row["tune_minutes_per_split"] = np.median([p["tune_seconds"] for p in params.values()]) / 60
        rows.append(row)
    return pd.DataFrame(rows), splits


def physics_pairs(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Paired, per-split differences. Descriptive here; formal tests come in Stage 3."""
    rows = []
    for a, b in PHYSICS_PAIRS:
        if a not in splits or b not in splits:
            continue
        d = splits[a]["accuracy"].to_numpy() - splits[b]["accuracy"].to_numpy()
        f = splits[a]["macro_f1"].to_numpy() - splits[b]["macro_f1"].to_numpy()
        rows.append({"comparison": f"{MODELS[a].label} vs {MODELS[b].label}",
                     "mean_accuracy_gain": d.mean(), "std": d.std(ddof=1),
                     "mean_macro_f1_gain": f.mean(),
                     "splits_better": int((d > 1e-9).sum()), "splits_equal": int((abs(d) <= 1e-9).sum()),
                     "splits_worse": int((d < -1e-9).sum())})
    return pd.DataFrame(rows)


def fitted_offsets() -> dict:
    out = {}
    for key in ("mlp_phys", "pi_mlp", "ft_phys", "pi_ft"):
        files = sorted((MODEL_DIR / key).glob("*.pt"))
        if files:
            d0 = [float(torch.load(f, weights_only=True)["state_dict"]["features.offset"])
                  for f in files]
            out[key] = {"median": float(np.median(d0)), "min": float(min(d0)),
                        "max": float(max(d0))}
    return out


def tuned_choices(key: str) -> dict:
    params = json.loads((CV_DIR / key / "params.json").read_text())
    table = pd.DataFrame([p["params"] for p in params.values()])
    out = {}
    for col in table.columns:
        if table[col].nunique() <= 1:
            continue
        if table[col].dtype.kind == "f":
            out[col] = f"median {table[col].median():.3g} (range {table[col].min():.3g}-{table[col].max():.3g})"
        else:
            out[col] = {str(k): int(v) for k, v in table[col].value_counts().items()}
    return out


def loso_tables(cv_preds) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    frames = [pd.read_csv(LOSO_DIR / k / "predictions.csv") for k in LOSO_SPECS
              if (LOSO_DIR / k / "predictions.csv").exists()]
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    df["pred"] = df[PROB_COLS].to_numpy().argmax(1)
    per_rep = (df.groupby(["model", "held_out", "rep"])
                 .apply(lambda g: pd.Series(basic_metrics(g["y"].to_numpy(), g["pred"].to_numpy())),
                        include_groups=False).reset_index())
    by_soil = per_rep.pivot_table(index="model", columns="held_out", values="accuracy")
    by_soil = by_soil[SOIL_NAMES]
    rep_means = per_rep.groupby(["model", "rep"])[METRICS].mean()
    overall = rep_means.groupby("model").agg(["mean", "std"])
    summary = pd.DataFrame({
        "model": [LOSO_SPECS[k].label for k in overall.index],
        **{m: [fmt(overall.loc[k, (m, "mean")], overall.loc[k, (m, "std")]) for k in overall.index]
           for m in METRICS},
        "accuracy_mean": overall[("accuracy", "mean")].to_numpy(),
    }, index=overall.index)

    # Reference when the soil *was* seen in training: main nested-CV accuracy of
    # the same model family (main-CV neural models use the 6-way soil embedding),
    # averaged over soils so it is weighted like the LOSO number.
    _, soil, _ = load_data()
    family = {"mlp_none": "mlp", "mlp_embed": "mlp", "mlp_factor": "mlp",
              "pi_mlp_factor": "pi_mlp", "ft_factor": "ft", "pi_ft_factor": "pi_ft",
              "knn": "knn", "svm": "svm", "rf": "rf", "xgb": "xgb"}
    seen = {}
    for key in set(family.values()) & set(cv_preds):
        c = cv_preds[key].assign(soil=lambda d: soil[d["row"]])
        c["pred"] = c[PROB_COLS].to_numpy().argmax(1)
        seen[key] = float(np.mean([(g["pred"] == g["y"]).mean() for _, g in c.groupby("soil")]))
    summary["main_cv_family"] = [MODELS[family[k]].label for k in summary.index]
    summary["main_cv_accuracy_soil_seen"] = [seen.get(family[k], np.nan) for k in summary.index]
    by_soil.index = [LOSO_SPECS[k].label for k in by_soil.index]
    return summary.sort_values("accuracy_mean", ascending=False), by_soil


def fig_cv(table: pd.DataFrame, splits: dict) -> None:
    order = table.sort_values("accuracy_mean")["key"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(P.FULL_WIDTH, 3.0), sharey=True,
                             constrained_layout=True)
    y = np.arange(len(order))
    for ax, metric, title in ((axes[0], "accuracy", "(a) Accuracy"),
                              (axes[1], "mine_missed", "(b) Mine-missed rate (lower is safer)")):
        for i, key in enumerate(order):
            vals = splits[key][metric].to_numpy()
            color = P.BLUE if MODELS[key].kind == "dl" else P.MUTED
            ax.scatter(vals, np.full(len(vals), i) + np.linspace(-0.25, 0.25, len(vals)),
                       s=5, color=color, alpha=0.6, linewidths=0)
            ax.errorbar(vals.mean(), i, xerr=vals.std(ddof=1), fmt="o", color=P.INK,
                        markersize=4, capsize=2, linewidth=1)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [MODELS[k].label for k in order])
    axes[0].set_xlabel("Per-split value (25 outer test folds); dot = mean ± std")
    axes[1].set_xlabel("Share of real mines predicted as 'no mine'")
    P.save(fig, "fig08_nested_cv_results")


def fig_loso(by_soil: pd.DataFrame, summary: pd.DataFrame) -> None:
    by_soil = by_soil.loc[[LOSO_SPECS[k].label for k in summary.index]]
    fig, ax = plt.subplots(figsize=(P.FULL_WIDTH, 0.32 * len(by_soil) + 1.0),
                           constrained_layout=True)
    image = ax.imshow(by_soil.to_numpy(), cmap=P.SEQUENTIAL, vmin=0.2, vmax=0.9, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(by_soil.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.6 else P.INK)
    ax.set_xticks(range(len(SOIL_NAMES)), [f"{s}\n(held out)" for s in SOIL_NAMES])
    ax.set_yticks(range(len(by_soil)), by_soil.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Accuracy on a soil type never seen in training (mean of 5 seeds)", loc="left")
    fig.colorbar(image, ax=ax, shrink=0.8).outline.set_visible(False)
    P.save(fig, "fig09_leave_one_soil_out")


def main() -> None:
    P.apply_style()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_cv()
    table, splits = cv_table(preds)
    table = table.sort_values(["type", "accuracy_mean"], ascending=[False, False])
    table.drop(columns=[c for c in table if c.endswith("_mean")]).to_csv(
        TABLES_DIR / "cv_summary.csv", index=False, float_format="%.4f")
    pairs = physics_pairs(splits)
    pairs.to_csv(TABLES_DIR / "physics_comparison.csv", index=False, float_format="%.4f")
    extra = {"fitted_offset_d0": fitted_offsets(),
             "tuned_hyperparameters": {k: tuned_choices(k) for k in preds}}
    (TABLES_DIR / "cv_details.json").write_text(json.dumps(extra, indent=1) + "\n")
    fig_cv(table, splits)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(table[["model", *METRICS, "inner_macro_f1", "tune_minutes_per_split"]].to_string(index=False))
    print()
    print(pairs.to_string(index=False))
    print(json.dumps(extra["fitted_offset_d0"], indent=1))

    loso, by_soil = loso_tables(preds)
    if loso is not None:
        loso.drop(columns="accuracy_mean").to_csv(TABLES_DIR / "loso_summary.csv", float_format="%.4f")
        by_soil.to_csv(TABLES_DIR / "loso_accuracy_by_soil.csv", float_format="%.4f")
        fig_loso(by_soil, loso)
        print()
        print(loso.drop(columns="accuracy_mean").to_string(index=False))
        print(by_soil.round(3).to_string())


if __name__ == "__main__":
    main()
