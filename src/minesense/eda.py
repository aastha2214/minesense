"""Exploratory data analysis: figures and summary tables.

    python -m minesense.eda
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LinearRegression

from . import plotting as P
from .config import CLASS_NAMES, N_CLASSES, N_SOILS, SOIL_NAMES, TABLES_DIR
from .dataset import load_raw, validate

V_LABEL = "Sensor voltage V (normalised)"
SATURATION = 0.999  # readings pinned at the top of the voltage range
H_LABEL = "Sensor height H (normalised)"


def class_handles():
    return [Line2D([], [], color=c, marker=m, label=n)
            for c, m, n in zip(P.CLASS_COLORS, P.CLASS_MARKERS, CLASS_NAMES)]


def annotated_heatmap(ax, table, cmap, fmt, vmin, vmax, text_flip):
    image = ax.imshow(table, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    for (i, j), value in np.ndenumerate(table):
        ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7,
                color="white" if text_flip(value) else P.INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return image


def fig_class_balance(df):
    counts = df["M"].value_counts().sort_index().to_numpy()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y = np.arange(N_CLASSES)
    ax.barh(y, counts, height=0.62, color=P.BLUE)
    for i, n in enumerate(counts):
        ax.text(n + 1, i, f"{n}  ({n / counts.sum():.1%})", va="center", fontsize=7)
    ax.set_yticks(y, CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("Number of measurements")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Class balance (n = {counts.sum()})", loc="left")
    P.save(fig, "fig01_class_balance")


def fig_design_coverage(df):
    soil = pd.crosstab(df["S"], df["M"]).to_numpy()
    height = pd.crosstab(df["H"], df["M"]).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(P.FULL_WIDTH, 3.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    short = ["No\nmine", "Anti-\ntank", "Anti-\npers.", "Booby\ntrap AP", "M14\nAP"]
    annotated_heatmap(axes[0], soil, P.SEQUENTIAL, "{:.0f}", 0, soil.max(), lambda v: v > 7)
    axes[0].set_yticks(range(N_SOILS), SOIL_NAMES)
    axes[0].set_title("(a) Rows per soil type and class", loc="left")
    annotated_heatmap(axes[1], height, P.SEQUENTIAL, "{:.0f}", 0, height.max(), lambda v: v > 3.5)
    heights = np.sort(df["H"].unique())
    axes[1].set_yticks(range(len(heights)), [f"{h:.2f}" for h in heights])
    axes[1].set_ylabel(H_LABEL)
    axes[1].set_title("(b) Rows per height and class", loc="left")
    for ax in axes:
        ax.set_xticks(range(N_CLASSES), short)
    P.save(fig, "fig02_design_coverage")


def fig_voltage_distribution(df, rng):
    fig, axes = plt.subplots(1, 2, figsize=(P.FULL_WIDTH, 2.6), constrained_layout=True)
    axes[0].hist(df["V"], bins=np.linspace(0.15, 1.0, 35), color=P.BLUE,
                 edgecolor="white", linewidth=0.6)
    axes[0].set(xlabel=V_LABEL, ylabel="Number of measurements")
    axes[0].set_title("(a) All measurements", loc="left")
    ax = axes[1]
    groups = [df.loc[df["M"] == m + 1, "V"].to_numpy() for m in range(N_CLASSES)]
    ax.boxplot(groups, orientation="horizontal", widths=0.6, showfliers=False,
               medianprops={"color": P.INK, "linewidth": 1.4},
               boxprops={"color": P.INK_2}, whiskerprops={"color": P.INK_2},
               capprops={"color": P.INK_2})
    for m, values in enumerate(groups):
        ax.scatter(values, m + 1 + rng.uniform(-0.22, 0.22, len(values)), s=9,
                   color=P.CLASS_COLORS[m], marker=P.CLASS_MARKERS[m], alpha=0.75, linewidths=0)
    ax.set_yticks(range(1, N_CLASSES + 1), CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel(V_LABEL)
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) By class", loc="left")
    P.save(fig, "fig03_voltage_distribution")


def fig_voltage_height_by_soil(df):
    fig, axes = plt.subplots(2, 3, figsize=(P.FULL_WIDTH, 4.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for s, ax in enumerate(axes.flat):
        for m in range(N_CLASSES):
            part = df[(df["S"] == s + 1) & (df["M"] == m + 1)].sort_values("H")
            ax.plot(part["H"], part["V"], color=P.CLASS_COLORS[m],
                    marker=P.CLASS_MARKERS[m], markeredgecolor="white", markeredgewidth=0.4)
        ax.set_title(SOIL_NAMES[s], loc="left")
    fig.supxlabel(H_LABEL, fontsize=8, color=P.INK_2)
    fig.supylabel(V_LABEL, fontsize=8, color=P.INK_2)
    fig.legend(handles=class_handles(), loc="outside upper center", ncol=N_CLASSES,
               handlelength=1.8, columnspacing=1.2)
    P.save(fig, "fig04_voltage_vs_height_by_soil")


def fig_voltage_height_by_class(df):
    texture_color = P.CLASS_COLORS[:3]
    texture = ["Sandy", "Humus", "Limy"]
    fig, axes = plt.subplots(2, 3, figsize=(P.FULL_WIDTH, 4.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for m, ax in zip(range(N_CLASSES), axes.flat):
        for s in range(N_SOILS):
            part = df[(df["S"] == s + 1) & (df["M"] == m + 1)].sort_values("H")
            humid = s >= 3
            ax.plot(part["H"], part["V"], color=texture_color[s % 3],
                    linestyle="--" if humid else "-", marker=P.CLASS_MARKERS[s % 3],
                    markerfacecolor="white" if humid else texture_color[s % 3], markersize=3.5)
        ax.set_title(CLASS_NAMES[m], loc="left")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = [Line2D([], [], color=texture_color[i], marker=P.CLASS_MARKERS[i], label=t)
               for i, t in enumerate(texture)]
    handles += [Line2D([], [], color=P.INK_2, linestyle="-", label="Dry (solid)"),
                Line2D([], [], color=P.INK_2, linestyle="--", label="Humid (dashed)")]
    legend_ax.legend(handles=handles, loc="center", title="Soil type")
    for ax in axes[0, 2:]:
        ax.tick_params(labelbottom=True)
    fig.supxlabel(H_LABEL, fontsize=8, color=P.INK_2)
    fig.supylabel(V_LABEL, fontsize=8, color=P.INK_2)
    P.save(fig, "fig05_voltage_vs_height_by_class")


def fig_feature_space(df):
    fig, axes = plt.subplots(1, 5, figsize=(P.FULL_WIDTH, 1.9), sharex=True, sharey=True,
                             constrained_layout=True)
    for m, ax in enumerate(axes):
        mine = df["M"] == m + 1
        ax.scatter(df.loc[~mine, "H"], df.loc[~mine, "V"], s=7, color=P.MUTED, linewidths=0)
        ax.scatter(df.loc[mine, "H"], df.loc[mine, "V"], s=12, color=P.CLASS_COLORS[m],
                   marker=P.CLASS_MARKERS[m], linewidths=0)
        ax.set_title(CLASS_NAMES[m], loc="left")
    fig.supxlabel(H_LABEL, fontsize=8, color=P.INK_2)
    axes[0].set_ylabel("V (normalised)")
    P.save(fig, "fig06_feature_space")


# ---------------------------------------------------------------- statistics

def _sse_linear(x, v):
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    return float(((design @ coef - v) ** 2).sum())


def fit_height_laws(df):
    """Pooled within-series R^2 of V against H, for each (class, soil) series.

    Compares a straight line with V = a + b / (H + d0)^3, using one offset d0
    shared by all series (H is normalised, so the zero point of the true
    sensor-to-object distance is unknown). This is descriptive, in-sample.
    """
    series = [g.sort_values("H") for _, g in df.groupby(["M", "S"]) if len(g) >= 3]
    total = sum(float(((g["V"] - g["V"].mean()) ** 2).sum()) for g in series)
    offsets = np.geomspace(0.05, 20.0, 400)
    sse_cube = np.array([
        sum(_sse_linear((g["H"].to_numpy() + d0) ** -3, g["V"].to_numpy()) for g in series)
        for d0 in offsets])
    best = int(sse_cube.argmin())
    sse_lin = sum(_sse_linear(g["H"].to_numpy(), g["V"].to_numpy()) for g in series)
    return {
        "rows": int(sum(len(g) for g in series)),
        "series": len(series),
        "R2_linear_in_H": 1 - sse_lin / total,
        "R2_inverse_cube_shared_offset": 1 - float(sse_cube[best]) / total,
        "best_shared_offset_d0": float(offsets[best]),
        "offset_at_search_limit": bool(best in (0, len(offsets) - 1)),
        "offset_search_range": [float(offsets[0]), float(offsets[-1])],
    }


def height_law_analysis(df):
    rows = []
    for (m, s), g in df.groupby(["M", "S"]):
        g = g.sort_values("H")
        rho, p = stats.spearmanr(g["H"], g["V"])
        rows.append({"class": CLASS_NAMES[m - 1], "soil": SOIL_NAMES[s - 1], "n": len(g),
                     "spearman_rho": rho, "p_value": p,
                     "V_at_lowest_H": g["V"].iloc[0], "V_at_highest_H": g["V"].iloc[-1]})
    table = pd.DataFrame(rows)
    saturated = df["V"] >= SATURATION
    summary = {
        "series": len(table),
        "series_with_negative_rho": int((table["spearman_rho"] < 0).sum()),
        "series_with_rho_below_minus_0.8": int((table["spearman_rho"] < -0.8).sum()),
        "median_spearman_rho_by_class": table.groupby("class", sort=False)["spearman_rho"]
                                             .median().round(3).to_dict(),
        "saturation_threshold": SATURATION,
        "saturated_rows": int(saturated.sum()),
        "saturated_rows_by_class": {CLASS_NAMES[m - 1]: int(n) for m, n in
                                    df.loc[saturated, "M"].value_counts().sort_index().items()},
        "fits": {
            "all_rows": fit_height_laws(df),
            "unsaturated_rows": fit_height_laws(df[~saturated]),
            "unsaturated_mine_rows": fit_height_laws(df[~saturated & (df["M"] > 1)]),
        },
    }
    return table, summary


def fig_height_trend(table):
    grid = table.pivot(index="class", columns="soil", values="spearman_rho")
    grid = grid.loc[CLASS_NAMES, SOIL_NAMES]
    fig, ax = plt.subplots(figsize=(5.0, 2.6), constrained_layout=True)
    image = annotated_heatmap(ax, grid.to_numpy(), P.DIVERGING, "{:+.2f}", -1, 1,
                              lambda v: abs(v) > 0.65)
    ax.set_xticks(range(N_SOILS), [n.replace(" ", "\n") for n in SOIL_NAMES])
    ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_title("Rank correlation between voltage and height", loc="left")
    bar = fig.colorbar(image, ax=ax, shrink=0.9, ticks=[-1, 0, 1])
    bar.outline.set_visible(False)
    P.save(fig, "fig07_height_trend")


def variance_explained(df):
    """In-sample R^2 of linear models for V, to show which factor drives it."""
    blocks = {
        "H": df[["H"]].to_numpy(),
        "S": pd.get_dummies(df["S"]).to_numpy(float),
        "M": pd.get_dummies(df["M"]).to_numpy(float),
    }
    out = {}
    for combo in (["H"], ["S"], ["M"], ["H", "S"], ["M", "H"], ["M", "H", "S"]):
        x = np.hstack([blocks[c] for c in combo])
        out["+".join(combo)] = float(LinearRegression().fit(x, df["V"]).score(x, df["V"]))
    return out


def class_tests(df):
    groups = [g["V"].to_numpy() for _, g in df.groupby("M")]
    h, p = stats.kruskal(*groups)
    pairwise = {}
    pairs = [(a, b) for a in range(N_CLASSES) for b in range(a + 1, N_CLASSES)]
    for a, b in pairs:
        raw = stats.mannwhitneyu(groups[a], groups[b]).pvalue
        pairwise[f"{CLASS_NAMES[a]} vs {CLASS_NAMES[b]}"] = float(min(1.0, raw * len(pairs)))
    soil_p = {}
    for m, g in df.groupby("M"):
        soil_p[CLASS_NAMES[m - 1]] = float(
            stats.kruskal(*[x["V"].to_numpy() for _, x in g.groupby("S")]).pvalue)
    return {"kruskal_V_across_classes": {"H": float(h), "p": float(p)},
            "mannwhitney_V_pairwise_bonferroni_p": pairwise,
            "kruskal_V_across_soils_within_class_p": soil_p}


def main() -> None:
    df = load_raw()
    validate(df)
    P.apply_style()
    rng = np.random.default_rng(0)  # jitter only
    fig_class_balance(df)
    fig_design_coverage(df)
    fig_voltage_distribution(df, rng)
    fig_voltage_height_by_soil(df)
    fig_voltage_height_by_class(df)
    fig_feature_space(df)
    trend_table, trend = height_law_analysis(df)
    fig_height_trend(trend_table)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = df.groupby("M")["V"].describe()
    summary.index = pd.Index(CLASS_NAMES, name="class")
    summary.round(4).to_csv(TABLES_DIR / "voltage_by_class.csv")
    trend_table.round(4).to_csv(TABLES_DIR / "height_trend.csv", index=False)
    report = {"height_law": trend, "V_variance_explained_R2": variance_explained(df),
              "tests": class_tests(df)}
    (TABLES_DIR / "eda_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.round(3).to_string())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
