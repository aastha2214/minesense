"""Architecture diagram, README banner, one-slide summary and demo GIF.

    python -m minesense.docs_figures

Every number drawn on the summary slide is read from results/tables.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from . import plotting as P
from .config import FIGURES_DIR, ROOT, TABLES_DIR

DOCS = ROOT / "docs"
APP = ROOT / "app"


def box(ax, x, y, w, h, title, body="", color=P.BLUE, fill="#eef5fd"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=1.2, edgecolor=color, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.5,
            fontweight="bold", color=P.INK)
    if body:
        ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=6.2,
                color=P.INK, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=P.INK_2):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1,
                            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8})


def architecture() -> None:
    fig = plt.figure(figsize=(P.FULL_WIDTH, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top, h = 0.50, 0.42
    box(ax, 0.010, top, 0.115, h, "Sensor reading",
        "V  magnetic anomaly\n(normalised voltage)\n\nH  sensor height\n\nS  soil type\n(6 classes)",
        color=P.INK_2, fill="#f4f4f4")
    box(ax, 0.150, top, 0.200, h, "Physics feature layer",
        "r = H + d0   (d0 fitted on\ntraining rows, inverse-cube law)\n"
        "dV = V - b(soil)  (soil background)\n\n"
        "log V,  V r³,  V / r,  dV r³,\nsaturation flag\n→ standardised (training stats)")
    box(ax, 0.375, top, 0.165, h, "Feature tokenizer",
        "each numeric feature i:\ntoken = xᵢ wᵢ + bᵢ\n\nsoil → learned embedding\ntoken\n\n+ [CLS] token")
    box(ax, 0.565, top, 0.170, h, "Transformer blocks ×N",
        "pre-norm\nmulti-head self-attention\n+ feed-forward (GELU)\n\nattention / FFN dropout\n"
        "(N, width, dropout\ntuned by Optuna)")
    box(ax, 0.760, top, 0.230, h, "Output",
        "[CLS] → LayerNorm → ReLU →\nLinear → softmax over 5 classes\n\n"
        "predictive entropy > threshold\n→ 'uncertain - inspect manually'\n\n"
        "exported to ONNX (FP32 / INT8)")
    for x0, x1 in ((0.125, 0.150), (0.350, 0.375), (0.540, 0.565), (0.735, 0.760)):
        arrow(ax, x0, top + h / 2, x1, top + h / 2)
    box(ax, 0.150, 0.10, 0.840, 0.30, "Training objective (AdamW, early stopping on validation loss)",
        "", color=P.ORANGE, fill="#fef3ec")
    items = [
        ("Cross-entropy", "5 classes:\nno mine + 4 mine types"),
        ("Safety penalty", "λ · mean over true mines of\n−log(1 − p(no mine))\nλ = 2, fixed in advance"),
        ("Physics consistency", "move sensor by δ ~ U(−0.25, 0.25):\nV' = b + (V − b)(r / r')³\n"
                                "KL(p(x) ‖ p(x')), weight tuned"),
        ("Augmentation", "Gaussian input noise,\nscale tuned per split"),
    ]
    for i, (t, b) in enumerate(items):
        x = 0.165 + i * 0.207
        ax.text(x + 0.095, 0.30, t, ha="center", va="top", fontsize=7, fontweight="bold",
                color=P.INK)
        ax.text(x + 0.095, 0.245, b, ha="center", va="top", fontsize=6.2, color=P.INK,
                linespacing=1.35)
    ax.text(0.068, 0.23, "PI-FT-Transformer\n(best neural model)", ha="center", va="center",
            fontsize=7, color=P.INK_2, style="italic")
    P.save(fig, "fig00_architecture")


def banner() -> None:
    sys.path.insert(0, str(APP))
    import minefield as mf
    from inference import meta, predict

    m = meta()
    fld = mf.make_field(10, 40, 0.14, m["simulator"]["h_levels"][2], 1.0, seed=11)
    path = mf.sweep_path(10, 40)
    mf.scan(fld, path[:300], m["simulator"], lambda v, h, s: predict(v, h, s))
    risk = np.where(fld.scanned, 1 - np.nan_to_num(fld.probs[..., 0]), np.nan)
    fig = plt.figure(figsize=(12, 3), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    cmap = P.SEQUENTIAL.copy()
    cmap.set_bad("#eef1f4")
    ax.imshow(risk, cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(0, 40, 10, 0))
    for r in range(11):
        ax.axhline(r, color="white", lw=2)
    for c in range(41):
        ax.axvline(c, color="white", lw=2)
    ax.add_patch(FancyBboxPatch((1.0, 2.3), 19.5, 5.4, boxstyle="round,pad=0.2,rounding_size=0.4",
                                facecolor="white", edgecolor="none", alpha=0.94))
    ax.text(1.8, 4.3, "MineSense", fontsize=44, fontweight="bold", color=P.INK, va="center")
    ax.text(1.8, 6.4, "Physics-informed deep learning for passive landmine\n"
                      "detection and classification, with an uncertainty flag",
            fontsize=13, color=P.INK_2, va="center", linespacing=1.4)
    ax.set_xlim(0, 40)
    ax.set_ylim(10, 0)
    ax.axis("off")
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / "banner.png", dpi=150)
    plt.close(fig)


def summary_slide() -> None:
    summ = pd.read_csv(TABLES_DIR / "stage3_summary.csv").sort_values("accuracy")
    noise = pd.read_csv(TABLES_DIR / "noise_robustness.csv")
    loso = pd.read_csv(TABLES_DIR / "loso_summary.csv")
    sel = pd.read_csv(TABLES_DIR / "selective_prediction.csv")
    edge = json.loads((TABLES_DIR / "edge_export.json").read_text())
    best = "PI-FT-Transformer (physics features + loss)"

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.02, 0.94, "MineSense: results at a glance", fontsize=24, fontweight="bold",
             color=P.INK)
    fig.text(0.02, 0.895, "UCI Land Mines data (338 readings, 5 classes). Nested CV: stratified "
             "5-fold × 5 seeds, Optuna tuning inside each training fold.", fontsize=12,
             color=P.INK_2)
    gs = fig.add_gridspec(2, 3, left=0.14, right=0.98, top=0.83, bottom=0.08, hspace=0.45,
                          wspace=0.35)

    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(summ))
    colors = [P.BLUE if k in ("mlp", "mlp_phys", "pi_mlp", "ft", "ft_phys", "pi_ft") else P.MUTED
              for k in summ["key"]]
    ax.barh(y, summ["accuracy"], xerr=summ["accuracy_std"], color=colors, height=0.6,
            error_kw={"elinewidth": 0.8, "ecolor": P.INK_2})
    ax.set_yticks(y, [m.replace(" (physics features + loss)", "").replace(" + physics features",
                                                                           " + phys.")
                      for m in summ["model"]], fontsize=9)
    for yi, v, col in zip(y, summ["accuracy"], colors):
        ax.text(0.02, yi, f"{v:.3f}", va="center", fontsize=8,
                color="white" if col == P.BLUE else P.INK)
    ax.set_xlim(0, 1)
    ax.set_title("5-class accuracy (mean ± std, 25 folds)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(gs[0, 1])
    for model, color, label in ((best, P.BLUE, "PI-FT-Transformer"), ("SVM (RBF)", P.INK_2, "SVM")):
        t = noise[noise["model"] == model]
        ax.plot(t["sigma"], t["mine_missed_mean"], "-o", color=color, ms=3.5, label=label)
    ax.set_xlabel("Input noise (× training std)")
    ax.set_ylabel("Share of mines missed")
    ax.set_title("Under sensor noise, fewer mines missed", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    s = sel.set_index(["model", "score"])
    vals = [(best, "MC-dropout entropy", "PI-FT"), ("SVM (RBF)", "max probability", "SVM")]
    acc50 = [s.loc[(m, k), "acc_at_50_mean"] for m, k, _ in vals]
    acc100 = [summ.set_index("model").loc[m, "accuracy"] for m, _, _ in vals]
    xx = np.arange(2)
    ax.bar(xx - 0.18, acc100, 0.34, color=P.MUTED, label="all readings")
    ax.bar(xx + 0.18, acc50, 0.34, color=P.BLUE, label="50% most confident")
    for xi, a, b in zip(xx, acc100, acc50):
        ax.text(xi - 0.18, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xx, [v[2] for v in vals])
    ax.set_ylim(0.5, 1.1)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title("Refer uncertain readings, gain accuracy", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[1, 1])
    l = loso.set_index("model")
    pairs = [("mlp_none", "MLP"), ("svm", "SVM"), ("pi_ft_factor", "PI-FT")]
    seen = [l.loc[k, "main_cv_accuracy_soil_seen"] for k, _ in pairs]
    unseen = [float(l.loc[k, "accuracy"].split(" ")[0]) for k, _ in pairs]
    xx = np.arange(len(pairs))
    ax.bar(xx - 0.18, seen, 0.34, color=P.MUTED, label="soil seen in training")
    ax.bar(xx + 0.18, unseen, 0.34, color=P.ORANGE, label="soil never seen")
    ax.set_xticks(xx, [p[1] for p in pairs])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Accuracy")
    ax.set_title("Open problem: unseen soils", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    t = edge["timing_one_cpu_thread"]
    lines = [("Edge model", f"{edge['parameters']:,} parameters"),
             ("ONNX INT8 size", f"{edge['size_kb']['onnx_int8']:.0f} KB"),
             ("Latency, 1 reading", f"{1000 * t['onnx_int8']['batch1_ms']:.0f} µs (laptop, 1 thread)"),
             ("Best accuracy", f"SVM {summ.set_index('key').loc['svm', 'accuracy']:.3f}, "
                               f"PI-FT {summ.set_index('key').loc['pi_ft', 'accuracy']:.3f}"),
             ("", "difference not significant"),
             ("Leakage check", "shuffled labels → chance level")]
    ax.set_title("Deployment and honesty checks", loc="left", fontsize=11)
    for i, (k, v) in enumerate(lines):
        ax.text(0.0, 0.88 - i * 0.16, k, fontsize=10, fontweight="bold", color=P.INK,
                transform=ax.transAxes)
        ax.text(0.45, 0.88 - i * 0.16, v, fontsize=10, color=P.INK, transform=ax.transAxes)
    P.save(fig, "fig19_summary_slide")


def demo_gif(frames_per_row: int = 1) -> None:
    from PIL import Image

    sys.path.insert(0, str(APP))
    import minefield as mf
    from inference import flagged, meta, predict

    m = meta()
    rows, cols = 10, 14
    # Mid sweep height and 2x the fitted sensor noise, as stated in the README caption.
    fld = mf.make_field(rows, cols, 0.16, m["simulator"]["h_levels"][5], 2.0, seed=7)
    path = mf.sweep_path(rows, cols)
    soil_colors = ["#f6d9a8", "#b08b5b", "#e8e3d3", "#c9a66b", "#6f5232", "#bdb6a0"]
    cmap = P.SEQUENTIAL.copy()
    cmap.set_bad("#eef1f4")
    images = []
    steps = list(range(0, rows * cols + 1, 2)) + [rows * cols] * 12
    for step in steps:
        done = path[:step]
        if done:
            todo = [c for c in done if not fld.scanned[c]]
            if todo:
                mf.scan(fld, todo, m["simulator"], lambda v, h, s: predict(v, h, s))
        final = step == rows * cols
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.2, 3.9), dpi=110,
                                     gridspec_kw={"width_ratios": [1, 2.6]})
        a0.imshow(np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:], 16)]
                            for c in soil_colors])[fld.soil] / 255)
        a0.set_title("Soil map", loc="left", fontsize=9)
        risk = np.where(fld.scanned, 1 - np.nan_to_num(fld.probs[..., 0]), np.nan)
        a1.imshow(risk, cmap=cmap, vmin=0, vmax=1)
        for k in range(rows + 1):
            a1.axhline(k - 0.5, color="white", lw=1.5)
        for k in range(cols + 1):
            a1.axvline(k - 0.5, color="white", lw=1.5)
        p = np.nan_to_num(fld.probs.reshape(-1, 5))
        flag = np.zeros(rows * cols, bool)
        flag[fld.scanned.ravel()] = flagged(p[fld.scanned.ravel()])
        for k in np.flatnonzero(flag):
            a1.text(k % cols, k // cols, "?", ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#f2a600")
        if done and not final:
            r, c = done[-1]
            a1.add_patch(plt.Circle((c, r), 0.45, fill=False, color=P.ORANGE, lw=2))
        if final:
            rr, cc = np.nonzero(fld.truth)
            a1.scatter(cc, rr, s=150, facecolors="none", edgecolors=P.INK, linewidths=1.8)
            s = mf.score(fld, flagged)
            a1.set_title(f"Revealed: {s['found']}/{s['mines']} mines found, {s['missed']} missed, "
                         f"{s['false_alarms']} false alarms, {s['flagged']} flagged '?'",
                         loc="left", fontsize=8.5)
        else:
            a1.set_title("Detector sweep: live P(mine); ? = inspect manually", loc="left",
                         fontsize=9)
        for a in (a0, a1):
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    DOCS.mkdir(exist_ok=True)
    images[0].save(DOCS / "demo_minefield.gif", save_all=True, append_images=images[1:],
                   duration=120, loop=0, optimize=True)


def main() -> None:
    P.apply_style()
    architecture()
    banner()
    summary_slide()
    demo_gif()


if __name__ == "__main__":
    main()
