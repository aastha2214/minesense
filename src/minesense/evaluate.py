"""Stage 3 core evaluation of the nested-CV predictions.

    python -m minesense.evaluate

  * per-model summary with detection accuracy (mine vs no mine)
  * per-class precision and recall, confusion matrices
  * train / validation loss curves of every neural model
  * significance tests: best neural model vs every baseline, physics variants
  * label-permutation check for leakage
  * benchmark against Yilmaz et al. (2018)
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import plotting as P
from .config import CLASS_NAMES, N_CLASSES, NO_MINE, TABLES_DIR
from .cv import CV_DIR
from .dataset import load_data
from .metrics import PROB_COLS, basic_metrics, per_split
from .registry import BASELINE_SPECS, DL_SPECS, MODELS, fit_predict
from .stage2 import BEST_DL, split_name, splits, tuned_params
from .stats import compare_paired, holm, mcnemar_exact

SHORT = ["None", "AT", "AP", "BT-AP", "M14"]


def load_predictions() -> dict[str, pd.DataFrame]:
    out = {}
    for key in MODELS:
        df = pd.read_csv(CV_DIR / key / "predictions.csv")
        df["pred"] = df[PROB_COLS].to_numpy().argmax(1)
        out[key] = df.sort_values(["seed", "fold", "row"]).reset_index(drop=True)
    return out


def detection_accuracy(y, pred) -> float:
    return float(np.mean((y != NO_MINE) == (pred != NO_MINE)))


def summary_table(preds) -> tuple[pd.DataFrame, dict]:
    rows, per = [], {}
    for key, df in preds.items():
        s = per_split(df)
        s["detection"] = [detection_accuracy(g["y"].to_numpy(), g["pred"].to_numpy())
                          for _, g in df.groupby(["seed", "fold"])]
        per[key] = s
        row = {"key": key, "model": MODELS[key].label, "kind": MODELS[key].kind}
        for m in ("accuracy", "macro_f1", "detection", "mine_missed", "false_alarm"):
            row[m] = s[m].mean()
            row[f"{m}_std"] = s[m].std(ddof=1)
        row["accuracy_min"], row["accuracy_max"] = s["accuracy"].min(), s["accuracy"].max()
        rows.append(row)
    return pd.DataFrame(rows), per


def per_class_table(preds) -> pd.DataFrame:
    rows = []
    for key, df in preds.items():
        prec, rec = [], []
        for _, g in df.groupby(["seed", "fold"]):
            p, r, _, _ = precision_recall_fscore_support(g["y"], g["pred"], labels=range(N_CLASSES),
                                                         zero_division=0)
            prec.append(p)
            rec.append(r)
        prec, rec = np.array(prec), np.array(rec)
        for c in range(N_CLASSES):
            rows.append({"model": MODELS[key].label, "class": CLASS_NAMES[c],
                         "precision": prec[:, c].mean(), "precision_std": prec[:, c].std(ddof=1),
                         "recall": rec[:, c].mean(), "recall_std": rec[:, c].std(ddof=1)})
    return pd.DataFrame(rows)


def confusion(df) -> np.ndarray:
    """Row-normalised confusion matrix pooled over all 25 test folds."""
    cm = confusion_matrix(df["y"], df["pred"], labels=range(N_CLASSES)).astype(float)
    return cm / cm.sum(1, keepdims=True)


def fig_confusion(preds, keys) -> None:
    fig, axes = plt.subplots(1, len(keys), figsize=(P.FULL_WIDTH, 2.6), constrained_layout=True)
    for ax, key in zip(axes, keys):
        cm = confusion(preds[key])
        ax.imshow(cm, cmap=P.SEQUENTIAL, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else P.INK)
        ax.set_xticks(range(N_CLASSES), SHORT, fontsize=7)
        ax.set_yticks(range(N_CLASSES), SHORT, fontsize=7)
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(MODELS[key].label.split(" (")[0], loc="left", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7)
    axes[0].set_ylabel("True class (row %)", fontsize=7)
    P.save(fig, "fig10_confusion_matrices")


def fig_loss_curves() -> None:
    keys = [s.key for s in DL_SPECS]
    fig, axes = plt.subplots(2, 3, figsize=(P.FULL_WIDTH, 4.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        hist = json.loads((CV_DIR / key / "histories.json").read_text())
        for name, color in (("train_loss", P.BLUE), ("val_loss", P.ORANGE)):
            curves = [np.array(h[name]) for h in hist.values()]
            for c in curves:
                ax.plot(c, color=color, alpha=0.12, linewidth=0.6)
            # Median over the splits still training at each epoch (while at least half are).
            length = int(np.median([len(c) for c in curves]))
            med = [np.median([c[e] for c in curves if len(c) > e]) for e in range(length)]
            ax.plot(med, color=color, linewidth=1.8,
                    label="train" if name == "train_loss" else "validation")
        best = np.median([h["best_epoch"] for h in hist.values()])
        ax.axvline(best, color=P.INK_2, linewidth=0.8, linestyle="--")
        ax.set_title(MODELS[key].label.replace(" (physics features + loss)", "")
                     .replace(" + physics features", " + phys. feat."), loc="left", fontsize=8)
    axes[0, 0].set_ylim(0, 2.0)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cross-entropy")
    P.save(fig, "fig11_loss_curves")


def significance(preds, per) -> pd.DataFrame:
    rows = []
    pairs = [(BEST_DL, s.key, "best neural vs baseline") for s in BASELINE_SPECS]
    pairs += [(a, b, "physics") for a, b in (("mlp_phys", "mlp"), ("pi_mlp", "mlp"),
                                            ("ft_phys", "ft"), ("pi_ft", "ft"))]
    pairs += [("pi_ft", "pi_mlp", "architecture"), ("ft", "mlp", "architecture")]
    for a, b, family in pairs:
        for metric in ("accuracy", "macro_f1", "mine_missed", "false_alarm"):
            t = compare_paired(per[a][metric].to_numpy(), per[b][metric].to_numpy())
            row = {"family": family, "model_a": MODELS[a].label, "model_b": MODELS[b].label,
                   "metric": metric, **t}
            if metric == "accuracy":
                pa, pb = preds[a], preds[b]
                mc = [mcnemar_exact(*(g["pred"].to_numpy() == g["y"].to_numpy()
                                      for g in (pa[pa.seed == s], pb[pb.seed == s])))
                      for s in sorted(pa.seed.unique())]
                row["mcnemar_p_per_seed"] = ";".join(f"{m['p']:.3g}" for m in mc)
                row["mcnemar_seeds_p_lt_0.05"] = sum(m["p"] < 0.05 for m in mc)
            rows.append(row)
    table = pd.DataFrame(rows)
    # Holm correction within each family of comparisons and metric.
    for (_, _), idx in table.groupby(["family", "metric"]).groups.items():
        for col in ("p_wilcoxon", "p_corrected_t"):
            table.loc[idx, f"{col}_holm"] = holm(table.loc[idx, col].to_numpy())
    return table


def _permuted_run(key, seed, fold):
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    x, soil, y = load_data()
    sp = next(s for s in splits() if (s.seed, s.fold) == (seed, fold))
    rng = np.random.default_rng(10_000 + 10 * seed + fold)
    y_perm = y.copy()
    dev = sp.train_val
    y_perm[dev] = rng.permutation(y[dev])  # labels shuffled among training rows only
    params = tuned_params(key)[split_name(sp)]
    proba, *_ = fit_predict(MODELS[key], params, x, soil, y_perm, sp.train, sp.val, sp.test,
                            seed=1000 * seed + fold)
    return basic_metrics(y[sp.test], proba.argmax(1))["accuracy"]


def permutation_check(keys=("svm", "pi_mlp", "pi_ft")) -> pd.DataFrame:
    """If labels are shuffled, any accuracy clearly above 1/5 would mean leakage."""
    rows = []
    with ProcessPoolExecutor(10) as pool:
        for key in keys:
            jobs = [(key, sp.seed, sp.fold) for sp in splits()]
            acc = np.array(list(pool.map(_permuted_run, *zip(*jobs))))
            rows.append({"model": MODELS[key].label, "accuracy_mean": acc.mean(),
                         "accuracy_std": acc.std(ddof=1), "accuracy_max": acc.max(),
                         "chance_level": 1 / N_CLASSES})
    return pd.DataFrame(rows)


# Values as stated in the text of Yilmaz, Kahraman and Soyler, IEEE Access 6 (2018);
# page numbers refer to the journal pagination.
PAPER_ROWS = [
    {"source": "Yilmaz et al. 2018", "model": "Meta-heuristic k-NN (fuzzy metric, k=4)",
     "protocol": "one random split, 113 test samples; feature weights optimised with a "
                 "fitness computed on 'test observations' (p. 47877)",
     "classification_accuracy": "85.8% (16 of 113 wrong, p. 47884)",
     "detection_accuracy": "98.2% (2 of 113 wrong, p. 47883)"},
    {"source": "Yilmaz et al. 2018", "model": "ANN, inputs V, H, S",
     "protocol": "same split", "classification_accuracy": "71.3% (p. 47881)",
     "detection_accuracy": "not stated in the text"},
    {"source": "Yilmaz et al. 2018", "model": "Starting point the paper improves on",
     "protocol": "as stated: 'This rate has reached up to 85.8% from 57.7%'",
     "classification_accuracy": "57.7% (p. 47871)", "detection_accuracy": "-"},
]


def benchmark(summary: pd.DataFrame) -> pd.DataFrame:
    rows = list(PAPER_ROWS)
    for _, r in summary.sort_values("accuracy", ascending=False).iterrows():
        rows.append({
            "source": "This work", "model": r["model"],
            "protocol": "nested CV, stratified 5-fold x 5 seeds (25 test folds of 67-68)",
            "classification_accuracy": f"{100 * r['accuracy']:.1f} ± {100 * r['accuracy_std']:.1f}% "
                                       f"(single folds {100 * r['accuracy_min']:.1f}-"
                                       f"{100 * r['accuracy_max']:.1f}%)",
            "detection_accuracy": f"{100 * r['detection']:.1f} ± {100 * r['detection_std']:.1f}%"})
    return pd.DataFrame(rows)


def main() -> None:
    P.apply_style()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    preds = load_predictions()
    summary, per = summary_table(preds)
    summary.to_csv(TABLES_DIR / "stage3_summary.csv", index=False, float_format="%.4f")
    print(summary[["model", "accuracy", "macro_f1", "detection", "mine_missed", "false_alarm",
                   "accuracy_min", "accuracy_max"]].round(3).to_string(index=False))

    pc = per_class_table(preds)
    pc.to_csv(TABLES_DIR / "per_class_metrics.csv", index=False, float_format="%.4f")
    print(pc.pivot(index="model", columns="class", values="recall")[CLASS_NAMES].round(3))
    cms = {MODELS[k].label: confusion(df).round(4).tolist() for k, df in preds.items()}
    (TABLES_DIR / "confusion_matrices.json").write_text(json.dumps(cms, indent=1) + "\n")
    fig_confusion(preds, [BEST_DL, "mlp", "svm", "rf"])
    fig_loss_curves()

    sig = significance(preds, per)
    sig.to_csv(TABLES_DIR / "significance_tests.csv", index=False, float_format="%.4g")
    print(sig[["model_a", "model_b", "metric", "mean_diff", "wins", "losses", "p_wilcoxon",
               "p_corrected_t", "p_corrected_t_holm", "mcnemar_seeds_p_lt_0.05"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    perm = permutation_check()
    perm.to_csv(TABLES_DIR / "permutation_check.csv", index=False, float_format="%.4f")
    print(perm.round(3).to_string(index=False))

    bench = benchmark(summary)
    bench.to_csv(TABLES_DIR / "benchmark_yilmaz2018.csv", index=False)
    print(bench[["source", "model", "classification_accuracy", "detection_accuracy"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
