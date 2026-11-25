"""Uncertainty (MC dropout) and calibration (ECE, temperature scaling).

    python -m minesense.reliability

Uses the models fitted in Stage 2. For each outer split:
  * temperature T is fitted on the split's validation rows (never the test fold);
  * MC dropout: 50 stochastic forward passes with dropout active (batch norm
    stays in inference mode); predictive entropy and mutual information;
  * "uncertain - inspect manually" flag: MC predictive entropy above the 80th
    percentile of the same quantity on the validation rows, so roughly one
    reading in five is referred to a human, with the threshold fixed before the
    test fold is seen.
Baselines are scored for calibration as they are (the SVM already uses Platt
scaling); they have no held-out validation rows to fit a temperature on.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from . import plotting as P
from .config import NO_MINE, RESULTS_DIR, TABLES_DIR
from .cv import CV_DIR
from .dataset import load_data
from .registry import BASELINE_SPECS, DL_SPECS, MODELS
from .metrics import PROB_COLS
from .stage2 import BEST_DL, fitted_model, splits
from .stats import compare_paired

MC_PASSES = 50
FLAG_QUANTILE = 0.80
N_BINS = 10
OUT_DIR = RESULTS_DIR / "reliability"


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def nll(p, y) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def brier(p, y) -> float:
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


def entropy(p) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


@torch.no_grad()
def logits_of(model, x, soil) -> torch.Tensor:
    model.eval()
    return model(torch.as_tensor(x), torch.as_tensor(soil))


def fit_temperature(logits: torch.Tensor, y: np.ndarray) -> float:
    yt = torch.as_tensor(y)

    def loss(log_t):
        return float(F.cross_entropy(logits / np.exp(log_t), yt))

    return float(np.exp(minimize_scalar(loss, bounds=(-3, 3), method="bounded").x))


@torch.no_grad()
def mc_dropout(model, x, soil, passes: int, seed: int) -> np.ndarray:
    """(passes, n, classes) probabilities with dropout layers sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.MultiheadAttention)):
            m.train()
    torch.manual_seed(seed)
    xt, st = torch.as_tensor(x), torch.as_tensor(soil)
    out = np.stack([torch.softmax(model(xt, st), 1).numpy() for _ in range(passes)])
    model.eval()
    return out


def run_dl(key: str) -> pd.DataFrame:
    torch.set_num_threads(1)  # tiny models; avoids the macOS OpenMP clash seen with SHAP
    x, soil, y = load_data()
    frames = []
    for sp in splits():
        model = fitted_model(key, sp)
        seed = 1000 * sp.seed + sp.fold
        lv, lt = logits_of(model, x[sp.val], soil[sp.val]), logits_of(model, x[sp.test], soil[sp.test])
        temp = fit_temperature(lv, y[sp.val])
        mc_val = mc_dropout(model, x[sp.val], soil[sp.val], MC_PASSES, seed).mean(0)
        mc = mc_dropout(model, x[sp.test], soil[sp.test], MC_PASSES, seed + 1)
        mean = mc.mean(0)
        h = entropy(mean)
        threshold = np.quantile(entropy(mc_val), FLAG_QUANTILE)
        det = torch.softmax(lt, 1).numpy()
        # The deployed flag: one deterministic pass, same 80th-percentile rule.
        det_threshold = np.quantile(entropy(torch.softmax(lv, 1).numpy()), FLAG_QUANTILE)
        df = pd.DataFrame({"model": key, "seed": sp.seed, "fold": sp.fold, "row": sp.test,
                           "y": y[sp.test], "temperature": temp,
                           "entropy": h, "mutual_info": h - entropy(mc).mean(0),
                           "threshold": threshold, "flagged": h > threshold,
                           "det_entropy": entropy(det), "det_threshold": det_threshold,
                           "flagged_det": entropy(det) > det_threshold})
        ts = torch.softmax(lt / temp, 1).numpy()
        for name, p in (("det", det), ("ts", ts), ("mc", mean)):
            for c in range(p.shape[1]):
                df[f"{name}_p{c}"] = p[:, c]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def probs(df, name) -> np.ndarray:
    return df[[f"{name}_p{c}" for c in range(5)]].to_numpy()


def per_seed(df, fn, name) -> float:
    """Metric computed on each seed's 338 out-of-fold rows, averaged over seeds."""
    return float(np.mean([fn(probs(g, name), g["y"].to_numpy()) for _, g in df.groupby("seed")]))


def calibration_table(dl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, df in dl.groupby("model", sort=False):
        row = {"model": MODELS[key].label, "median_temperature": df.groupby(["seed", "fold"])
               ["temperature"].first().median()}
        for name, label in (("det", "raw"), ("ts", "temp_scaled"), ("mc", "mc_dropout")):
            row[f"ece_{label}"] = per_seed(df, ece, name)
            row[f"nll_{label}"] = per_seed(df, nll, name)
            row[f"brier_{label}"] = per_seed(df, brier, name)
        rows.append(row)
    for spec in BASELINE_SPECS:
        cv = pd.read_csv(CV_DIR / spec.key / "predictions.csv")
        cv = cv.rename(columns={c: f"det_{c}" for c in PROB_COLS})
        rows.append({"model": spec.label, "ece_raw": per_seed(cv, ece, "det"),
                     "nll_raw": per_seed(cv, nll, "det"), "brier_raw": per_seed(cv, brier, "det")})
    return pd.DataFrame(rows)


def uncertainty_table(dl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, df in dl.groupby("model", sort=False):
        y = df["y"].to_numpy()
        mine = y != NO_MINE
        for policy, name, flag_col in (("MC dropout (50 passes)", "mc", "flagged"),
                                       ("softmax entropy (1 pass)", "det", "flagged_det")):
            pred = probs(df, name).argmax(1)
            flag = df[flag_col].to_numpy()
            rows.append({
                "model": MODELS[key].label, "flag_policy": policy,
                "accuracy_all": (pred == y).mean(),
                "flagged_share": flag.mean(),
                "accuracy_confident": (pred[~flag] == y[~flag]).mean(),
                "accuracy_flagged": (pred[flag] == y[flag]).mean(),
                "mine_missed_all": (pred[mine] == NO_MINE).mean(),
                # A missed mine that was flagged goes to manual inspection, so the
                # residual risk is a mine predicted empty *and* not flagged.
                "mine_missed_unflagged": ((pred == NO_MINE) & ~flag)[mine].mean(),
                "aurc": aurc(pred == y, -df["entropy" if name == "mc" else "det_entropy"]
                             .to_numpy()),
                "median_mutual_info": df["mutual_info"].median() if name == "mc" else np.nan})
    return pd.DataFrame(rows)


def risk_coverage(correct: np.ndarray, score: np.ndarray):
    """Error rate among the most confident fraction, for every coverage level."""
    order = np.argsort(-score, kind="stable")
    err = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    return np.arange(1, len(order) + 1) / len(order), err


def aurc(correct, score) -> float:
    return float(risk_coverage(np.asarray(correct), np.asarray(score))[1].mean())


def accuracy_at_coverage(correct, score, coverage) -> float:
    order = np.argsort(-np.asarray(score), kind="stable")
    k = max(1, int(round(coverage * len(order))))
    return float(np.asarray(correct)[order[:k]].mean())


def selective_table(dl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selective prediction per split: AURC and accuracy when only the most
    confident 50% / 80% of readings are decided automatically."""
    rows = []
    for key, df in dl.groupby("model", sort=False):
        for (seed, fold), g in df.groupby(["seed", "fold"]):
            y = g["y"].to_numpy()
            for name, correct, score in (
                    ("MC-dropout entropy", probs(g, "mc").argmax(1) == y, -g["entropy"].to_numpy()),
                    ("max softmax", probs(g, "det").argmax(1) == y, probs(g, "det").max(1))):
                rows.append({"key": key, "score": name, "seed": seed, "fold": fold,
                             "aurc": aurc(correct, score),
                             "acc_at_50": accuracy_at_coverage(correct, score, 0.5),
                             "acc_at_80": accuracy_at_coverage(correct, score, 0.8)})
    for spec in BASELINE_SPECS:
        cv = pd.read_csv(CV_DIR / spec.key / "predictions.csv")
        for (seed, fold), g in cv.groupby(["seed", "fold"]):
            p, y = g[PROB_COLS].to_numpy(), g["y"].to_numpy()
            correct, score = p.argmax(1) == y, p.max(1)
            rows.append({"key": spec.key, "score": "max probability", "seed": seed, "fold": fold,
                         "aurc": aurc(correct, score),
                         "acc_at_50": accuracy_at_coverage(correct, score, 0.5),
                         "acc_at_80": accuracy_at_coverage(correct, score, 0.8)})
    per = pd.DataFrame(rows).sort_values(["key", "score", "seed", "fold"])
    summary = (per.groupby(["key", "score"], sort=False)[["aurc", "acc_at_50", "acc_at_80"]]
               .agg(["mean", "std"]).reset_index())
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.insert(0, "model", summary.pop("key").map(lambda k: MODELS[k].label))
    best = per[(per.key == BEST_DL) & (per.score == "MC-dropout entropy")]
    tests = []
    for spec in BASELINE_SPECS:
        base = per[per.key == spec.key]
        for metric in ("aurc", "acc_at_50", "acc_at_80"):
            tests.append({"baseline": spec.label, "metric": metric,
                          **compare_paired(best[metric].to_numpy(), base[metric].to_numpy())})
    return summary, pd.DataFrame(tests)


def reliability_curve(p, y, n_bins=N_BINS):
    conf, correct = p.max(1), p.argmax(1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = [(conf[bins == b].mean(), correct[bins == b].mean(), (bins == b).sum())
           for b in range(n_bins) if (bins == b).any()]
    return np.array(out)


def fig_reliability(dl: pd.DataFrame) -> None:
    best = dl[dl["model"] == BEST_DL]
    svm = pd.read_csv(CV_DIR / "svm" / "predictions.csv")
    panels = [(f"{MODELS[BEST_DL].label.split(' (')[0]}, raw", probs(best, "det"), best),
              ("... after temperature scaling", probs(best, "ts"), best),
              ("... MC dropout mean", probs(best, "mc"), best),
              ("SVM (Platt scaling)", svm[PROB_COLS].to_numpy(), svm)]
    fig, axes = plt.subplots(1, 4, figsize=(P.FULL_WIDTH, 2.2), sharey=True,
                             constrained_layout=True)
    for ax, (title, p, frame) in zip(axes, panels):
        y, seed = frame["y"].to_numpy(), frame["seed"].to_numpy()
        curve = reliability_curve(p, y)  # pooled over all 25 test folds
        ax.plot([0, 1], [0, 1], color=P.AXIS, linewidth=1, linestyle="--")
        ax.bar(curve[:, 0], curve[:, 1], width=0.07, color=P.BLUE, alpha=0.85)
        e = np.mean([ece(p[seed == s], y[seed == s]) for s in np.unique(seed)])
        ax.text(0.04, 0.9, f"ECE {e:.3f}", fontsize=7, transform=ax.transAxes)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=7)
    axes[0].set_ylabel("Accuracy", fontsize=7)
    P.save(fig, "fig12_reliability_diagrams")


def fig_risk_coverage(dl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
    best = dl[dl["model"] == BEST_DL]
    y = best["y"].to_numpy()
    mc_correct = probs(best, "mc").argmax(1) == y
    det = probs(best, "det")
    svm = pd.read_csv(CV_DIR / "svm" / "predictions.csv").sort_values(["seed", "fold", "row"])
    sp_ = svm[PROB_COLS].to_numpy()
    for label, correct, score, color in (
            ("PI-FT, MC-dropout entropy", mc_correct, -best["entropy"].to_numpy(), P.BLUE),
            ("PI-FT, max softmax", det.argmax(1) == y, det.max(1), P.ORANGE),
            ("SVM, max probability", sp_.argmax(1) == svm["y"].to_numpy(), sp_.max(1), P.INK_2)):
        cov, err = risk_coverage(correct, score)
        ax.plot(cov, 1 - err, color=color, linewidth=1.5, label=label)
    flag_share = best["flagged"].mean()
    ax.axvline(1 - flag_share, color=P.AXIS, linestyle="--", linewidth=0.8)
    ax.text(1 - flag_share - 0.02, 0.52, "flag threshold", rotation=90, fontsize=6,
            color=P.INK_2, ha="right")
    ax.set_xlabel("Coverage (share of readings decided automatically)")
    ax.set_ylabel("Accuracy on those readings")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Accuracy vs. referral to manual inspection", loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    P.save(fig, "fig13_risk_coverage")


def main() -> None:
    P.apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    dl = pd.concat([run_dl(s.key) for s in DL_SPECS], ignore_index=True)
    dl.to_csv(OUT_DIR / "dl_predictions.csv", index=False, float_format="%.6f")
    # Sanity: deterministic test probabilities must equal the Stage 2 predictions.
    for key, df in dl.groupby("model"):
        ref = pd.read_csv(CV_DIR / key / "predictions.csv").sort_values(["seed", "fold", "row"])
        diff = np.abs(probs(df.sort_values(["seed", "fold", "row"]), "det")
                      - ref[PROB_COLS].to_numpy()).max()
        assert diff < 1e-5, (key, diff)
    cal = calibration_table(dl)
    cal.to_csv(TABLES_DIR / "calibration.csv", index=False, float_format="%.4f")
    unc = uncertainty_table(dl)
    unc.to_csv(TABLES_DIR / "mc_dropout_uncertainty.csv", index=False, float_format="%.4f")
    print(cal.round(3).to_string(index=False))
    print(unc.round(3).to_string(index=False))
    sel, sel_tests = selective_table(dl)
    sel.to_csv(TABLES_DIR / "selective_prediction.csv", index=False, float_format="%.4f")
    sel_tests.to_csv(TABLES_DIR / "selective_prediction_tests.csv", index=False,
                     float_format="%.4g")
    print(sel.round(3).to_string(index=False))
    print(f"{MODELS[BEST_DL].label} (MC-dropout entropy) minus baseline, paired over 25 splits")
    print(sel_tests[["baseline", "metric", "mean_diff", "wins", "losses", "p_wilcoxon",
                     "p_corrected_t"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    fig_reliability(dl)
    fig_risk_coverage(dl)


if __name__ == "__main__":
    main()
