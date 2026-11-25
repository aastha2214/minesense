"""Paired significance tests for repeated cross-validation.

Per-split scores from repeated k-fold CV are not independent (training sets
overlap), so a plain paired t-test or Wilcoxon test is too liberal. We report
the Wilcoxon signed-rank test, as requested, alongside the corrected
repeated-k-fold t-test of Nadeau and Bengio (2003) as given by Bouckaert and
Frank (2004), which inflates the variance by n_test / n_train. McNemar's exact
test compares two models row by row within one seed, where every row is
tested exactly once, so its pairs are genuinely disjoint.
"""
import numpy as np
from scipy import stats

from .config import N_FOLDS

TEST_TRAIN_RATIO = 1 / (N_FOLDS - 1)  # each test fold is 1/k of the data


def corrected_t_test(diff: np.ndarray, ratio: float = TEST_TRAIN_RATIO) -> float:
    """Two-sided p-value of the corrected repeated k-fold CV t-test."""
    diff = np.asarray(diff, float)
    j = len(diff)
    var = diff.var(ddof=1)
    if var == 0:
        return 1.0 if diff.mean() == 0 else 0.0
    t = diff.mean() / np.sqrt((1 / j + ratio) * var)
    return float(2 * stats.t.sf(abs(t), df=j - 1))


def wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, float)
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff, zero_method="zsplit").pvalue)


def compare_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """a and b are per-split scores of two models on identical splits."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
            "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
            "losses": int((d < -1e-12).sum()),
            "p_wilcoxon": wilcoxon(d), "p_corrected_t": corrected_t_test(d)}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on the discordant pairs."""
    b = int(np.sum(correct_a & ~correct_b))   # only model A right
    c = int(np.sum(~correct_a & correct_b))   # only model B right
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"a_only": b, "b_only": c, "p": p}


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
