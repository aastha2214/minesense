"""Unit tests for the statistics and calibration helpers."""
import numpy as np

import minesense  # noqa: F401  (sets library load order)
from minesense.reliability import accuracy_at_coverage, aurc, ece
from minesense.stats import compare_paired, corrected_t_test, holm, mcnemar_exact


def test_corrected_t_is_more_conservative_than_plain_t():
    from scipy import stats

    rng = np.random.default_rng(0)
    d = rng.normal(0.02, 0.05, 25)
    plain = stats.ttest_1samp(d, 0).pvalue
    assert corrected_t_test(d) > plain


def test_identical_models_give_p_one():
    a = np.linspace(0.6, 0.8, 25)
    r = compare_paired(a, a)
    assert r["p_wilcoxon"] == 1.0 and r["p_corrected_t"] == 1.0 and r["ties"] == 25


def test_holm_matches_hand_computation():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_mcnemar_counts_discordant_pairs():
    a = np.array([1, 1, 1, 0, 0], bool)
    b = np.array([1, 0, 0, 0, 1], bool)
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (2, 1)


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(p, np.array([0, 1])) == 0.0


def test_selective_prediction_rewards_good_ranking():
    correct = np.array([True, True, False, False])
    good, bad = np.array([4, 3, 2, 1]), np.array([1, 2, 3, 4])
    assert aurc(correct, good) < aurc(correct, bad)
    assert accuracy_at_coverage(correct, good, 0.5) == 1.0
