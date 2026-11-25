"""Leakage guards. These run on synthetic labels, so no data download is needed."""
import numpy as np

from minesense.config import N_FOLDS, SEEDS
from minesense.splits import outer_splits

Y = np.repeat(np.arange(5), [71, 70, 66, 66, 65])


def test_parts_are_disjoint_and_cover_all_rows():
    for sp in outer_splits(Y):
        joined = np.concatenate([sp.train, sp.val, sp.test])
        assert len(joined) == len(np.unique(joined)) == len(Y)


def test_each_row_is_tested_once_per_seed():
    tested = {seed: [] for seed in SEEDS}
    for sp in outer_splits(Y):
        tested[sp.seed].extend(sp.test)
    for rows in tested.values():
        assert sorted(rows) == list(range(len(Y)))


def test_splits_are_reproducible():
    first, second = list(outer_splits(Y)), list(outer_splits(Y))
    assert len(first) == N_FOLDS * len(SEEDS)
    for a, b in zip(first, second):
        assert np.array_equal(a.train, b.train)
        assert np.array_equal(a.val, b.val)
        assert np.array_equal(a.test, b.test)


def test_every_class_is_in_every_part():
    for sp in outer_splits(Y):
        for part in (sp.train, sp.val, sp.test):
            assert len(np.unique(Y[part])) == 5
