"""The single source of truth for train / validation / test splits.

Every model uses exactly these splits, which is what makes paired statistical
tests valid. Anything learned from data (scaling, augmentation noise, early
stopping, hyperparameters, calibration temperature) must use only ``train`` and
``val``; ``test`` is used once, for scoring.
"""
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import N_FOLDS, SEEDS, VAL_FRACTION


@dataclass(frozen=True)
class Split:
    seed: int
    fold: int
    train: np.ndarray  # fit parameters
    val: np.ndarray    # early stopping and calibration
    test: np.ndarray   # scoring only

    @property
    def train_val(self) -> np.ndarray:
        """All non-test rows, for models that need no validation set."""
        return np.sort(np.concatenate([self.train, self.val]))


def outer_splits(y: np.ndarray, seeds: list[int] = SEEDS) -> Iterator[Split]:
    """Stratified K-fold repeated once per seed."""
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (dev, test) in enumerate(skf.split(np.zeros(len(y)), y)):
            train, val = train_test_split(
                dev,
                test_size=VAL_FRACTION,
                stratify=y[dev],
                random_state=1000 * seed + fold,
            )
            yield Split(seed, fold, np.sort(train), np.sort(val), np.sort(test))
