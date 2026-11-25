"""Data quality and leakage audit.

    python -m minesense.audit

Writes results/tables/data_audit.json and data/processed/splits.csv.
"""
import json

import numpy as np
import pandas as pd

from .config import (CLASS_NAMES, FEATURES, KAGGLE_CSV, N_FOLDS, PROCESSED_DIR, SEEDS,
                     TABLES_DIR, TARGET, UCI_XLS)
from .dataset import load_raw, validate
from .download import sha256
from .splits import outer_splits


def quality_checks(df: pd.DataFrame) -> dict:
    counts_per_feature = df.groupby(FEATURES)[TARGET].nunique()
    h_values = np.sort(df["H"].unique())
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df.duplicated(FEATURES).sum()),
        "feature_vectors_with_conflicting_labels": int((counts_per_feature > 1).sum()),
        "duplicate_rows_after_rounding_V_to_4dp": int(
            df.assign(V=df["V"].round(4)).duplicated().sum()
        ),
        "class_counts": {CLASS_NAMES[m - 1]: int(n)
                         for m, n in df[TARGET].value_counts().sort_index().items()},
        "soil_counts": {int(s): int(n) for s, n in df["S"].value_counts().sort_index().items()},
        "V_min": float(df["V"].min()),
        "V_max": float(df["V"].max()),
        "V_unique_values": int(df["V"].nunique()),
        "rows_sharing_V_with_another_row": int(df["V"].duplicated(keep=False).sum()),
        "H_unique_values": [float(h) for h in h_values],
        "H_values_times_11": [float(round(h * 11, 6)) for h in h_values],
    }


def compare_with_uci(df: pd.DataFrame) -> dict:
    """Row-by-row comparison of the Kaggle copy with the original UCI spreadsheet."""
    if not UCI_XLS.exists():
        return {"checked": False, "reason": "run: python -m minesense.download --with-uci"}
    uci = pd.read_excel(UCI_XLS, sheet_name="Normalized_Data", engine="calamine")
    # The original stores soil as 0, 0.2, ..., 1.0; the Kaggle copy as 1..6.
    uci = uci.assign(S=(uci["S"] * 5).round().astype(int) + 1)
    same_shape = uci.shape == df.shape
    result = {"checked": True, "uci_rows": int(len(uci)), "same_shape": bool(same_shape)}
    if same_shape:
        diff = np.abs(uci[df.columns].to_numpy(float) - df.to_numpy(float))
        result["max_abs_difference_same_row_order"] = float(diff.max())
        result["identical_within_1e-8"] = bool(diff.max() < 1e-8)
    return result


def design_structure(df: pd.DataFrame) -> dict:
    """Describe the soil x height x class grid behind the measurements."""
    cells = df.groupby(["S", "H", TARGET]).size()
    full = df["S"].nunique() * df["H"].nunique() * df[TARGET].nunique()
    return {
        "soil_levels": int(df["S"].nunique()),
        "height_levels": int(df["H"].nunique()),
        "class_levels": int(df[TARGET].nunique()),
        "cells_in_full_grid": int(full),
        "cells_observed": int(len(cells)),
        "cells_missing": int(full - len(cells)),
        "max_rows_in_one_cell": int(cells.max()),
    }


def split_checks(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = df[TARGET].to_numpy() - 1
    keys = df[FEATURES].apply(tuple, axis=1).to_numpy()
    rows, overlap_index, overlap_features = [], 0, 0
    tested = {seed: np.zeros(len(df), int) for seed in SEEDS}
    test_sizes, min_test_per_class = [], []
    for sp in outer_splits(y):
        parts = {"train": sp.train, "val": sp.val, "test": sp.test}
        names = list(parts)
        for a in range(3):
            for b in range(a + 1, 3):
                ia, ib = parts[names[a]], parts[names[b]]
                overlap_index += len(np.intersect1d(ia, ib))
                overlap_features += len(set(keys[ia]) & set(keys[ib]))
        tested[sp.seed][sp.test] += 1
        test_sizes.append(len(sp.test))
        min_test_per_class.append(int(np.bincount(y[sp.test], minlength=5).min()))
        for name, idx in parts.items():
            rows += [(sp.seed, sp.fold, name, int(i)) for i in idx]
    summary = {
        "seeds": SEEDS,
        "folds_per_seed": N_FOLDS,
        "total_splits": len(test_sizes),
        "shared_row_indices_between_parts": int(overlap_index),
        "shared_feature_vectors_between_parts": int(overlap_features),
        "each_row_tested_exactly_once_per_seed": bool(all((t == 1).all() for t in tested.values())),
        "test_fold_size_range": [min(test_sizes), max(test_sizes)],
        "train_size_first_split": int(sum(1 for r in rows if r[:3] == (SEEDS[0], 0, "train"))),
        "val_size_first_split": int(sum(1 for r in rows if r[:3] == (SEEDS[0], 0, "val"))),
        "smallest_class_count_in_any_test_fold": min(min_test_per_class),
    }
    return summary, pd.DataFrame(rows, columns=["seed", "fold", "part", "row"])


def main() -> None:
    df = load_raw()
    validate(df)
    split_summary, split_table = split_checks(df)
    report = {
        "source_csv_sha256": sha256(KAGGLE_CSV),
        "quality": quality_checks(df),
        "kaggle_vs_uci": compare_with_uci(df),
        "design": design_structure(df),
        "splits": split_summary,
    }
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    split_table.to_csv(PROCESSED_DIR / "splits.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
