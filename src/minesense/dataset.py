"""Loading and schema validation."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES, KAGGLE_CSV, N_CLASSES, N_SOILS, TARGET


def load_raw(path: Path = KAGGLE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m minesense.download")
    # utf-8-sig strips the byte-order mark at the start of the file.
    return pd.read_csv(path, encoding="utf-8-sig")


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the table does not match the expected schema."""
    if list(df.columns) != FEATURES + [TARGET]:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("missing or non-finite values present")
    if not df["S"].isin(range(1, N_SOILS + 1)).all():
        raise ValueError("soil code outside 1..6")
    if not df[TARGET].isin(range(1, N_CLASSES + 1)).all():
        raise ValueError("mine code outside 1..5")
    for col in ("V", "H"):
        if not df[col].between(0.0, 1.0).all():
            raise ValueError(f"{col} is outside the normalised range [0, 1]")


def load_data(path: Path = KAGGLE_CSV) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (continuous [V, H] as float32, soil index 0..5, label 0..4)."""
    df = load_raw(path)
    validate(df)
    x_cont = df[["V", "H"]].to_numpy(dtype=np.float32)
    soil = df["S"].to_numpy(dtype=np.int64) - 1
    y = df[TARGET].to_numpy(dtype=np.int64) - 1
    return x_cont, soil, y
