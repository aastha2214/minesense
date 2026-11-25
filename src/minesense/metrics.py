"""Evaluation metrics, including the safety-critical mine-missed rate."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import N_CLASSES, NO_MINE

PROB_COLS = [f"p{k}" for k in range(N_CLASSES)]


def mine_missed_rate(y, pred) -> float:
    """Share of real mines (any type) that were predicted as 'no mine'."""
    mine = y != NO_MINE
    return float(np.mean(pred[mine] == NO_MINE))


def false_alarm_rate(y, pred) -> float:
    """Share of mine-free readings that were predicted as some mine."""
    empty = y == NO_MINE
    return float(np.mean(pred[empty] != NO_MINE))


def basic_metrics(y, pred) -> dict:
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=range(N_CLASSES),
                                       zero_division=0)),
            "mine_missed": mine_missed_rate(y, pred),
            "false_alarm": false_alarm_rate(y, pred)}


def per_split(pred_df: pd.DataFrame, keys=("seed", "fold")) -> pd.DataFrame:
    rows = []
    for key, g in pred_df.groupby(list(keys)):
        pred = g[PROB_COLS].to_numpy().argmax(1)
        rows.append(dict(zip(keys, key)) | basic_metrics(g["y"].to_numpy(), pred))
    return pd.DataFrame(rows)


def mean_std(table: pd.DataFrame, cols=("accuracy", "macro_f1", "mine_missed", "false_alarm")):
    return {c: (float(table[c].mean()), float(table[c].std(ddof=1))) for c in cols}
