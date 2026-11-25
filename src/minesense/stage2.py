"""Load what Stage 2 produced: tuned configurations, fitted models and splits.

Stage 3 experiments reuse the hyperparameters that nested CV selected for each
outer split (chosen on inner folds only), so no outer test fold ever
influences a configuration.
"""
import json
from dataclasses import fields

import joblib
import numpy as np

from .cv import CV_DIR, MODEL_DIR
from .dataset import load_data
from .registry import MODELS
from .splits import Split, outer_splits
from .train import TrainConfig, load_model

BEST_DL = "pi_ft"  # highest mean inner-CV macro-F1 and outer accuracy among neural models


def splits() -> list[Split]:
    _, _, y = load_data()
    return list(outer_splits(y))


def split_name(sp: Split) -> str:
    return f"s{sp.seed}_f{sp.fold}"


def tuned_params(key: str) -> dict:
    """{split name: TrainConfig (neural) or parameter dict (baseline)}."""
    raw = json.loads((CV_DIR / key / "params.json").read_text())
    names = {f.name for f in fields(TrainConfig)}
    out = {}
    for split, entry in raw.items():
        p = entry["params"]
        out[split] = TrainConfig(**{k: v for k, v in p.items() if k in names}) \
            if MODELS[key].kind == "dl" else p
    return out


def fitted_model(key: str, sp: Split):
    """The model fitted in Stage 2 for this outer split."""
    if MODELS[key].kind == "dl":
        model, _ = load_model(MODEL_DIR / key / f"{split_name(sp)}.pt")
        return model
    return joblib.load(MODEL_DIR / key / f"{split_name(sp)}.joblib")


def proba(key: str, model, x: np.ndarray, soil: np.ndarray) -> np.ndarray:
    """Class probabilities from a Stage 2 model, whatever its kind."""
    from .baselines import baseline_features
    from .train import predict_proba

    if MODELS[key].kind == "dl":
        return predict_proba(model, x.astype(np.float32), soil)
    return model.predict_proba(baseline_features(x, soil, "onehot"))
