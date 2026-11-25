"""Fit the model that is deployed in the app and exported to ONNX.

    python -m minesense.final

Nested CV (Stage 2) estimated the performance of a *procedure*: tune PI-FT
with Optuna on 3 inner folds, then fit with early stopping on a stratified 15%
validation split. The deployed model is that same procedure applied once to
all 338 readings, so the nested-CV numbers are its performance estimate. No
new number from this script is a test score.

The uncertainty flag uses the predictive entropy of a single deterministic
pass. Stage 3 found that this ranks readings as well as 50-pass MC dropout
(equal area under the risk-coverage curve), and one pass suits an edge
device. As in Stage 3, the threshold is the 80th percentile of the entropy on
the validation rows.
"""
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import GLOBAL_SEED, RESULTS_DIR, VAL_FRACTION
from .cv import INNER_FOLDS
from .dataset import load_data
from .registry import MODELS, params_to_json, tune
from .train import fit, predict_proba, save_model

FINAL_DIR = RESULTS_DIR / "final"
FINAL_KEY = "pi_ft"
N_TRIALS = 40
FLAG_QUANTILE = 0.80


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def main() -> None:
    import optuna
    import torch

    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x, soil, y = load_data()
    rows = np.arange(len(y))
    skf = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=GLOBAL_SEED)
    inner = [(rows[a], rows[b]) for a, b in skf.split(rows, y)]
    start = time.perf_counter()
    params, inner_score, study = tune(MODELS[FINAL_KEY], x, soil, y, inner, N_TRIALS, GLOBAL_SEED)
    tune_seconds = time.perf_counter() - start
    train, val = train_test_split(rows, test_size=VAL_FRACTION, stratify=y, random_state=GLOBAL_SEED)
    train, val = np.sort(train), np.sort(val)
    model, history = fit(params, x, soil, y, train, val, seed=GLOBAL_SEED)
    p_val = predict_proba(model, x[val], soil[val])
    threshold = float(np.quantile(entropy(p_val), FLAG_QUANTILE))
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, params, FINAL_DIR / f"{FINAL_KEY}.pt")
    info = {"model": MODELS[FINAL_KEY].label, "params": params_to_json(params),
            "inner_macro_f1": inner_score, "trials": len(study.trials),
            "tune_seconds": tune_seconds, "n_train": len(train), "n_val": len(val),
            "best_epoch": history["best_epoch"],
            "fitted_offset_d0": float(model.features.offset),
            "flag_entropy_threshold": threshold, "flag_quantile": FLAG_QUANTILE}
    (FINAL_DIR / "final.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
