"""Nested cross-validation: Optuna tuning inside every outer split.

    python -m minesense.cv                       # all models
    python -m minesense.cv --models mlp,svm      # a subset

Outer loop: stratified 5-fold CV repeated over 5 seeds (25 splits, from
splits.py). Inner loop: stratified 3-fold CV on the outer training rows
(train + val) for hyperparameter search. The outer test fold is used once.

Writes, per model, results/cv/<model>/{predictions.csv, params.json,
histories.json} and the fitted models to results/models/<model>/.
"""
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import RESULTS_DIR
from .dataset import load_data
from .metrics import PROB_COLS, mean_std, per_split
from .registry import MODELS, fit_predict, params_to_json, tune
from .splits import outer_splits
from .train import save_model

CV_DIR = RESULTS_DIR / "cv"
MODEL_DIR = RESULTS_DIR / "models"
INNER_FOLDS = 3


def worker_init() -> None:
    import optuna
    import torch

    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)


def run_split(key, seed, fold, train, val, test, n_trials):
    spec = MODELS[key]
    x, soil, y = load_data()
    dev = np.sort(np.concatenate([train, val]))
    run_seed = 1000 * seed + fold
    skf = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=run_seed)
    inner = [(dev[a], dev[b]) for a, b in skf.split(dev, y[dev])]

    start = time.perf_counter()
    params, inner_score, study = tune(spec, x, soil, y, inner, n_trials, run_seed)
    tune_seconds = time.perf_counter() - start
    proba, model, history = fit_predict(spec, params, x, soil, y, train, val, test, run_seed)

    name = f"s{seed}_f{fold}"
    if spec.kind == "dl":
        save_model(model, params, MODEL_DIR / key / f"{name}.pt")
    else:
        (MODEL_DIR / key).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / key / f"{name}.joblib")
    preds = pd.DataFrame(proba, columns=PROB_COLS)
    preds.insert(0, "y", y[test])
    preds.insert(0, "row", test)
    preds.insert(0, "fold", fold)
    preds.insert(0, "seed", seed)
    pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    return {"split": name, "predictions": preds, "history": history,
            "params": {"params": params_to_json(params), "inner_macro_f1": inner_score,
                       "trials": len(study.trials), "pruned": pruned,
                       "tune_seconds": tune_seconds}}


def run_model(key: str, n_trials: int, workers: int) -> pd.DataFrame:
    _, _, y = load_data()
    splits = list(outer_splits(y))
    with ProcessPoolExecutor(workers, initializer=worker_init) as pool:
        futures = [pool.submit(run_split, key, sp.seed, sp.fold, sp.train, sp.val, sp.test,
                               n_trials) for sp in splits]
        results = [f.result() for f in futures]
    out = CV_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.concat([r["predictions"] for r in results], ignore_index=True)
    preds.to_csv(out / "predictions.csv", index=False, float_format="%.6f")
    (out / "params.json").write_text(json.dumps({r["split"]: r["params"] for r in results},
                                                indent=1) + "\n")
    if MODELS[key].kind == "dl":
        (out / "histories.json").write_text(json.dumps({r["split"]: r["history"]
                                                        for r in results}) + "\n")
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--trials-dl", type=int, default=40)
    parser.add_argument("--trials-baseline", type=int, default=60)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    for key in args.models.split(","):
        start = time.perf_counter()
        trials = args.trials_dl if MODELS[key].kind == "dl" else args.trials_baseline
        preds = run_model(key, trials, args.workers)
        summary = mean_std(per_split(preds))
        print(f"{MODELS[key].label:45s} " + "  ".join(
            f"{m} {mu:.3f}±{sd:.3f}" for m, (mu, sd) in summary.items())
              + f"  [{time.perf_counter() - start:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
