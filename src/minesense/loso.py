"""Leave-one-soil-out: can a model classify mines in a soil it has never seen?

    python -m minesense.loso

For each of the 6 soils: train on the other 5, test on the held-out one.
Hyperparameters are tuned with Optuna using an inner leave-one-soil-out loop
over the 5 training soils, so the search also rewards transfer to new soils.
The tuned model is then refitted with 5 seeds (85/15 stratified train/val
split of the 5 training soils for early stopping).

Soil encodings compared for the neural models:
  embed   6-way embedding; the unseen soil gets the mean of the trained vectors
  factor  moisture embedding + texture embedding (each seen in training)
  none    soil ignored
Baselines receive soil as factorised indicators (humid + 3 texture columns).
"""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import N_SOILS, RESULTS_DIR, SEEDS, SOIL_NAMES, VAL_FRACTION
from .dataset import load_data
from .metrics import PROB_COLS
from .registry import MODELS, ModelSpec, fit_predict, params_to_json, tune
from .train import TrainConfig

LOSO_DIR = RESULTS_DIR / "loso"


def _dl(key, label, space, **cfg):
    return ModelSpec(key, label, "dl", TrainConfig(**cfg), space)


LOSO_SPECS = {s.key: s for s in [
    _dl("mlp_none", "MLP, no soil input", "mlp", arch="mlp", soil_mode="none"),
    _dl("mlp_embed", "MLP, soil embedding (mean vector for unseen soil)", "mlp",
        arch="mlp", soil_mode="embed"),
    _dl("mlp_factor", "MLP, factorised soil embedding", "mlp", arch="mlp", soil_mode="factor"),
    _dl("pi_mlp_factor", "PI-MLP, factorised soil embedding", "mlp_pi",
        arch="mlp", physics=True, soil_mode="factor"),
    _dl("ft_factor", "FT-Transformer, factorised soil embedding", "ft",
        arch="ft", soil_mode="factor"),
    _dl("pi_ft_factor", "PI-FT-Transformer, factorised soil embedding", "ft_pi",
        arch="ft", physics=True, soil_mode="factor"),
    *(MODELS[k] for k in ("knn", "svm", "rf", "xgb")),
]}


def worker_init() -> None:
    import optuna
    import torch

    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)


def run_soil(key: str, held_out: int, n_trials: int):
    spec = LOSO_SPECS[key]
    x, soil, y = load_data()
    pool = np.flatnonzero(soil != held_out)
    test = np.flatnonzero(soil == held_out)
    inner = [(pool[soil[pool] != g], pool[soil[pool] == g]) for g in np.unique(soil[pool])]
    params, inner_score, _ = tune(spec, x, soil, y, inner, n_trials, seed=100 + held_out,
                                  soil_feat="factor")
    frames = []
    for rep in SEEDS:
        train, val = train_test_split(pool, test_size=VAL_FRACTION, stratify=y[pool],
                                      random_state=rep)
        proba, *_ = fit_predict(spec, params, x, soil, y, np.sort(train), np.sort(val), test,
                                seed=1000 * rep + held_out, soil_feat="factor")
        df = pd.DataFrame(proba, columns=PROB_COLS)
        df.insert(0, "y", y[test])
        df.insert(0, "row", test)
        df.insert(0, "rep", rep)
        df.insert(0, "held_out", SOIL_NAMES[held_out])
        frames.append(df)
    return pd.concat(frames), {"params": params_to_json(params), "inner_macro_f1": inner_score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(LOSO_SPECS))
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    LOSO_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.models.split(","):
        with ProcessPoolExecutor(args.workers, initializer=worker_init) as pool:
            futures = [pool.submit(run_soil, key, s, args.trials) for s in range(N_SOILS)]
            results = [f.result() for f in futures]
        preds = pd.concat([r[0] for r in results], ignore_index=True)
        preds.insert(0, "model", key)
        out = LOSO_DIR / key
        out.mkdir(exist_ok=True)
        preds.to_csv(out / "predictions.csv", index=False, float_format="%.6f")
        (out / "params.json").write_text(json.dumps(
            {SOIL_NAMES[s]: r[1] for s, r in enumerate(results)}, indent=1) + "\n")
        acc = (preds[PROB_COLS].to_numpy().argmax(1) == preds["y"]).mean()
        print(f"{LOSO_SPECS[key].label:50s} accuracy on unseen soils {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
