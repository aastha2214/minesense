"""Every compared model, its search space, and a uniform fit / predict / tune API."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import optuna
from sklearn.metrics import f1_score

from .baselines import baseline_features, make_baseline, suggest_params
from .train import TrainConfig, fit, predict_proba


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str                      # "dl" or "baseline"
    base: TrainConfig | None = None
    space: str = ""                # search space for neural models


DL_SPECS = [
    ModelSpec("mlp", "MLP", "dl", TrainConfig(arch="mlp"), "mlp"),
    ModelSpec("mlp_phys", "MLP + physics features", "dl",
              TrainConfig(arch="mlp", physics=True), "mlp"),
    ModelSpec("pi_mlp", "PI-MLP (physics features + loss)", "dl",
              TrainConfig(arch="mlp", physics=True), "mlp_pi"),
    ModelSpec("ft", "FT-Transformer", "dl", TrainConfig(arch="ft"), "ft"),
    ModelSpec("ft_phys", "FT-Transformer + physics features", "dl",
              TrainConfig(arch="ft", physics=True), "ft"),
    ModelSpec("pi_ft", "PI-FT-Transformer (physics features + loss)", "dl",
              TrainConfig(arch="ft", physics=True), "ft_pi"),
]
BASELINE_SPECS = [
    ModelSpec("knn", "k-NN", "baseline"),
    ModelSpec("svm", "SVM (RBF)", "baseline"),
    ModelSpec("rf", "Random forest", "baseline"),
    ModelSpec("xgb", "XGBoost", "baseline"),
]
MODELS = {s.key: s for s in DL_SPECS + BASELINE_SPECS}


def suggest_config(spec: ModelSpec, trial, base: TrainConfig | None = None) -> TrainConfig:
    """Sample a training configuration for a neural model."""
    base = base or spec.base
    p = {}
    if spec.space.startswith("mlp"):
        p["hidden"] = trial.suggest_categorical("hidden", [32, 64, 128])
        p["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        p["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
        p["emb_dim"] = trial.suggest_int("emb_dim", 2, 6)
        p["lr"] = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    else:
        p["d_token"] = trial.suggest_categorical("d_token", [16, 32, 64])
        p["n_blocks"] = trial.suggest_int("n_blocks", 1, 3)
        p["attn_dropout"] = trial.suggest_float("attn_dropout", 0.0, 0.3)
        p["ffn_dropout"] = trial.suggest_float("ffn_dropout", 0.0, 0.3)
        p["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    p["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    p["noise"] = trial.suggest_float("noise", 0.02, 0.3, log=True)
    if spec.space.endswith("_pi"):
        p["phys_weight"] = trial.suggest_float("phys_weight", 0.05, 2.0, log=True)
    return replace(base, **p)


def fit_predict(spec: ModelSpec, params, x, soil, y, train, val, test, seed,
                soil_feat="onehot", probability=True):
    """Fit on train (val for early stopping) and return (test probabilities, model, history).

    ``params`` is a TrainConfig for neural models and a dict for baselines.
    Baselines have no early stopping, so they are fitted on train and val together.
    """
    if spec.kind == "dl":
        model, history = fit(params, x, soil, y, train, val, seed)
        model.soil.fill_unseen(sorted(set(soil[train].tolist())))
        return predict_proba(model, x[test], soil[test]), model, history
    feats = baseline_features(x, soil, soil_feat)
    fit_rows = np.concatenate([train, val])
    est = make_baseline(spec.key, params, seed, probability).fit(feats[fit_rows], y[fit_rows])
    if probability:
        return est.predict_proba(feats[test]), est, None
    return np.eye(5)[est.predict(feats[test])], est, None  # one-hot of the label


def tune(spec: ModelSpec, x, soil, y, inner: list[tuple[np.ndarray, np.ndarray]],
         n_trials: int, seed: int, soil_feat="onehot", base: TrainConfig | None = None):
    """Optuna TPE search maximising mean macro-F1 over the inner folds.

    Neural models use each inner validation fold both for early stopping and
    for scoring; this makes inner scores slightly optimistic but affects all
    configurations alike, and the outer test fold is never touched.
    Returns (best params, best inner score, study).
    """
    def objective(trial):
        params = (suggest_config(spec, trial, base) if spec.kind == "dl"
                  else suggest_params(spec.key, trial))
        scores = []
        for k, (tr, va) in enumerate(inner):
            stop_rows = va if spec.kind == "dl" else va[:0]  # baselines never see va
            proba, *_ = fit_predict(spec, params, x, soil, y, tr, stop_rows, va, seed,
                                    soil_feat, probability=False)
            scores.append(f1_score(y[va], proba.argmax(1), average="macro"))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials)
    fixed = optuna.trial.FixedTrial(study.best_params)
    best = (suggest_config(spec, fixed, base) if spec.kind == "dl"
            else suggest_params(spec.key, fixed))
    return best, float(study.best_value), study


def params_to_json(params) -> dict:
    return asdict(params) if isinstance(params, TrainConfig) else dict(params)
