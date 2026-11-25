"""Classical machine learning baselines (for comparison only)."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import N_SOILS
from .models import soil_factors

BASELINES = ["knn", "svm", "rf", "xgb"]


def baseline_features(x: np.ndarray, soil: np.ndarray, soil_mode: str = "onehot") -> np.ndarray:
    """[V, H] plus soil as one-hot (6 columns) or factorised (humid + 3 textures)."""
    if soil_mode == "onehot":
        enc = np.eye(N_SOILS)[soil]
    elif soil_mode == "factor":
        moisture, texture = soil_factors(soil)
        enc = np.column_stack([moisture, np.eye(3)[texture]])
    else:
        raise ValueError(soil_mode)
    return np.column_stack([x, enc]).astype(np.float64)


def suggest_params(name: str, trial) -> dict:
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_categorical("p", [1, 2])}
    if name == "svm":
        # Widened after a first run put the tuned C at the old upper limit (1e3).
        return {"C": trial.suggest_float("C", 1e-2, 1e5, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e2, log=True)}
    if name == "rf":
        return {"max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", [0.3, 0.6, 1.0])}
    if name == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 600),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    raise ValueError(name)


def make_baseline(name: str, params: dict, seed: int, probability: bool = True):
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**params))
    if name == "svm":
        svc = SVC(random_state=seed, **params)
        if probability:  # Platt scaling, fitted on internal cross-validation folds
            svc = CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False)
        return make_pipeline(StandardScaler(), svc)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1, **params)
    if name == "xgb":
        return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=1,
                             eval_metric="mlogloss", **params)
    raise ValueError(name)
