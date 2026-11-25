"""Model inference for the app: ONNX Runtime only, no PyTorch needed."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

ASSETS = Path(__file__).resolve().parent / "assets"
MODEL_FILES = {"FP32": "minesense_fp32.onnx", "INT8 (quantised)": "minesense_int8.onnx"}


@lru_cache(maxsize=1)
def meta() -> dict:
    return json.loads((ASSETS / "meta.json").read_text())


@lru_cache(maxsize=4)
def session(variant: str = "FP32") -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(str(ASSETS / MODEL_FILES[variant]), opts,
                                providers=["CPUExecutionProvider"])


def predict(v, h, soil, variant: str = "FP32") -> np.ndarray:
    """Class probabilities for normalised voltage v, height h and soil index."""
    v, h = np.atleast_1d(v), np.atleast_1d(h)
    x = np.column_stack([v, h]).astype(np.float32)
    s = np.broadcast_to(np.atleast_1d(soil), (len(x),)).astype(np.int64)
    return session(variant).run(None, {"x": x, "soil": s})[0]


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def flagged(p: np.ndarray) -> np.ndarray:
    """'Uncertain - inspect manually': entropy above the validation threshold."""
    return entropy(p) > meta()["flag_entropy_threshold"]
