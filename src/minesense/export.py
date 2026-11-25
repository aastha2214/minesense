"""Edge export: ONNX, INT8 quantisation, size and latency; app assets.

    python -m minesense.export

Writes to app/assets/:
  minesense_fp32.onnx, minesense_int8.onnx   the deployed PI-FT model
  meta.json                                  labels, flag threshold, simulator
and results/tables/edge_export.json with parity, size and timing figures.

Timings are measured on the development laptop (Apple M5, one CPU thread),
not on an embedded device; they are reported as such.
"""
import json
import platform
import time

import numpy as np
import torch

from .config import CLASS_NAMES, N_CLASSES, N_SOILS, ROOT, SOIL_NAMES, TABLES_DIR
from .dataset import load_data
from .final import FINAL_DIR, FINAL_KEY, entropy
from .models import SATURATION, fit_offset
from .train import load_model

ASSETS = ROOT / "app" / "assets"
H_LEVELS = (np.arange(12) / 11).tolist()


class Deployable(torch.nn.Module):
    """Raw (V, H) and soil index in, class probabilities out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, soil):
        return torch.softmax(self.model(x, soil), dim=1)


def export_onnx(model, path) -> None:
    x = torch.tensor([[0.5, 0.5], [0.3, 0.1]], dtype=torch.float32)
    soil = torch.tensor([0, 3], dtype=torch.int64)
    batch = torch.export.Dim("batch", min=1, max=1_000_000)
    torch.onnx.export(Deployable(model).eval(), (x, soil), str(path), dynamo=True,
                      input_names=["x", "soil"], output_names=["probabilities"],
                      dynamic_shapes={"x": {0: batch}, "soil": {0: batch}}, opset_version=18,
                      external_data=False)
    # The exporter stores per-node debugging metadata (source locations, module
    # names); it does not affect inference and almost doubles the file size.
    import onnx

    proto = onnx.load(str(path))
    for node in proto.graph.node:
        del node.metadata_props[:]
        node.doc_string = ""
    del proto.graph.value_info[:]
    del proto.metadata_props[:]
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))


def fit_simulator(x, soil, y) -> dict:
    """V = a + b / (H + d0)^3 per (class, soil) series, with Gaussian residuals.

    Fitted on unsaturated readings; generated values are clipped to [0, 1], so
    strong near-field signals saturate as they do in the real data.
    """
    v, h = x[:, 0].astype(float), x[:, 1].astype(float)
    d0 = fit_offset(v, h, soil, y)
    series, resid = [], []
    for c in range(N_CLASSES):
        for s in range(N_SOILS):
            m = (y == c) & (soil == s) & (v < SATURATION)
            z = (h[m] + d0) ** -3.0
            b, a = np.polyfit(z, v[m], 1)
            r = v[m] - (a + b * z)
            resid.append(r)
            series.append({"class": c, "soil": s, "a": float(a), "b": float(b),
                           "sd": float(r.std(ddof=2)) if m.sum() > 3 else None,
                           "n_fit": int(m.sum())})
    pooled = float(np.concatenate(resid).std())
    for item in series:
        if item["sd"] is None:
            item["sd"] = pooled
    all_unsat = v < SATURATION
    pred = np.array([next(i for i in series if i["class"] == c and i["soil"] == s)
                     for c, s in zip(y, soil)])
    fitted = np.array([p["a"] + p["b"] * (hh + d0) ** -3 for p, hh in zip(pred, h)])
    ss_res = ((v - fitted)[all_unsat] ** 2).sum()
    ss_tot = ((v[all_unsat] - v[all_unsat].mean()) ** 2).sum()
    return {"d0": d0, "pooled_sd": pooled, "r2_unsaturated": float(1 - ss_res / ss_tot),
            "h_levels": H_LEVELS, "series": series}


def latency(fn, n: int) -> float:
    for _ in range(50):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def main() -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    torch.set_num_threads(1)
    ASSETS.mkdir(parents=True, exist_ok=True)
    x, soil, y = load_data()
    model, cfg = load_model(FINAL_DIR / f"{FINAL_KEY}.pt")
    info = json.loads((FINAL_DIR / "final.json").read_text())

    fp32, int8 = ASSETS / "minesense_fp32.onnx", ASSETS / "minesense_int8.onnx"
    export_onnx(model, fp32)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                for name, p in (("fp32", fp32), ("int8", int8))}

    # Parity on the 338 real readings and on 20,000 random in-range readings.
    rng = np.random.default_rng(0)
    xr = rng.uniform(0, 1, (20_000, 2)).astype(np.float32)
    sr = rng.integers(0, N_SOILS, 20_000)
    with torch.no_grad():
        ref_real = Deployable(model)(torch.as_tensor(x), torch.as_tensor(soil)).numpy()
        ref_rand = Deployable(model)(torch.as_tensor(xr), torch.as_tensor(sr)).numpy()
    parity = {}
    for name, sess in sessions.items():
        out_real = sess.run(None, {"x": x, "soil": soil.astype(np.int64)})[0]
        out_rand = sess.run(None, {"x": xr, "soil": sr.astype(np.int64)})[0]
        parity[name] = {
            "max_abs_prob_diff_real": float(np.abs(out_real - ref_real).max()),
            "max_abs_prob_diff_random": float(np.abs(out_rand - ref_rand).max()),
            "argmax_agreement_real": float((out_real.argmax(1) == ref_real.argmax(1)).mean()),
            "argmax_agreement_random": float((out_rand.argmax(1) == ref_rand.argmax(1)).mean()),
            "flag_agreement_random": float(
                ((entropy(out_rand) > info["flag_entropy_threshold"])
                 == (entropy(ref_rand) > info["flag_entropy_threshold"])).mean())}

    one_x, one_s = x[:1], soil[:1].astype(np.int64)
    big = rng.integers(0, len(y), 4096)
    big_x, big_s = x[big], soil[big].astype(np.int64)
    timing = {}
    with torch.no_grad():
        dm = Deployable(model).eval()
        tx, ts = torch.as_tensor(one_x), torch.as_tensor(one_s)
        timing["pytorch_eager"] = {"batch1_ms": 1e3 * latency(lambda: dm(tx, ts), 2000)}
        bx, bs = torch.as_tensor(big_x), torch.as_tensor(big_s)
        timing["pytorch_eager"]["readings_per_s_batch4096"] = 4096 / latency(lambda: dm(bx, bs), 50)
    for name, sess in sessions.items():
        timing[f"onnx_{name}"] = {
            "batch1_ms": 1e3 * latency(lambda: sess.run(None, {"x": one_x, "soil": one_s}), 2000),
            "readings_per_s_batch4096": 4096 / latency(
                lambda: sess.run(None, {"x": big_x, "soil": big_s}), 50)}

    report = {
        "model": info["model"],
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "size_kb": {"pytorch_state_dict": (FINAL_DIR / f"{FINAL_KEY}.pt").stat().st_size / 1024,
                    "onnx_fp32": fp32.stat().st_size / 1024,
                    "onnx_int8": int8.stat().st_size / 1024},
        "parity_vs_pytorch": parity,
        "timing_one_cpu_thread": timing,
        "machine": f"{platform.machine()} {platform.platform()}",
        "onnxruntime": ort.__version__, "torch": torch.__version__,
    }
    (TABLES_DIR / "edge_export.json").write_text(json.dumps(report, indent=1) + "\n")

    simulator = fit_simulator(x, soil, y)
    meta = {"class_names": CLASS_NAMES, "soil_names": SOIL_NAMES, "no_mine": 0,
            "flag_entropy_threshold": info["flag_entropy_threshold"],
            "flag_quantile": info["flag_quantile"], "model": info["model"],
            "fitted_offset_d0": info["fitted_offset_d0"], "simulator": simulator}
    (ASSETS / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    print("simulator R^2 on unsaturated readings:", round(simulator["r2_unsaturated"], 3),
          "d0:", round(simulator["d0"], 3))


if __name__ == "__main__":
    main()
