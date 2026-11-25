"""Smoke tests for the Streamlit app and the minefield engine (need app/assets)."""
import sys
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parent.parent / "app"
pytestmark = pytest.mark.skipif(not (APP / "assets" / "meta.json").exists(),
                                reason="run `python -m minesense.export` first")
sys.path.insert(0, str(APP))


def test_onnx_predictions_are_probabilities():
    from inference import predict

    p = predict(np.linspace(0, 1, 50), np.full(50, 0.5), np.arange(50) % 6)
    assert p.shape == (50, 5)
    assert np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()


def test_int8_model_mostly_agrees_with_fp32():
    from inference import predict

    rng = np.random.default_rng(1)
    v, h, s = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500), rng.integers(0, 6, 500)
    a = predict(v, h, s, "FP32").argmax(1)
    b = predict(v, h, s, "INT8 (quantised)").argmax(1)
    assert (a == b).mean() > 0.95


def test_minefield_sweep_scores_add_up():
    import minefield as mf
    from inference import flagged, meta, predict

    m = meta()
    fld = mf.make_field(8, 9, 0.2, m["simulator"]["h_levels"][2], 1.0, seed=3)
    mf.scan(fld, mf.sweep_path(8, 9), m["simulator"], lambda v, h, s: predict(v, h, s))
    s = mf.score(fld, flagged)
    assert s["scanned"] == 72
    assert s["found"] + s["missed"] == s["mines"] == int((fld.truth > 0).sum())
    assert s["missed_but_flagged"] <= s["missed"]


def page(name: str):
    from streamlit.testing.v1 import AppTest

    script = f"import sys\nsys.path.insert(0, {str(APP)!r})\nimport views\nviews.{name}()\n"
    return AppTest.from_string(script, default_timeout=120)


def test_entry_point_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception


def test_single_reading_page():
    at = page("single_reading").run()
    assert not at.exception
    for v in (0.2, 0.35, 0.9):
        at.slider(key="v").set_value(v).run()
        assert not at.exception
        assert at.error or at.warning or at.success  # a verdict is shown


def test_minefield_page_sweeps_and_scores():
    at = page("minefield").run()
    assert not at.exception
    at.sidebar.slider[0].set_value(6)   # rows
    at.sidebar.slider[1].set_value(6)   # columns
    at.sidebar.slider[-1].set_value(20)  # fastest sweep speed
    at.run()
    at.button[0].click().run()          # "Sweep"
    assert not at.exception
    assert any("Mines found" in m.label for m in at.metric)
    at.toggle[0].set_value(True).run()  # reveal mines
    assert not at.exception


def test_results_page():
    at = page("results").run()
    assert not at.exception
    assert at.dataframe
