"""Page functions of the MineSense app (imported by streamlit_app.py and the tests)."""

import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minefield as mf  # noqa: E402
from inference import MODEL_FILES, entropy, flagged, meta, predict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
M = meta()
CLASSES, SOILS = M["class_names"], M["soil_names"]
H_LEVELS = M["simulator"]["h_levels"]
RISK_SCALE = alt.Scale(domain=[0, 1], range=["#cce3fd", "#338ced", "#0e3d7e"])



def single_reading() -> None:
    st.title("Classify one detector reading")
    st.caption(f"Model: {M['model']}, exported to ONNX. The dataset publishes normalised "
               "readings only (voltage scaled to at most 1, height in 12 levels from 0 to 1), "
               "so inputs here are on that scale.")
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        v = st.slider("Voltage V (normalised)", 0.0, 1.0, 0.35, 0.001, key="v")
        h = st.select_slider("Sensor height H (normalised)", options=H_LEVELS, value=H_LEVELS[3],
                             format_func=lambda x: f"{x:.3f}", key="h")
        soil = st.selectbox("Soil type", range(len(SOILS)), format_func=lambda i: SOILS[i],
                            key="soil")
        variant = st.radio("Model file", list(MODEL_FILES), horizontal=True, key="variant1")
    p = predict(v, h, soil, variant)[0]
    k = int(p.argmax())
    with right:
        is_flagged = bool(flagged(p[None])[0])
        if is_flagged:
            st.warning(f"**Uncertain: inspect manually.** Best guess: {CLASSES[k]} "
                       f"({p[k]:.0%}).", icon="⚠️")
        elif k == M["no_mine"]:
            st.success(f"**{CLASSES[k]}** ({p[k]:.0%})", icon="✅")
        else:
            st.error(f"**Mine: {CLASSES[k]}** ({p[k]:.0%})", icon="🚩")
        df = pd.DataFrame({"class": CLASSES, "probability": p})
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=22).encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
            y=alt.Y("class:N", sort=CLASSES, title=None),
            color=alt.condition(alt.datum["class"] == CLASSES[k], alt.value("#257de0"),
                                alt.value("#b8c4d0")),
            tooltip=["class", alt.Tooltip("probability:Q", format=".3f")])
        st.altair_chart(bars.properties(height=190), use_container_width=True)
        st.caption(f"Predictive entropy {entropy(p[None])[0]:.3f}; readings above "
                   f"{M['flag_entropy_threshold']:.3f} (the {M['flag_quantile']:.0%} quantile on "
                   "validation data) are flagged for manual inspection.")


def field_frame(fld: mf.Field, reveal: bool) -> pd.DataFrame:
    rows, cols = fld.truth.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    p = fld.probs.reshape(-1, 5)
    scanned = fld.scanned.ravel()
    risk = np.where(scanned, 1 - np.nan_to_num(p[:, M["no_mine"]]), np.nan)
    flag = np.zeros(len(p), bool)
    flag[scanned] = flagged(p[scanned])
    pred = np.where(scanned, np.nan_to_num(p).argmax(1), -1)
    truth = fld.truth.ravel()
    return pd.DataFrame({
        "row": rr.ravel(), "col": cc.ravel(), "risk": risk, "scanned": scanned, "flag": flag,
        "soil": [SOILS[s] for s in fld.soil.ravel()],
        "prediction": [CLASSES[k] if k >= 0 else "not scanned" for k in pred],
        "truth": [CLASSES[t] if reveal else "hidden" for t in truth],
        "mine": (truth != M["no_mine"]) & reveal,
        "alarm": scanned & (pred != M["no_mine"])})


def field_chart(df: pd.DataFrame, cell: int) -> alt.LayerChart:
    base = alt.Chart(df).encode(
        x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None),
        tooltip=["row", "col", "soil", "prediction", alt.Tooltip("risk:Q", format=".2f"),
                 "flag", "truth"])
    unscanned = base.transform_filter(~alt.datum.scanned).mark_rect(
        color="#eef1f4", stroke="white", strokeWidth=1.5)
    heat = base.transform_filter(alt.datum.scanned).mark_rect(stroke="white", strokeWidth=1.5).encode(
        color=alt.Color("risk:Q", scale=RISK_SCALE, title="P(mine)",
                        legend=alt.Legend(orient="bottom", gradientLength=200)))
    query = base.transform_filter(alt.datum.flag).mark_text(
        text="?", fontSize=cell * 0.55, fontWeight="bold", color="#f2a600")
    mines = base.transform_filter(alt.datum.mine).mark_point(
        shape="circle", size=cell * cell * 0.45, strokeWidth=2.5, color="#191919", filled=False)
    return (unscanned + heat + query + mines).properties(
        width=cell * df["col"].nunique(), height=cell * df["row"].nunique())


def soil_chart(df: pd.DataFrame, cell: int) -> alt.Chart:
    return alt.Chart(df).mark_rect(stroke="white", strokeWidth=1.5).encode(
        x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None),
        color=alt.Color("soil:N", scale=alt.Scale(domain=SOILS, range=[
            "#f6d9a8", "#b08b5b", "#e8e3d3", "#c9a66b", "#6f5232", "#bdb6a0"]),
            legend=alt.Legend(orient="bottom", columns=3, title="Soil")),
        tooltip=["soil"]).properties(width=cell * 0.5 * df["col"].nunique(),
                                     height=cell * 0.5 * df["row"].nunique())


def show_score(fld: mf.Field, slot) -> None:
    s = mf.score(fld, flagged)
    with slot.container():
        a, b, c, d = st.columns(4)
        a.metric("Mines found", f"{s['found']} / {s['mines']}")
        b.metric("Mines missed", s["missed"],
                 help="Real mines classified as 'no mine'. Of these, the number flagged "
                      "for manual inspection is shown below.")
        c.metric("False alarms", s["false_alarms"])
        d.metric("Uncertain cells", s["flagged"])
        st.caption(f"{s['scanned']} cells scanned. Missed mines that were flagged for "
                   f"inspection anyway: {s['missed_but_flagged']}. Mine type correct for "
                   f"{s['type_correct']} of {s['mines']} mines.")


def minefield() -> None:
    st.title("Minefield sweep simulation")
    st.caption("Hidden mines in mixed soil; the detector sweeps cell by cell and the map shows "
               "the model's mine probability live. Readings are synthetic, generated from the "
               "inverse-cube height law fitted to the dataset, so this is a demonstration, not "
               "an accuracy test.")
    with st.sidebar:
        st.header("Field")
        rows = st.slider("Rows", 6, 20, 12)
        cols = st.slider("Columns", 6, 24, 16)
        density = st.slider("Mine density", 0.05, 0.4, 0.15, 0.01)
        seed = st.number_input("Random seed", 0, 10_000, 7)
        st.header("Detector")
        h = st.select_slider("Sweep height (normalised)", options=H_LEVELS, value=H_LEVELS[5],
                             format_func=lambda x: f"{x:.2f}")
        noise = st.slider("Sensor noise (× fitted noise)", 0.0, 5.0, 2.0, 0.25)
        variant = st.radio("Model file", list(MODEL_FILES), key="variant2")
        speed = st.slider("Sweep speed (rows per second)", 1, 20, 6)
    config = (rows, cols, density, seed, h, noise)
    if st.session_state.get("config") != config:
        st.session_state.config = config
        st.session_state.field = mf.make_field(rows, cols, density, h, noise, int(seed))
    fld: mf.Field = st.session_state.field

    b1, b2, b3, _ = st.columns([1, 1, 1.3, 3])
    run = b1.button("Sweep", type="primary", use_container_width=True)
    if b2.button("Reset", use_container_width=True):
        st.session_state.field = fld = mf.make_field(rows, cols, density, h, noise, int(seed))
    reveal = b3.toggle("Reveal mines", value=False)

    cell = max(18, min(34, 560 // max(rows, cols)))
    score_slot = st.empty()
    left, right = st.columns([3, 1.2], gap="large")
    map_slot = left.empty()
    right.altair_chart(soil_chart(field_frame(fld, False), cell), use_container_width=False)
    right.caption("Soil map (known to the operator). **?** marks cells flagged *uncertain - "
                  "inspect manually*; circles show true mines when revealed.")

    def predict_fn(v, hh, s):
        return predict(v, hh, s, variant)

    if run:
        path = mf.sweep_path(rows, cols)
        for r in range(rows):
            todo = [cell_rc for cell_rc in path[r * cols:(r + 1) * cols] if not fld.scanned[cell_rc]]
            if todo:
                mf.scan(fld, todo, M["simulator"], predict_fn)
            map_slot.altair_chart(field_chart(field_frame(fld, reveal), cell))
            show_score(fld, score_slot)
            time.sleep(1 / speed)
    map_slot.altair_chart(field_chart(field_frame(fld, reveal), cell))
    if fld.scanned.any():
        show_score(fld, score_slot)


def results() -> None:
    st.title("Cross-validated results")
    st.caption("Nested cross-validation: stratified 5-fold × 5 seeds, hyperparameters tuned "
               "with Optuna inside every training fold. Values are means over the 25 test folds.")
    path = TABLES / "stage3_summary.csv"
    if path.exists():
        t = pd.read_csv(path).sort_values("accuracy", ascending=False)
        show = pd.DataFrame({
            "Model": t["model"],
            "Accuracy": [f"{m:.3f} ± {s:.3f}" for m, s in zip(t["accuracy"], t["accuracy_std"])],
            "Macro F1": t["macro_f1"].map("{:.3f}".format),
            "Detection (mine vs none)": t["detection"].map("{:.3f}".format),
            "Mines missed": t["mine_missed"].map("{:.1%}".format),
            "False alarms": t["false_alarm"].map("{:.1%}".format)})
        st.dataframe(show, hide_index=True, use_container_width=True)
    for name, caption in (("fig14_noise_robustness", "Robustness to sensor noise"),
                          ("fig13_risk_coverage", "Accuracy when uncertain readings are referred"),
                          ("fig09_leave_one_soil_out", "Accuracy on a soil never seen in training")):
        f = FIGURES / f"{name}.png"
        if f.exists():
            st.image(str(f), caption=caption)

