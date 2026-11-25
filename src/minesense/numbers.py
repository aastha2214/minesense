"""Single source of every number quoted in the README and the report.

    python -m minesense.numbers        # writes report/generated/* and README.md

All values are read from results/ (tables written by the experiment scripts).
The report uses the generated LaTeX macros and tables; the README is rendered
from docs/README.template.md by replacing {{name}} placeholders. The checker
(minesense.check_docs) verifies that both are up to date and that no result
number was typed by hand.
"""
import json
import re

import pandas as pd

from .config import ROOT, TABLES_DIR
from .metrics import per_split

GEN = ROOT / "report" / "generated"
RESULTS = ROOT / "results"
SHORT = {"mlp": "MLP", "mlp_phys": "MLP + phys. features", "pi_mlp": "PI-MLP",
         "ft": "FT-Transformer", "ft_phys": "FT-Transformer + phys. features",
         "pi_ft": "PI-FT-Transformer", "knn": "k-NN", "svm": "SVM (RBF)",
         "rf": "Random forest", "xgb": "XGBoost"}
LABEL_TO_KEY = {
    "MLP": "mlp", "MLP + physics features": "mlp_phys", "PI-MLP (physics features + loss)": "pi_mlp",
    "FT-Transformer": "ft", "FT-Transformer + physics features": "ft_phys",
    "PI-FT-Transformer (physics features + loss)": "pi_ft", "k-NN": "knn", "SVM (RBF)": "svm",
    "Random forest": "rf", "XGBoost": "xgb"}
DL = ["pi_ft", "ft", "ft_phys", "pi_mlp", "mlp_phys", "mlp"]
BASE = ["svm", "rf", "xgb", "knn"]


def t(name):
    return pd.read_csv(TABLES_DIR / name)


def j(name):
    return json.loads((TABLES_DIR / name).read_text())


class Fmt:
    """Formatting for one target ('md' or 'tex')."""

    def __init__(self, target):
        self.tex = target == "tex"

    def minus(self, text):
        return "$-$" + text[1:] if self.tex and text.startswith("-") else text

    def f(self, x, d=3):
        return self.minus(f"{x:.{d}f}")

    def pct(self, x, d=1):
        return self.minus(f"{100 * x:.{d}f}")

    def p(self, x):
        if x < 0.001:
            return r"$<$\,0.001" if self.tex else "< 0.001"
        return f"{x:.2g}" if x < 0.01 else f"{x:.2f}" if x >= 0.1 else f"{x:.3f}"

    def peq(self, x):
        """p-value with its relation, for running text: '= 0.66' or '< 0.001'."""
        out = self.p(x)
        if out.startswith("$<$") or out.startswith("<"):
            return out
        return f"$=$ {out}" if self.tex else f"= {out}"

    def pm(self, m, s, d=3):
        return f"{m:.{d}f} $\\pm$ {s:.{d}f}" if self.tex else f"{m:.{d}f} ± {s:.{d}f}"


def build(target: str) -> dict:
    F = Fmt(target)
    n = {}
    audit, eda = j("data_audit.json"), j("eda_summary.json")
    q = audit["quality"]
    n["rows"] = str(q["rows"])
    for name, cnt in q["class_counts"].items():
        n[f"count_{name}"] = str(cnt)
    n["grid_observed"] = str(audit["design"]["cells_observed"])
    n["grid_cells"] = str(audit["design"]["cells_in_full_grid"])
    n["grid_missing"] = str(audit["design"]["cells_missing"])
    n["saturated_rows"] = str(eda["height_law"]["saturated_rows"])
    n["kaggle_uci_diff"] = f"{audit['kaggle_vs_uci']['max_abs_difference_same_row_order']:.0e}"
    n["test_fold_min"], n["test_fold_max"] = map(str, audit["splits"]["test_fold_size_range"])
    n["train_first"] = str(audit["splits"]["train_size_first_split"])
    n["val_first"] = str(audit["splits"]["val_size_first_split"])
    hl = eda["height_law"]
    n["series_negative_rho"] = str(hl["series_with_negative_rho"])
    n["series_rho_below"] = str(hl["series_with_rho_below_minus_0.8"])
    fu = hl["fits"]["unsaturated_rows"]
    n["eda_r2_linear"] = F.f(fu["R2_linear_in_H"])
    n["eda_r2_cube"] = F.f(fu["R2_inverse_cube_shared_offset"])
    n["eda_d0"] = F.f(fu["best_shared_offset_d0"], 2)
    fa = hl["fits"]["all_rows"]
    n["eda_all_r2_linear"] = F.f(fa["R2_linear_in_H"])
    n["eda_all_r2_cube"] = F.f(fa["R2_inverse_cube_shared_offset"])
    ve = eda["V_variance_explained_R2"]
    n["r2_class"], n["r2_class_height"], n["r2_all"] = (F.f(ve["M"], 2), F.f(ve["M+H"], 2),
                                                        F.f(ve["M+H+S"], 2))

    s = t("stage3_summary.csv").set_index("key")
    for k in s.index:
        r = s.loc[k]
        n[f"acc_{k}"] = F.f(r["accuracy"])
        n[f"accsd_{k}"] = F.f(r["accuracy_std"])
        n[f"accpct_{k}"] = F.pct(r["accuracy"])
        n[f"f1_{k}"] = F.f(r["macro_f1"])
        n[f"det_{k}"] = F.pct(r["detection"])
        n[f"detsd_{k}"] = F.pct(r["detection_std"])
        n[f"miss_{k}"] = F.pct(r["mine_missed"])
        n[f"fa_{k}"] = F.pct(r["false_alarm"])
        n[f"accmin_{k}"] = F.pct(r["accuracy_min"])
        n[f"accmax_{k}"] = F.pct(r["accuracy_max"])
    narrow = per_split(pd.read_csv(RESULTS / "cv" / "svm_narrow_range" / "predictions.csv"))
    n["acc_svm_narrow"] = F.f(narrow["accuracy"].mean())
    cvsum = t("cv_summary.csv").set_index("key")
    n["tune_min_pi_ft"] = F.f(cvsum.loc["pi_ft", "tune_minutes_per_split"], 1)

    params = json.loads((RESULTS / "cv" / "svm" / "params.json").read_text())
    c = pd.Series([v["params"]["C"] for v in params.values()])
    n["svm_c_median"] = f"{c.median():,.0f}"

    sig = t("significance_tests.csv")
    for _, r in sig.iterrows():
        a, b = LABEL_TO_KEY[r["model_a"]], LABEL_TO_KEY[r["model_b"]]
        tag = f"{a}_vs_{b}_{r['metric']}"
        n[f"diff_{tag}"] = F.pct(abs(r["mean_diff"]))
        n[f"pw_{tag}"] = F.peq(r["p_wilcoxon"])
        n[f"pt_{tag}"] = F.peq(r["p_corrected_t"])
        n[f"pth_{tag}"] = F.peq(r["p_corrected_t_holm"])
        n[f"wins_{tag}"], n[f"losses_{tag}"] = str(r["wins"]), str(r["losses"])

    phys = t("physics_comparison.csv")
    for _, r in phys.iterrows():
        a, b = [LABEL_TO_KEY[x.strip()] for x in r["comparison"].split(" vs ")]
        n[f"physgain_{a}_vs_{b}"] = F.pct(r["mean_accuracy_gain"])
        n[f"physbetter_{a}_vs_{b}"] = str(r["splits_better"])
        n[f"physworse_{a}_vs_{b}"] = str(r["splits_worse"])
    d0 = j("cv_details.json")["fitted_offset_d0"]["pi_ft"]
    n["d0_median"], n["d0_min"], n["d0_max"] = (F.f(d0["median"], 2), F.f(d0["min"], 2),
                                                F.f(d0["max"], 2))

    loso = t("loso_summary.csv").set_index("model.1")  # column 1 = key, "model.1" = label
    lkeys = {"MLP, no soil input": "mlp_none", "SVM (RBF)": "svm", "k-NN": "knn",
             "MLP, soil embedding (mean vector for unseen soil)": "mlp_embed",
             "Random forest": "rf", "XGBoost": "xgb",
             "PI-FT-Transformer, factorised soil embedding": "pi_ft_factor",
             "FT-Transformer, factorised soil embedding": "ft_factor",
             "MLP, factorised soil embedding": "mlp_factor",
             "PI-MLP, factorised soil embedding": "pi_mlp_factor"}
    for label, k in lkeys.items():
        n[f"loso_{k}"] = F.f(float(loso.loc[label, "accuracy"].split(" ")[0]))
        n[f"losopct_{k}"] = F.pct(float(loso.loc[label, "accuracy"].split(" ")[0]))
        n[f"losomiss_{k}"] = F.pct(float(loso.loc[label, "mine_missed"].split(" ")[0]))
    by_soil = t("loso_accuracy_by_soil.csv").set_index("Unnamed: 0" if "Unnamed: 0" in
                                                       t("loso_accuracy_by_soil.csv") else "held_out")
    n["loso_dry_humus_min"] = F.pct(by_soil["Dry humus"].min())
    n["loso_dry_humus_max"] = F.pct(by_soil["Dry humus"].max())
    loso_acc = [float(v.split(" ")[0]) for v in loso["accuracy"]]
    n["loso_min"], n["loso_max"] = F.pct(min(loso_acc)), F.pct(max(loso_acc))

    ab = t("ablation.csv")
    ab["key"] = ab["model"].map(LABEL_TO_KEY)
    names = {"no V input": "nov", "no H input": "noh", "no soil input": "nosoil",
             "no physics features": "nophysfeat", "no physics loss": "nophysloss",
             "no physics at all": "nophys", "no noise augmentation": "noaug",
             "no batch norm": "nobn", "no dropout": "nodrop",
             "one-hot soil (no embedding)": "onehot"}
    for _, r in ab.iterrows():
        if r["variant"] in names:
            tag = f"{r['key']}_{names[r['variant']]}"
            n[f"ab_{tag}"] = F.pct(r["delta_accuracy"])
            n[f"abp_{tag}"] = F.peq(r["p_accuracy_corrected_t"])
            n[f"abpw_{tag}"] = F.peq(r["p_accuracy_wilcoxon"])
    sw = t("safety_weight_sweep.csv")
    sw["key"] = sw["model"].map(LABEL_TO_KEY)
    for _, r in sw.iterrows():
        lam = {0.0: "zero", 0.5: "half", 1.0: "one", 2.0: "two", 4.0: "four", 8.0: "eight"}[r["lambda"]]
        n[f"sw_{r['key']}_{lam}_miss"] = F.pct(r["mine_missed"])
        n[f"sw_{r['key']}_{lam}_fa"] = F.pct(r["false_alarm"])
        n[f"sw_{r['key']}_{lam}_acc"] = F.f(r["accuracy"])

    cal = t("calibration.csv").set_index("model")
    for label, k in LABEL_TO_KEY.items():
        n[f"ece_{k}"] = F.f(cal.loc[label, "ece_raw"])
        if k in DL:
            n[f"ecets_{k}"] = F.f(cal.loc[label, "ece_temp_scaled"])
            n[f"ecemc_{k}"] = F.f(cal.loc[label, "ece_mc_dropout"])
            n[f"temp_{k}"] = F.f(cal.loc[label, "median_temperature"], 2)
    unc = t("mc_dropout_uncertainty.csv")
    for _, r in unc.iterrows():
        k = LABEL_TO_KEY[r["model"]]
        pol = "mc" if r["flag_policy"].startswith("MC") else "sm"
        for col in ("flagged_share", "accuracy_confident", "accuracy_flagged",
                    "mine_missed_all", "mine_missed_unflagged"):
            n[f"unc_{k}_{pol}_{col}"] = F.pct(r[col])
        n[f"unc_{k}_{pol}_aurc"] = F.f(r["aurc"])
    sel = t("selective_prediction.csv")
    for _, r in sel.iterrows():
        k = LABEL_TO_KEY[r["model"]]
        sc = {"MC-dropout entropy": "mc", "max softmax": "sm", "max probability": "mp"}[r["score"]]
        n[f"sel_{k}_{sc}_fifty"] = F.pct(r["acc_at_50_mean"])
        n[f"sel_{k}_{sc}_eighty"] = F.pct(r["acc_at_80_mean"])
        n[f"sel_{k}_{sc}_aurc"] = F.f(r["aurc_mean"])
    st = t("selective_prediction_tests.csv")
    for _, r in st.iterrows():
        k = LABEL_TO_KEY[r["baseline"]]
        n[f"selp_{k}_{r['metric']}"] = F.peq(r["p_corrected_t"])
        n[f"selpw_{k}_{r['metric']}"] = F.peq(r["p_wilcoxon"])
        n[f"selwins_{k}_{r['metric']}"] = str(r["wins"])
        n[f"sellosses_{k}_{r['metric']}"] = str(r["losses"])

    nz = t("noise_robustness.csv")
    nz["key"] = nz["model"].map(LABEL_TO_KEY)
    for _, r in nz.iterrows():
        sig_name = {0.0: "zero", 0.05: "fivehundredth", 0.1: "tenth", 0.2: "fifth",
                    0.3: "threetenths", 0.5: "half", 0.75: "threequarters", 1.0: "one"}[r["sigma"]]
        n[f"noise_{r['key']}_{sig_name}_acc"] = F.f(r["accuracy_mean"])
        n[f"noise_{r['key']}_{sig_name}_miss"] = F.pct(r["mine_missed_mean"])
    nt = t("noise_robustness_tests.csv")
    for _, r in nt.iterrows():
        sig_name = {0.0: "zero", 0.05: "fivehundredth", 0.1: "tenth", 0.2: "fifth",
                    0.3: "threetenths", 0.5: "half", 0.75: "threequarters", 1.0: "one"}[r["sigma"]]
        n[f"noisep_{sig_name}_{r['metric']}"] = F.peq(r["p_corrected_t"])
        n[f"noisewins_{sig_name}_{r['metric']}"] = str(r["wins"])
        n[f"noiselosses_{sig_name}_{r['metric']}"] = str(r["losses"])
    # Holm correction over the 8 noise levels (mine-missed comparison).
    from .stats import holm

    mm = nt[nt["metric"] == "mine_missed"].reset_index(drop=True)
    adj = holm(mm["p_corrected_t"].to_numpy())
    survive = [s for s, p in zip(mm["sigma"], adj) if p < 0.05]
    n["noise_holm_min_sigma"] = f"{min(survive):g}" if survive else "none"
    n["noise_holm_half"] = F.peq(adj[list(mm["sigma"]).index(0.5)])
    perm = t("permutation_check.csv")
    n["perm_min"] = F.pct(perm["accuracy_mean"].min())
    n["perm_max"] = F.pct(perm["accuracy_mean"].max())

    shap = t("shap_global.csv")
    tot = shap.groupby(["model", "feature"])["mean_abs_shap"].sum()
    for label, k in (("PI-FT-Transformer (physics features + loss)", "pi_ft"), ("SVM (RBF)", "svm")):
        share = tot[label] / tot[label].sum()
        n[f"shapv_{k}"] = F.pct(share["V (voltage)"], 0)
        n[f"shaph_{k}"] = F.pct(share["H (height)"], 0)
        n[f"shaps_{k}"] = F.pct(share["S (soil)"], 0)

    edge = j("edge_export.json")
    n["edge_params"] = f"{edge['parameters']:,}"
    n["edge_kb_fp32"] = F.f(edge["size_kb"]["onnx_fp32"], 0)
    n["edge_kb_int8"] = F.f(edge["size_kb"]["onnx_int8"], 0)
    n["edge_kb_pt"] = F.f(edge["size_kb"]["pytorch_state_dict"], 0)
    tm = edge["timing_one_cpu_thread"]
    for k, v in tm.items():
        n[f"lat_{k}"] = F.f(1000 * v["batch1_ms"], 0)
        n[f"thr_{k}"] = f"{v['readings_per_s_batch4096']:,.0f}"
    par = edge["parity_vs_pytorch"]
    n["parity_fp32"] = f"{par['fp32']['max_abs_prob_diff_real']:.0e}"
    n["parity_int8_diff"] = F.f(par["int8"]["max_abs_prob_diff_random"], 3)
    n["parity_int8_agree"] = F.pct(par["int8"]["argmax_agreement_random"], 2)
    n["parity_int8_agree_real"] = F.pct(par["int8"]["argmax_agreement_real"], 1)
    n["parity_int8_flag"] = F.pct(par["int8"]["flag_agreement_random"], 2)

    fin = json.loads((RESULTS / "final" / "final.json").read_text())
    n["final_threshold"] = F.f(fin["flag_entropy_threshold"], 3)
    n["final_d0"] = F.f(fin["fitted_offset_d0"], 2)
    n["final_inner_f1"] = F.f(fin["inner_macro_f1"], 3)
    n["final_blocks"] = str(fin["params"]["n_blocks"])
    n["final_dtoken"] = str(fin["params"]["d_token"])
    meta = json.loads((ROOT / "app" / "assets" / "meta.json").read_text())
    n["sim_r2"] = F.f(meta["simulator"]["r2_unsaturated"], 3)

    dev = j("device_benchmark.json")["results"]
    n["dev_ft_cpu"] = F.f(dev["FT-Transformer / cpu"]["train_seconds_per_100_epochs"], 2)
    n["dev_ft_mps"] = F.f(dev["FT-Transformer / mps"]["train_seconds_per_100_epochs"], 2)
    n["dev_mlp_cpu"] = F.f(dev["MLP / cpu"]["train_seconds_per_100_epochs"], 2)
    n["dev_mlp_mps"] = F.f(dev["MLP / mps"]["train_seconds_per_100_epochs"], 2)
    return n


# ---------------------------------------------------------------- tables
def md_main_table(F: Fmt) -> str:
    s = t("stage3_summary.csv").set_index("key")
    lines = ["| Model | Accuracy | Macro F1 | Detection (mine vs none) | Mines missed | False alarms |",
             "|---|---|---|---|---|---|"]
    for k in s.sort_values("accuracy", ascending=False).index:
        r = s.loc[k]
        name = f"**{SHORT[k]}**" if k == "pi_ft" else SHORT[k]
        kind = "" if k in DL else " *(baseline)*"
        lines.append(f"| {name}{kind} | {F.pm(r['accuracy'], r['accuracy_std'])} | "
                     f"{F.f(r['macro_f1'])} | {F.pct(r['detection'])}% | "
                     f"{F.pct(r['mine_missed'])}% | {F.pct(r['false_alarm'])}% |")
    return "\n".join(lines)


def tex_main_table(F: Fmt) -> str:
    s = t("stage3_summary.csv").set_index("key")
    rows = []
    for group in (DL, BASE):
        for k in s.loc[group].sort_values("accuracy", ascending=False).index:
            r = s.loc[k]
            name = rf"\textbf{{{SHORT[k]}}}" if k == "pi_ft" else SHORT[k]
            rows.append(f"{name} & {F.pm(r['accuracy'], r['accuracy_std'])} & {F.f(r['macro_f1'])} & "
                        f"{F.pct(r['detection'])} & {F.pct(r['mine_missed'])} & "
                        f"{F.pct(r['false_alarm'])} \\\\")
        rows.append(r"\midrule")
    rows.pop()
    return "\n".join(rows)


def tex_significance(F: Fmt) -> str:
    sig = t("significance_tests.csv")
    rows = []
    for _, r in sig[sig["family"] == "best neural vs baseline"].iterrows():
        if r["metric"] not in ("accuracy", "mine_missed"):
            continue
        b = SHORT[LABEL_TO_KEY[r["model_b"]]]
        metric = "accuracy" if r["metric"] == "accuracy" else "mines missed"
        rows.append(f"{b} & {metric} & {'+' if r['mean_diff'] >= 0 else '$-$'}"
                    f"{F.pct(abs(r['mean_diff']))} & {r['wins']}/{r['ties']}/{r['losses']} & "
                    f"{F.p(r['p_wilcoxon_holm'])} & {F.p(r['p_corrected_t_holm'])} \\\\")
    return "\n".join(rows)


def tex_loso(F: Fmt) -> str:
    loso = t("loso_summary.csv")
    rows = []
    for _, r in loso.iterrows():
        seen = "--" if pd.isna(r["main_cv_accuracy_soil_seen"]) else F.f(r["main_cv_accuracy_soil_seen"])
        acc = r["accuracy"].replace("±", "$\\pm$")
        rows.append(f"{r['model.1']} & {acc} & {F.pct(float(r['mine_missed'].split(' ')[0]))} & "
                    f"{seen} \\\\")
    return "\n".join(rows)


def tex_ablation(F: Fmt) -> str:
    ab = t("ablation.csv")
    ab = ab[~ab["variant"].str.startswith("safety") & (ab["variant"] != "full model")]
    variants = list(dict.fromkeys(ab["variant"]))
    rows = []
    for v in variants:
        cells = [v.replace("V input", "$V$ input").replace("H input", "$H$ input")]
        for k in ("PI-MLP (physics features + loss)", "PI-FT-Transformer (physics features + loss)"):
            r = ab[(ab["variant"] == v) & (ab["model"] == k)]
            if r.empty:
                cells += ["n/a", ""]
            else:
                r = r.iloc[0]
                sign = "+" if r["delta_accuracy"] >= 0 else "$-$"
                cells += [f"{sign}{F.pct(abs(r['delta_accuracy']))}", F.p(r["p_accuracy_corrected_t"])]
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join(rows)


def tex_uncertainty(F: Fmt) -> str:
    cal = t("calibration.csv").set_index("model")
    unc = t("mc_dropout_uncertainty.csv")
    rows = []
    for label in ("PI-FT-Transformer (physics features + loss)", "FT-Transformer",
                  "PI-MLP (physics features + loss)", "MLP"):
        k = LABEL_TO_KEY[label]
        u = unc[(unc["model"] == label)].set_index("flag_policy")
        mc, sm = u.loc["MC dropout (50 passes)"], u.loc["softmax entropy (1 pass)"]
        c = cal.loc[label]
        rows.append(f"{SHORT[k]} & {F.f(c['ece_raw'])} & {F.f(c['ece_temp_scaled'])} & "
                    f"{F.f(c['ece_mc_dropout'])} & {F.pct(mc['accuracy_confident'])} & "
                    f"{F.pct(mc['accuracy_flagged'])} & {F.pct(sm['accuracy_confident'])} & "
                    f"{F.pct(sm['mine_missed_unflagged'])} \\\\")
    for label in ("SVM (RBF)", "Random forest"):
        c = cal.loc[label]
        rows.append(f"{SHORT[LABEL_TO_KEY[label]]} & {F.f(c['ece_raw'])} & -- & -- & -- & -- & -- & -- \\\\")
    return "\n".join(rows)


def tex_edge(F: Fmt) -> str:
    e = j("edge_export.json")
    tm = e["timing_one_cpu_thread"]
    rows = [("PyTorch (eager)", e["size_kb"]["pytorch_state_dict"], tm["pytorch_eager"]),
            ("ONNX Runtime, FP32", e["size_kb"]["onnx_fp32"], tm["onnx_fp32"]),
            ("ONNX Runtime, INT8", e["size_kb"]["onnx_int8"], tm["onnx_int8"])]
    return "\n".join(f"{name} & {F.f(kb, 0)} & {F.f(1000 * v['batch1_ms'], 0)} & "
                     f"{v['readings_per_s_batch4096']:,.0f} \\\\" for name, kb, v in rows)


def tex_benchmark(F: Fmt) -> str:
    s = t("stage3_summary.csv").set_index("key")
    rows = [r"Meta-heuristic $k$-NN \cite{yilmaz2018} & single split, 113 test & 85.8\,\% & 98.2\,\% \\",
            r"ANN \cite{yilmaz2018} & single split, 113 test & 71.3\,\% & -- \\",
            r"\midrule"]
    for k in ("svm", "pi_ft", "ft", "pi_mlp", "rf"):
        r = s.loc[k]
        rows.append(f"{SHORT[k]} (ours) & nested CV, 25 folds & {F.pct(r['accuracy'])}\\,\\% "
                    f"[{F.pct(r['accuracy_min'])}--{F.pct(r['accuracy_max'])}] & "
                    f"{F.pct(r['detection'])}\\,\\% \\\\")
    return "\n".join(rows)


def macro_name(key: str) -> str:
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
             "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    parts = re.split(r"[^A-Za-z0-9]+", key)
    out = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return "v" + "".join(words.get(ch, ch) for ch in out)


def write_tex() -> dict:
    F = Fmt("tex")
    n = build("tex")
    GEN.mkdir(parents=True, exist_ok=True)
    seen = {}
    lines = ["% Generated by `python -m minesense.numbers` from results/. Do not edit."]
    for k, v in n.items():
        name = macro_name(k)
        assert name not in seen, (name, k, seen.get(name))
        seen[name] = k
        lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    for name, fn in (("table_main", tex_main_table), ("table_significance", tex_significance),
                     ("table_loso", tex_loso), ("table_ablation", tex_ablation),
                     ("table_uncertainty", tex_uncertainty), ("table_edge", tex_edge),
                     ("table_benchmark", tex_benchmark)):
        # \bottomrule lives in the generated file: an \input directly before
        # \bottomrule inside a tabular raises "Misplaced \noalign".
        (GEN / f"{name}.tex").write_text("% Generated from results/. Do not edit.\n" + fn(F)
                                         + "\n\\bottomrule\n")
    (GEN / "macro_index.json").write_text(json.dumps(seen, indent=0) + "\n")
    return n


def render_readme() -> str:
    F = Fmt("md")
    n = build("md")
    n["table_main"] = md_main_table(F)
    template = (ROOT / "docs" / "README.template.md").read_text()

    def sub(m):
        key = m.group(1)
        if key not in n:
            raise KeyError(f"README placeholder {{{{{key}}}}} has no value")
        return n[key]

    body = re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", sub, template)
    return ("<!-- Generated by `python -m minesense.numbers` from docs/README.template.md; "
            "edit the template. -->\n" + body)


def main() -> None:
    write_tex()
    (ROOT / "README.md").write_text(render_readme())
    print("wrote report/generated/ and README.md")


if __name__ == "__main__":
    main()
