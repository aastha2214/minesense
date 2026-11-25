"""Check that every number in the README and the report matches the results.

    python -m minesense.check_docs

1. report/generated/* and README.md must equal a fresh regeneration from
   results/ (so every macro, table and placeholder is current).
2. The headline table is recomputed from the raw per-fold prediction files and
   compared with results/tables/stage3_summary.csv.
3. Any numeral typed directly into report/main.tex or docs/README.template.md
   (outside macros and placeholders) must be a documented constant of the
   method or a figure quoted from the cited paper; anything else is reported.
Exits with status 1 on any problem.
"""
import re
import sys

import numpy as np
import pandas as pd

from .config import ROOT, TABLES_DIR
from .cv import CV_DIR
from .metrics import per_split
from .numbers import GEN, Fmt, build, render_readme, macro_name
from . import numbers as N

# Numerals that are not results: design constants, settings fixed before the
# experiments, citation details and figures quoted from Yilmaz et al. (2018).
ALLOWED = {
    # protocol, design and settings fixed before the experiments
    "0", "1", "2", "3", "4", "5", "6", "8", "10", "12", "15", "20", "25", "32", "40", "50",
    "60", "80", "100", "4096", "20,000", "0.25", "0.5", "0.05", "0-20",
    # Yilmaz et al. (2018): reported figures, test-set size, pages
    "113", "98.2", "85.8", "71.3", "57.7", "47877", "47870-47888",
    # licence version, Python version in the setup commands
    "4.0", "3.12",
    # EDA threshold (Spearman rho < -0.8); height levels are numbered 0..11
    "0.8", "11",
    # publication years in citations
    "2018", "2021",
}


def strip_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(cite|ref|label|input|includegraphics|graphicspath|bibliography"
                  r"|bibliographystyle|usepackage|documentclass)(\[[^\]]*\])?\{[^}]*\}+", "", text)
    text = re.sub(r"\\v[A-Za-z]+", "", text)                    # result macros
    text = re.sub(r"\\(setlength|tabcolsep)\{[^}]*\}\{[^}]*\}", "", text)
    text = re.sub(r"\[(width|t)[^\]]*\]", "", text)
    text = re.sub(r"\{[lcr|]+\}", "", text)                    # tabular specs
    text = re.sub(r"\\(section|subsection)\*?", "", text)
    return text


def strip_md(text: str) -> str:
    text = re.sub(r"\{\{[a-zA-Z0-9_]+\}\}", "", text)          # placeholders
    text = re.sub(r"```.*?```", "", text, flags=re.S)           # shell commands
    text = re.sub(r"<[^>]+>", "", text)                         # HTML tags
    text = re.sub(r"\([^)]*(png|gif|doi\.org)[^)]*\)", "", text)
    text = re.sub(r"\bfig\d+_\w+", "", text)
    text = re.sub(r"doi:\[[^\]]*\]", "", text)
    return text


def numerals(text: str) -> list[str]:
    text = text.replace("{,}", ",")
    return re.findall(r"(?<![A-Za-z_\\])\d+(?:,\d{3})*(?:\.\d+)?(?:-\d+)?", text)


def check_generated(problems: list) -> None:
    fresh = build("tex")
    lines = [f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}" for k, v in fresh.items()]
    on_disk = (GEN / "numbers.tex").read_text().splitlines()[1:]
    if on_disk != lines:
        problems.append("report/generated/numbers.tex is out of date")
    F = Fmt("tex")
    for name, fn in (("table_main", N.tex_main_table), ("table_significance", N.tex_significance),
                     ("table_loso", N.tex_loso), ("table_ablation", N.tex_ablation),
                     ("table_uncertainty", N.tex_uncertainty), ("table_edge", N.tex_edge),
                     ("table_benchmark", N.tex_benchmark)):
        body = (GEN / f"{name}.tex").read_text()
        if fn(F) not in body:
            problems.append(f"report/generated/{name}.tex is out of date")
    if (ROOT / "README.md").read_text() != render_readme():
        problems.append("README.md is out of date (run python -m minesense.numbers)")
    used = set(re.findall(r"\\(v[A-Za-z]+)", (ROOT / "report" / "main.tex").read_text()))
    defined = {macro_name(k) for k in fresh}
    for m in sorted(used - defined):
        problems.append(f"main.tex uses undefined macro \\{m}")
    print(f"generated files current; {len(used)} result macros used in the report")


def check_raw(problems: list) -> None:
    summ = pd.read_csv(TABLES_DIR / "stage3_summary.csv").set_index("key")
    for key in summ.index:
        s = per_split(pd.read_csv(CV_DIR / key / "predictions.csv"))
        for col in ("accuracy", "macro_f1", "mine_missed", "false_alarm"):
            if not np.isclose(s[col].mean(), summ.loc[key, col], atol=5e-5):
                problems.append(f"{key} {col}: table {summ.loc[key, col]} vs raw {s[col].mean()}")
    print(f"headline metrics of {len(summ)} models recomputed from raw predictions")


def check_literals(problems: list) -> None:
    for path, strip in ((ROOT / "report" / "main.tex", strip_tex),
                        (ROOT / "docs" / "README.template.md", strip_md)):
        found = [x for x in numerals(strip(path.read_text())) if x not in ALLOWED]
        for x in sorted(set(found)):
            problems.append(f"{path.relative_to(ROOT)}: hand-typed numeral '{x}' is not a "
                            "macro/placeholder or an allowed constant")
        print(f"{path.relative_to(ROOT)}: {len(found)} unexplained numerals")


def main() -> None:
    problems: list[str] = []
    check_generated(problems)
    check_raw(problems)
    check_literals(problems)
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        sys.exit(1)
    print("all numbers in README.md and report/main.tex trace back to results/")


if __name__ == "__main__":
    main()
