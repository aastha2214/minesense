"""Shared figure style. Figures are saved as 300 dpi PNG and vector PDF."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES_DIR

INK = "#191919"
INK_2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
MUTED = "#cfcfcf"

# Colour-vision-safe categorical order; markers give a second channel for print.
CLASS_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
CLASS_MARKERS = ["o", "s", "^", "D", "v"]
BLUE = CLASS_COLORS[0]
ORANGE = CLASS_COLORS[1]

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#1b64be", "#84b9f3", "#efefef", "#f29c9c", "#c02f30"])

FULL_WIDTH = 7.16  # inches, IEEE two-column text width


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": AXIS,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    # Empty metadata dates keep the PDF byte-stable between runs.
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    plt.close(fig)
