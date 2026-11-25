"""Paths, class definitions and experiment-wide constants.

Class codes, soil codes and the raw sensor ranges come from Table 1 of
Yilmaz, Kahraman and Soyler, IEEE Access 6, 2018,
doi:10.1109/ACCESS.2018.2866538.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

KAGGLE_DATASET = "ritwikb3/land-mines-detection"
KAGGLE_DIR = RAW_DIR / "kaggle"
KAGGLE_CSV = KAGGLE_DIR / "Land mines.csv"

UCI_URL = "https://archive.ics.uci.edu/static/public/763/land+mines-1.zip"
UCI_DIR = RAW_DIR / "uci"
UCI_XLS = UCI_DIR / "rar" / "Mine Dataset" / "Mine_Dataset.xls"

FEATURES = ["V", "H", "S"]
TARGET = "M"

# Model labels are 0..4 and equal the dataset code M minus 1.
CLASS_NAMES = [
    "No mine",
    "Anti-tank",
    "Anti-personnel",
    "Booby-trapped AP",
    "M14 AP",
]
NO_MINE = 0
N_CLASSES = len(CLASS_NAMES)

# Soil indices are 0..5 and equal the dataset code S minus 1.
SOIL_NAMES = [
    "Dry sandy",
    "Dry humus",
    "Dry limy",
    "Humid sandy",
    "Humid humus",
    "Humid limy",
]
N_SOILS = len(SOIL_NAMES)

# Physical bounds of the instrument as stated in the paper. The published data
# is normalised and the exact transform is not documented.
V_BOUNDS_VOLTS = (0.0, 10.6)
H_BOUNDS_CM = (0.0, 20.0)

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
VAL_FRACTION = 0.15  # carved out of each training fold for early stopping
GLOBAL_SEED = 42
