"""Minefield simulation engine (pure NumPy), shared by the app and the demo GIF.

A rectangular field is split into soil regions; mines of the four types are
hidden at random. A detector sweeps the field cell by cell in a boustrophedon
path at a fixed height. Each reading is generated from the inverse-cube model
fitted to the real dataset (V = a + b / (H + d0)^3 per mine type and soil,
plus the fitted residual noise times a user-set factor, clipped to [0, 1]).
These readings are synthetic: the demo illustrates the workflow, it is not an
evaluation. Held-out accuracy is reported in the README and the report.
"""
from dataclasses import dataclass, field

import numpy as np

NO_MINE = 0


@dataclass
class Field:
    soil: np.ndarray          # (rows, cols) soil index
    truth: np.ndarray         # (rows, cols) class, 0 = no mine
    height: float             # normalised sensor height used for the sweep
    noise: float              # multiplier on the fitted residual noise
    seed: int
    probs: np.ndarray = field(init=False)   # (rows, cols, 5), NaN until scanned
    scanned: np.ndarray = field(init=False)

    def __post_init__(self):
        self.probs = np.full(self.truth.shape + (5,), np.nan)
        self.scanned = np.zeros(self.truth.shape, bool)


def make_field(rows: int, cols: int, mine_density: float, height: float, noise: float,
               seed: int, n_regions: int = 4, n_soils: int = 6) -> Field:
    rng = np.random.default_rng(seed)
    centres = rng.uniform([0, 0], [rows, cols], (n_regions, 2))
    region_soil = rng.integers(0, n_soils, n_regions)
    rr, cc = np.mgrid[0:rows, 0:cols]
    d = ((rr[..., None] - centres[:, 0]) ** 2 + (cc[..., None] - centres[:, 1]) ** 2)
    soil = region_soil[d.argmin(-1)]
    mines = rng.random((rows, cols)) < mine_density
    truth = np.where(mines, rng.integers(1, 5, (rows, cols)), NO_MINE)
    return Field(soil, truth, height, noise, seed)


def sweep_path(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c if r % 2 == 0 else cols - 1 - c) for r in range(rows) for c in range(cols)]


def readings(fld: Field, cells: list[tuple[int, int]], simulator: dict) -> np.ndarray:
    """Synthetic normalised voltages for the given cells."""
    params = {(s["class"], s["soil"]): s for s in simulator["series"]}
    rng = np.random.default_rng([fld.seed, len(cells), *cells[0]])
    out = []
    for r, c in cells:
        p = params[(int(fld.truth[r, c]), int(fld.soil[r, c]))]
        v = p["a"] + p["b"] * (fld.height + simulator["d0"]) ** -3.0
        out.append(v + rng.normal() * p["sd"] * fld.noise)
    return np.clip(np.array(out), 0.0, 1.0)


def scan(fld: Field, cells, simulator: dict, predict_fn) -> None:
    """Measure and classify cells, storing probabilities in the field."""
    v = readings(fld, cells, simulator)
    r, c = np.array(cells).T
    fld.probs[r, c] = predict_fn(v, np.full(len(v), fld.height), fld.soil[r, c])
    fld.scanned[r, c] = True


def score(fld: Field, flag_fn) -> dict:
    """Alarm = predicted class is a mine. Counts over scanned cells."""
    s = fld.scanned
    p = fld.probs[s]
    alarm = p.argmax(1) != NO_MINE
    mine = fld.truth[s] != NO_MINE
    flag = flag_fn(p)
    return {"scanned": int(s.sum()), "mines": int(mine.sum()),
            "found": int((alarm & mine).sum()),
            "missed": int((~alarm & mine).sum()),
            "missed_but_flagged": int((~alarm & mine & flag).sum()),
            "false_alarms": int((alarm & ~mine).sum()),
            "flagged": int(flag.sum()),
            "type_correct": int((p.argmax(1) == fld.truth[s])[mine].sum())}
