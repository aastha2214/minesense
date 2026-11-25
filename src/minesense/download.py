"""Download the Kaggle dataset and, optionally, the original UCI archive.

    python -m minesense.download              # Kaggle copy only
    python -m minesense.download --with-uci   # also fetch the source for checks

The Kaggle CLI uses ``~/.kaggle/access_token`` when present. This public
dataset could also be downloaded without a token when the project was built.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import KAGGLE_CSV, KAGGLE_DATASET, KAGGLE_DIR, ROOT, UCI_DIR, UCI_URL, UCI_XLS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_dir(path: Path) -> Path:
    """Return an empty directory to extract untrusted archives into."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def download_kaggle() -> Path:
    cli = Path(sys.executable).parent / "kaggle"
    archive_dir = KAGGLE_DIR.parent / "kaggle_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(cli), "datasets", "download", KAGGLE_DATASET, "-p", str(archive_dir), "--force"],
        check=True,
    )
    archive = next(archive_dir.glob("*.zip"))
    with zipfile.ZipFile(archive) as zf:
        zf.extract(KAGGLE_CSV.name, _fresh_dir(KAGGLE_DIR))
    return KAGGLE_CSV


def download_uci() -> Path:
    UCI_DIR.mkdir(parents=True, exist_ok=True)
    archive = UCI_DIR / "uci_land_mines.zip"
    if not archive.exists():
        urllib.request.urlretrieve(UCI_URL, archive)
    # The archive holds a RAR plus an unrelated Windows installer; take only the RAR.
    outer = _fresh_dir(UCI_DIR / "extracted")
    with zipfile.ZipFile(archive) as zf:
        zf.extract("Mine Dataset.rar", outer)
    # bsdtar (preinstalled on macOS, libarchive-tools on Linux) reads RAR files.
    subprocess.run(
        ["bsdtar", "-xf", str(outer / "Mine Dataset.rar"), "-C", str(_fresh_dir(UCI_DIR / "rar"))],
        check=True,
    )
    return UCI_XLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-uci", action="store_true", help="also download the UCI original")
    args = parser.parse_args()
    paths = [download_kaggle()]
    if args.with_uci:
        paths.append(download_uci())
    for path in paths:
        print(f"{path.relative_to(ROOT)}  sha256={sha256(path)}")


if __name__ == "__main__":
    main()
