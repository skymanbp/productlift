"""Download the Olist Brazilian e-commerce dataset into data/raw. FULLY BUILT.

The dataset is real and public (CC BY-NC-SA 4.0), ~120 MB across 9 CSVs.

Two ways to get it:
  1. Kaggle API (automated): set KAGGLE_USERNAME / KAGGLE_KEY in .env, then
     `python -m scripts.download_data --kaggle`
  2. Manual: download from
     https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
     and unzip all CSVs into data/raw/.

Run `python -m scripts.download_data --check` to verify the expected files exist.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.data.load import FILES


def check(raw_dir) -> bool:
    missing = [f for f in FILES.values() if not (raw_dir / f).exists()]
    if missing:
        print("Missing files in", raw_dir)
        for m in missing:
            print("  -", m)
        print("\nSee data/README.md for download instructions.")
        return False
    print(f"All {len(FILES)} Olist files present in {raw_dir}.")
    return True


def download_kaggle(raw_dir) -> None:
    """Download + unzip via the Kaggle API. Requires the `kaggle` package and creds."""
    settings = get_settings()
    if not settings.kaggle_username or not settings.kaggle_key:
        sys.exit("Set KAGGLE_USERNAME and KAGGLE_KEY in .env first (see data/README.md).")
    import os

    os.environ.setdefault("KAGGLE_USERNAME", settings.kaggle_username)
    os.environ.setdefault("KAGGLE_KEY", settings.kaggle_key)
    raw_dir.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("Downloading olistbr/brazilian-ecommerce ...")
    api.dataset_download_files("olistbr/brazilian-ecommerce", path=str(raw_dir), unzip=True)
    check(raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Olist dataset")
    parser.add_argument("--kaggle", action="store_true", help="download via Kaggle API")
    parser.add_argument("--check", action="store_true", help="only verify files exist")
    args = parser.parse_args()

    raw_dir = get_settings().raw_dir
    if args.check:
        sys.exit(0 if check(raw_dir) else 1)
    if args.kaggle:
        download_kaggle(raw_dir)
    else:
        print(__doc__)
        check(raw_dir)


if __name__ == "__main__":
    main()
