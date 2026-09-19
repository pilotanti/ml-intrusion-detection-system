"""Download the NSL-KDD intrusion detection dataset into ./data.

NSL-KDD (Canadian Institute for Cybersecurity, UNB) is also published on
Kaggle as "hassan06/nslkdd". The Kaggle route needs an API token, so this
script pulls the identical files from a public GitHub mirror instead.
Pass --kaggle to use the Kaggle API if you have ~/.kaggle/kaggle.json.
"""

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]


def download_mirror() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        url = MIRROR + name.replace("+", "%2B")
        print(f"[get ] {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        print(f"[done] {name} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_kaggle() -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Install the Kaggle client first: pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("hassan06/nslkdd", path=str(DATA_DIR), unzip=True)
    print(f"Kaggle dataset extracted into {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", action="store_true", help="download via Kaggle API")
    args = parser.parse_args()
    download_kaggle() if args.kaggle else download_mirror()
