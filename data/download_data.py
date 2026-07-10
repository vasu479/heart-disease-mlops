"""
download_data.py
-----------------
Fetches the Heart Disease UCI dataset (Cleveland subset, 14 attributes).

Set HEART_DATA_SOURCE=mirror to force the exact GitHub mirror file every
number in the report was computed from. CI pins this deliberately — without
it, a GitHub-hosted runner's full internet access means it very likely
succeeds at the ucimlrepo source instead, which is UCI's larger combined
4-hospital dataset (~920 rows) and will NOT match the report's numbers.
"""

import os
import sys
from pathlib import Path

import pandas as pd

OUT_PATH = Path(__file__).parent / "heart_disease.csv"
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

DIRECT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
MIRROR_URL = "https://raw.githubusercontent.com/nyuvis/datasets/master/heart/processed.cleveland.data"


def try_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = COLUMN_NAMES
        print("[download_data] fetched via ucimlrepo (id=45)")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[download_data] ucimlrepo path failed: {exc!r}")
        return None


def try_direct_url(url: str, label: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(url, header=None, names=COLUMN_NAMES, na_values="?")
        if df.shape[0] < 250 or df.shape[1] != 14:
            raise ValueError(f"unexpected shape {df.shape}")
        print(f"[download_data] fetched via {label}: {url}")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[download_data] {label} failed: {exc!r}")
        return None


def main() -> None:
    forced_source = os.environ.get("HEART_DATA_SOURCE")

    if forced_source == "mirror":
        df = try_direct_url(MIRROR_URL, "GitHub mirror (pinned via HEART_DATA_SOURCE)")
    else:
        df = try_ucimlrepo()
        if df is None:
            df = try_direct_url(DIRECT_URL, "direct UCI URL")
        if df is None:
            df = try_direct_url(MIRROR_URL, "GitHub mirror")

    if df is None:
        print("[download_data] ALL sources failed. Check network access / proxy settings.")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[download_data] wrote {df.shape[0]} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print(f"[download_data] missing values per column:\n{df.isna().sum()}")


if __name__ == "__main__":
    main()
