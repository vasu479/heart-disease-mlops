"""
preprocessing.py
-----------------
Data loading and preprocessing for the Heart Disease UCI classifier.

Design notes (read before you copy this into your own submission and change it):
- The raw UCI "processed.cleveland.data" file has NO header row and uses "?" for
  missing values. Six of them exist in this dataset (4 in `ca`, 2 in `thal`).
- The original target `num` is an integer 0-4 (0 = no disease, 1-4 = increasing
  severity). Per the assignment ("binary target"), we collapse this to
  0 = no disease, 1 = disease present.
- Imputation + scaling + encoding are all fit INSIDE the pipeline so they are
  learned only on the training split and applied unchanged at inference time.
  Fitting an imputer/scaler on the full dataset before splitting is a common
  data-leakage bug in student submissions for this exact assignment — don't
  reproduce it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target"


def load_raw(path: str) -> pd.DataFrame:
    """Load data/heart_disease.csv, as produced by data/download_data.py.
    That file already carries a header row and its own NaNs; na_values="?" is
    kept only as a defensive fallback for anyone pointing this at a truly raw
    (headerless) UCI file instead."""
    df = pd.read_csv(path, na_values="?")
    missing_cols = set(COLUMN_NAMES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"input file is missing expected columns: {missing_cols}")
    return df


def clean_and_binarize(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize the target and coerce dtypes. Does NOT impute — that lives in
    the pipeline so it only ever sees training-fold statistics."""
    df = df.copy()
    df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    # ca/thal arrive as float64 because of the NaNs; keep them numeric-typed,
    # the ColumnTransformer below treats them as categorical regardless.
    return df


def build_preprocessor() -> ColumnTransformer:
    """Reusable preprocessing pipeline: impute -> scale (numeric) /
    impute -> one-hot (categorical). Persist this together with the model,
    never separately, or inference-time drift becomes a certainty."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience end-to-end loader returning (X, y)."""
    df = load_raw(path)
    df = clean_and_binarize(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
