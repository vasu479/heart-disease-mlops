"""
Tests for src/preprocessing.py — specifically the things that are easy to get
subtly wrong: missing-value handling and the train/test leakage boundary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    clean_and_binarize,
)


@pytest.fixture
def raw_df_with_missing():
    """A small synthetic frame that intentionally injects NaNs into 'ca' and
    'thal', mirroring the real UCI Cleveland file, so this test does not
    depend on the actual dataset containing missing values at all."""
    return pd.DataFrame({
        "age": [63, 67, 41, 57],
        "sex": [1, 1, 0, 1],
        "cp": [1, 4, 2, 4],
        "trestbps": [145, 160, 130, 140],
        "chol": [233, 286, 204, 192],
        "fbs": [1, 0, 0, 0],
        "restecg": [2, 2, 2, 0],
        "thalach": [150, 108, 172, 148],
        "exang": [0, 1, 0, 0],
        "oldpeak": [2.3, 1.5, 1.4, 0.4],
        "slope": [3, 2, 1, 2],
        "ca": [0.0, np.nan, 0.0, np.nan],
        "thal": [6.0, 3.0, np.nan, 3.0],
        "num": [0, 2, 0, 0],
    })


def test_binarize_collapses_multiclass_target(raw_df_with_missing):
    out = clean_and_binarize(raw_df_with_missing)
    assert set(out["target"].unique()) <= {0, 1}
    assert "num" not in out.columns
    # row with num=2 should map to target=1, the rest (num=0) map to target=0
    assert out.loc[1, "target"] == 1
    assert out.loc[0, "target"] == 0


def test_preprocessor_handles_missing_values_without_error(raw_df_with_missing):
    df = clean_and_binarize(raw_df_with_missing)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()


def test_preprocessor_output_is_2d_numeric_array(raw_df_with_missing):
    df = clean_and_binarize(raw_df_with_missing)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert arr.ndim == 2
    assert arr.shape[0] == len(X)
    assert np.issubdtype(arr.dtype, np.number)


def test_preprocessor_fit_on_train_transform_on_unseen_test(raw_df_with_missing):
    """Guards against the classic leakage bug: fit the preprocessor on a
    'train' slice only, then confirm it can still transform an unseen 'test'
    row (with its own missing values) without raising or needing a refit."""
    df = clean_and_binarize(raw_df_with_missing)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    preprocessor.fit(X.iloc[:3])
    transformed_test_row = preprocessor.transform(X.iloc[[3]])
    arr = transformed_test_row.toarray() if hasattr(transformed_test_row, "toarray") else transformed_test_row
    assert arr.shape[0] == 1
