import pandas as pd
import numpy as np
from pathlib import Path

from src.data_prep import preprocess_data, save_raw_data, save_processed_data


def make_dummy_df():
    """Create a small synthetic dataset for testing."""
    return pd.DataFrame({
        "age": [50, 60, np.nan, 45],
        "chol": [200, 240, 180, np.nan],
        "sex": ["M", "F", "M", "F"],
        "cp": ["typical", "asymptomatic", "non-anginal", "typical"],
        "target": [1, 0, 1, 0]
    })


def test_preprocess_data_shapes():
    df = make_dummy_df()
    processed = preprocess_data(df)
    # After dropna(), only rows 0 and 1 remain
    assert processed.shape[0] == 2
    # Target column must remain
    assert "target" in processed.columns
    # No NaNs should remain
    assert processed.isna().sum().sum() == 0


def test_preprocess_data_encoding_scaling():
    df = make_dummy_df()
    processed = preprocess_data(df)
    # Target is last column
    X = processed.drop("target", axis=1)
    # All features must be numeric after encoding
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)


def test_save_raw_data(tmp_path, monkeypatch):
    """Ensure save_raw_data writes a file to the correct location."""
    df = make_dummy_df()
    # Patch base_dir to temporary directory
    monkeypatch.setattr(
        "src.data_prep.base_dir",
        tmp_path
    )
    save_raw_data(df)
    expected = tmp_path / "data" / "raw" / "heart.csv"
    assert expected.exists()


def test_save_processed_data(tmp_path, monkeypatch):
    df = make_dummy_df()
    processed = preprocess_data(df)
    monkeypatch.setattr(
        "src.data_prep.base_dir",
        tmp_path
    )
    save_processed_data(processed)
    expected = tmp_path / "data" / "processed" / "heart_cleaned.csv"
    assert expected.exists()