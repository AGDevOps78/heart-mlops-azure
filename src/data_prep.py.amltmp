import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

base_dir = Path(__file__).resolve().parent.parent


def load_heart_data():
    """Download the raw Heart Disease dataset from OpenML."""
    heart = fetch_openml(name="heart-disease", version=1, as_frame=True)
    df = heart.frame
    return df


def save_raw_data(df):
    """Save the raw dataset to data/raw/heart.csv."""
    output_path = base_dir / "data" / "raw" / "heart.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved RAW dataset to: {output_path}")


def preprocess_data(df):
    """Clean NaNs, encode categoricals, normalize numericals."""
    df = df.copy()
    # 1. Remove rows with missing values
    df = df.dropna()
    # 2. Identify feature types
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Remove target from feature lists if present
    if "target" in numeric_features:
        numeric_features.remove("target")
    if "target" in categorical_features:
        categorical_features.remove("target")
    # 3. Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
    )
    # 4. Apply transformations
    X = df.drop("target", axis=1)
    y = df["target"]
    X_processed = preprocessor.fit_transform(X)
    # Convert back to DataFrame
    X_processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )
    processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)
    return processed_df


def save_processed_data(df):
    """Save cleaned & processed dataset to data/processed/heart_processed.csv."""
    output_path = base_dir / "data" / "processed" / "heart_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved PROCESSED dataset to: {output_path}")


if __name__ == "__main__":
    df_raw = load_heart_data()
    save_raw_data(df_raw)

    df_processed = preprocess_data(df_raw)
    save_processed_data(df_processed)