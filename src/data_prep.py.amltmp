import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

def load_heart_data():
    heart = fetch_openml(name="heart-disease", version=1, as_frame=True)
    df = heart.frame

    return df

def save_clean_data(df):
    base_dir = Path(__file__).resolve().parent.parent
    output_path = base_dir / "data" / "raw" / "heart.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")

if __name__ == "__main__":
    df = load_heart_data()
    save_clean_data(df)