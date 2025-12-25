import argparse
import pandas as pd
import mlflow
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV file passed as a uri_file input.
    """
    print(f"Loading CSV from: {data_path}")
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully. Shape:", df.shape)
    return df


def train_models(X_train, y_train):
    models = {
        "log_reg": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained {name}")
    return models


def evaluate(models, X_test, y_test):
    scores = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        scores[name] = acc
        print(f"{name} accuracy: {acc}")
    return scores


def save_best_model(models, scores):
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"Best model '{best_model_name}' saved to {model_path}")
    return best_model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--experiment-name", type=str)
    args = parser.parse_args()

    mlflow.start_run()

    df = load_data(args.data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    scores = evaluate(models, X_test, y_test)

    for name, acc in scores.items():
        mlflow.log_metric(f"{name}_accuracy", acc)

    best_model_name = save_best_model(models, scores)
    mlflow.log_param("best_model", best_model_name)

    mlflow.end_run()


if __name__ == "__main__":
    main()