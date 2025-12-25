import argparse
import pandas as pd
import mlflow
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

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

def evaluate_and_plot(models, X_test, y_test):
    """
    Evaluate models and generate ROC, PR curves, and confusion matrix.
    """
    scores = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Compute metrics
        scores[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs) if probs is not None else None
        }

        print(f"\n{name} metrics:")
        for metric, value in scores[name].items():
            print(f"  {metric}: {value}")

        # -------------------------
        # Plot ROC Curve
        # -------------------------
        if probs is not None:
            fpr, tpr, _ = roc_curve(y_test, probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {scores[name]['roc_auc']:.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend(loc="lower right")
            roc_path = f"roc_curve_{name}.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path)

        # -------------------------
        # Plot Precision-Recall Curve
        # -------------------------
        if probs is not None:
            precision, recall, _ = precision_recall_curve(y_test, probs)
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {name}")
            pr_path = f"pr_curve_{name}.png"
            plt.savefig(pr_path)
            plt.close()
            mlflow.log_artifact(pr_path)

        # -------------------------
        # Plot Confusion Matrix
        # -------------------------
        disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.title(f"Confusion Matrix - {name}")
        cm_path = f"confusion_matrix_{name}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

    return scores


def evaluate_with_crossval(models, X, y, cv_splits=5):
    """
    Evaluate models using Stratified K-Fold cross-validation.
    Logs accuracy, precision, recall, F1, and ROC-AUC.
    """
    scores = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    for name, model in models.items():
        print(f"\nRunning {cv_splits}-fold CV for {name}...")

        model_scores = {}
        for metric_name, metric in scoring.items():
            try:
                cv_result = cross_val_score(model, X, y, cv=cv, scoring=metric)
                model_scores[metric_name] = cv_result.mean()
                print(f"{name} {metric_name}: {cv_result.mean():.4f}")
            except Exception:
                # Some models may not support ROC-AUC or other metrics
                model_scores[metric_name] = None

        scores[name] = model_scores

    return scores

def save_best_model(models, scores):
    # as this is health related model choosing F1 score 
    best_model_name = max(scores, key=lambda m: scores[m]["f1"])

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

    with mlflow.start_run():
     df = load_data(args.data_path)
     X = df.drop("target", axis=1)
     y = df["target"]

     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
      )

     models = train_models(X_train, y_train)
     #_ = evaluate_and_plot(models, X_test, y_test)
     scores = evaluate_with_crossval(models, X, y, cv_splits=5)
     for name, metrics in scores.items():
        mlflow.log_metric(f"{name}_f1", metrics["f1"])


     best_model_name = save_best_model(models, scores)
     mlflow.log_param("best_model", best_model_name)



if __name__ == "__main__":
    main()