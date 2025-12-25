import pandas as pd
from src.train import train_models, evaluate

def _make_small_dataset():
    return pd.DataFrame({
        "age":   [50, 60, 55, 65, 58, 62],
        "chol":  [200, 210, 190, 230, 205, 215],
        "target":[0,   1,   0,   1,   0,   1],
    })

def test_train_model_returns_fitted_model():
    df = _make_small_dataset()
    X = df[["age", "chol"]]
    y = df["target"]

    model = train_model(X, y)

    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == len(y)

def test_evaluate_model_returns_valid_f1():
    df = _make_small_dataset()
    X = df[["age", "chol"]]
    y = df["target"]

    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)

    assert "f1" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0