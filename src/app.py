from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load model saved with joblib
model = joblib.load("model.pkl")

class InputData(BaseModel):
    age: float
    chol: float
    trestbps: float
    thalach: float
    cp: float
    exang: float

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[ 
        data.age, data.chol, data.trestbps,
        data.thalach, data.cp, data.exang
    ]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    return {
        "prediction": int(pred),
        "confidence": float(prob)
    }