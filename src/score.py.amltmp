import json
import joblib
import numpy as np

model = None

def init():
    global model
    model = joblib.load("best_model.pkl")
    print("Model loaded successfully")

def run(raw_data):
    try:
        data = json.loads(raw_data)

        features = np.array([[
            data["age"], data["sex"], data["cp"], data["trestbps"],
            data["chol"], data["fbs"], data["restecg"], data["thalach"],
            data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
        ]])

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        return {
            "prediction": int(pred),
            "confidence": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}