import json
import time
import logging
import numpy as np
import joblib

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

# ---------------------------------------------------
# Logging setup
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total API Requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API Request Latency",
    ["endpoint"]
)

# ---------------------------------------------------
# Load model at startup
# ---------------------------------------------------
MODEL_PATH = "outputs/best_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model from {MODEL_PATH}: {e}")
    raise e

# ---------------------------------------------------
# Middleware for logging + metrics
# ---------------------------------------------------
@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    # Logging
    logging.info(
        f"{request.method} {request.url.path} "
        f"Status={response.status_code} "
        f"Latency={latency:.4f}s"
    )

    # Prometheus metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(latency)

    return response

# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------
@app.post("/predict")
async def predict(request: Request):
    try:
        raw_data = await request.body()
        data = json.loads(raw_data)

        # Convert JSON fields into a 2D NumPy array
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
        logging.error(f"Prediction failed: {e}")
        return {"error": str(e)}

# ---------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())