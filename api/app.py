from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import time
from typing import List

# -------------------------------------------------
# Logging Configuration (Monitoring & Logging Marks)
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------
# App Initialization
# -------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML model serving API with logging",
    version="1.0"
)

# -------------------------------------------------
# Load Model (Reproducible Pipeline)
# -------------------------------------------------
model = joblib.load("models/model.pkl")
logging.info("Model loaded successfully")

# -------------------------------------------------
# Request Schema
# -------------------------------------------------
class PredictionRequest(BaseModel):
    features: List[float]

# -------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.time()

    logging.info(f"Prediction request received: {request.features}")

    prediction = model.predict([request.features])[0]
    probabilities = model.predict_proba([request.features])[0]
    confidence = float(max(probabilities))

    latency = time.time() - start_time

    logging.info(
        f"Prediction={prediction}, Confidence={confidence}, Latency={latency:.4f}s"
    )

    return {
        "prediction": int(prediction),
        "confidence": confidence,
        "latency_seconds": round(latency, 4)
    }