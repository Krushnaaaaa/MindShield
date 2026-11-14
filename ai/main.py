
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------
# APP AND MODEL
# ---------------------------------

app = FastAPI(
    title="MindShield AI API",
    description="Experimental physiological stress classification API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR / "data" / "processed" / "mindshield_stress_model.pkl"
)

model = joblib.load(MODEL_PATH)


# Keep the same feature order used during training.
FEATURES = [
    "EDA_mean",
    "EDA_std",
    "EDA_min",
    "EDA_max",
    "EDA_range",
    "EDA_change_mean",
    "EDA_change_std",
    "EDA_slope",
    "EDA_variability_ratio",
    "BVP_mean",
    "BVP_std",
    "BVP_min",
    "BVP_max",
    "BVP_range",
    "BVP_change_mean",
    "BVP_change_std",
    "ACC_mean",
    "ACC_std",
    "ACC_min",
    "ACC_max",
    "ACC_range",
    "ACC_change_mean",
    "ACC_change_std",
    "TEMP_mean",
    "TEMP_std",
    "TEMP_min",
    "TEMP_max",
    "TEMP_range",
]


# ---------------------------------
# REQUEST FORMAT
# ---------------------------------

class PredictionInput(BaseModel):
    features: dict[str, float]


# ---------------------------------
# ROUTES
# ---------------------------------

@app.get("/")
def home():
    return {
        "message": "MindShield AI API is running",
        "model_loaded": True,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: PredictionInput):

    missing = [name for name in FEATURES if name not in data.features]
    extra = [name for name in data.features if name not in FEATURES]

    if missing or extra:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    values = [data.features[name] for name in FEATURES]
    X = pd.DataFrame([values], columns=FEATURES)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)
    stress_index = classes.index(1)
    stress_probability = float(probabilities[stress_index])

    return {
        "prediction": (
            "STRESS" if prediction == 1 else "NON-STRESS"
        ),
        "stress_probability_percent": round(
            stress_probability * 100, 2
        ),
        "notice": (
            "Experimental estimate only; not a medical diagnosis."
        ),
    }