
from pathlib import Path
import logging
from collections import deque
from datetime import datetime, timezone
from threading import Lock

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.twilio_service import send_sos_sms


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mindshield")


# --------------------------------------------------
# PATHS AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "ai"
    / "data"
    / "processed"
    / "mindshield_stress_model.pkl"
)

model = None

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


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="MindShield Backend API",
    description=(
        "MindShield AI stress prediction, "
        "ESP32 sensor monitoring and SOS alert system"
    ),
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class PredictionInput(BaseModel):
    features: dict[str, float]


class AIExplanationItem(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    value: float


class SOSRequest(BaseModel):
    patient_name: str = Field(
        default="Patient",
        min_length=1,
        max_length=100,
    )

    reason: str = Field(
        default="Auto-monitoring",
        min_length=1,
        max_length=500,
    )

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    stress_mindset: str = Field(
        default="UNKNOWN",
        max_length=100,
    )

    ai_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    heart_rate: float | None = Field(default=None, ge=0)
    hrv: float | None = Field(default=None, ge=0)
    respiration: float | None = Field(default=None, ge=0)
    eda: float | None = Field(default=None, ge=0)

    notes: str = Field(
        default="Auto-triggered by MindShield AI",
        max_length=1000,
    )

    ai_explanation: list[AIExplanationItem] = Field(
        default_factory=list,
        max_length=5,
    )

    warning_seconds: int | None = Field(
        default=None,
        ge=0,
    )


# ESP32 SENSOR DATA MODEL

class SensorReading(BaseModel):
    t: float

    hr_bpm: float | None = None
    hrv_rmssd_ms: float | None = None

    eda_state: str = Field(
        default="UNKNOWN",
        max_length=30,
    )

    eda_change_pct: float | None = None

    resp_breaths_min: float | None = None

    acc_x_g: float | None = None
    acc_y_g: float | None = None
    acc_z_g: float | None = None

    dyn_g: float | None = None
    gyro_rms_dps: float | None = None

    activity: str = Field(
        default="UNKNOWN",
        max_length=30,
    )

    flag: str = Field(
        default="NONE",
        max_length=30,
    )

    ppg_status: str = Field(
        default="UNKNOWN",
        max_length=30,
    )

    mpu_err: int = Field(
        default=0,
        ge=0,
    )


# --------------------------------------------------
# ESP32 SENSOR STORAGE
# --------------------------------------------------

sensor_readings = deque(maxlen=300)
sensor_lock = Lock()


# --------------------------------------------------
# LOAD WESAD MODEL
# --------------------------------------------------

@app.on_event("startup")
def load_model():
    global model

    try:
        if not MODEL_PATH.exists():
            logger.error(
                "WESAD model not found: %s",
                MODEL_PATH,
            )
            return

        model = joblib.load(MODEL_PATH)

        logger.info(
            "WESAD Random Forest model loaded."
        )

    except Exception:
        logger.exception(
            "Could not load WESAD model."
        )


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to MindShield Backend",
        "version": "2.4.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/models",
            "/predict",
            "/predict/wesad",
            "/sos",
            "/api/readings",
            "/api/readings/latest",
        ],
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "wesad_model_loaded": model is not None,
        "sos_endpoint": "available",
        "esp32_sensor_endpoint": "available",
    }


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

@app.get("/models")
def get_models():
    return {
        "available_models": [
            {
                "name": "WESAD Random Forest",
                "loaded": model is not None,
                "features_required": len(FEATURES),
            }
        ]
    }


# --------------------------------------------------
# WESAD PREDICTION
# --------------------------------------------------

@app.post("/predict")
@app.post("/predict/wesad")
def predict_wesad(data: PredictionInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="WESAD model is not loaded.",
        )

    try:
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in data.features
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing required features.",
                    "missing_features": missing_features,
                },
            )

        input_data = {
            feature: data.features[feature]
            for feature in FEATURES
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURES,
        )

        prediction = model.predict(input_df)[0]

        stress_probability = 0.0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            classes = list(model.classes_)

            if 1 in classes:
                stress_index = classes.index(1)

                stress_probability = float(
                    probabilities[stress_index]
                )

        result = (
            "STRESS"
            if int(prediction) == 1
            else "NON-STRESS"
        )

        return {
            "dataset": "WESAD",
            "model": "Random Forest",
            "prediction": result,
            "stress_probability": round(
                stress_probability,
                4,
            ),
            "stress_probability_percent": round(
                stress_probability * 100,
                2,
            ),
            "medical_disclaimer": (
                "This is an experimental stress prediction, "
                "not a medical diagnosis."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "WESAD prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=f"WESAD prediction failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# ESP32 SENSOR READING - RECEIVE DATA
# --------------------------------------------------

@app.post("/api/readings")
def receive_sensor_reading(data: SensorReading):

    reading = data.model_dump()

    reading["received_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    with sensor_lock:
        sensor_readings.append(reading)

    logger.info(
        "ESP32 sensor reading received."
    )

    return {
        "success": True,
        "message": "Sensor reading received successfully.",
        "ml_prediction": "PENDING_FEATURE_PIPELINE",
        "reading": reading,
    }


# --------------------------------------------------
# ESP32 SENSOR READING - GET LATEST DATA
# --------------------------------------------------

@app.get("/api/readings/latest")
def get_latest_reading():

    with sensor_lock:
        if not sensor_readings:
            return {
                "success": False,
                "message": "No sensor readings received yet.",
                "reading": None,
            }

        latest = sensor_readings[-1]

    return {
        "success": True,
        "reading": latest,
    }


# --------------------------------------------------
# SOS SMS ENDPOINT
# --------------------------------------------------

@app.post("/sos")
def trigger_sos(data: SOSRequest):

    logger.info(
        "SOS request received for patient: %s",
        data.patient_name,
    )

    try:
        result = send_sos_sms(
            patient_name=data.patient_name,
            reason=data.reason,
            latitude=data.latitude,
            longitude=data.longitude,
            stress_mindset=data.stress_mindset,
            ai_probability=data.ai_probability,
            heart_rate=data.heart_rate,
            hrv=data.hrv,
            respiration=data.respiration,
            eda=data.eda,
            notes=data.notes,
            ai_explanation=[
                item.model_dump()
                for item in data.ai_explanation
            ],
            warning_seconds=data.warning_seconds,
        )

        logger.info(
            "SOS SMS request completed."
        )

        return {
            "success": True,
            "message": "SOS alert processed successfully.",
            "sms": result,
        }

    except ValueError as exc:
        logger.error(
            "SOS configuration error: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "SOS SMS sending failed."
        )

        raise HTTPException(
            status_code=502,
            detail=f"SOS SMS failed: {str(exc)}",
        ) from exc
