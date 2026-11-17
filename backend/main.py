
from pathlib import Path
from contextlib import contextmanager
import json
import logging
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.twilio_service import send_sos_sms


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mindshield")


# --------------------------------------------------
# PATHS, MODEL AND DATABASE
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "ai"
    / "data"
    / "processed"
    / "mindshield_stress_model.pkl"
)

DATABASE_PATH = (
    Path(__file__).resolve().parent
    / "mindshield_history.db"
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
# DATABASE CONNECTION
# --------------------------------------------------

@contextmanager
def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

def initialize_database():

    try:
        with get_db_connection() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS sensor_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL,
                    reading_json TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS sos_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    error_message TEXT
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_received_at
                ON sensor_history(received_at)
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_sos_created_at
                ON sos_history(created_at)
            """)

        logger.info(
            "MindShield SQLite database initialized: %s",
            DATABASE_PATH,
        )

    except Exception:
        logger.exception(
            "Could not initialize MindShield database."
        )
        raise


# --------------------------------------------------
# SAVE SENSOR READING
# --------------------------------------------------

def save_sensor_reading(reading):

    with get_db_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO sensor_history (
                received_at,
                reading_json
            )
            VALUES (?, ?)
            """,
            (
                reading["received_at"],
                json.dumps(reading),
            ),
        )

        return cursor.lastrowid


# --------------------------------------------------
# SAVE SOS EVENT
# --------------------------------------------------

def save_sos_event(
    request_data,
    status,
    response_data=None,
    error_message=None,
):

    with get_db_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO sos_history (
                created_at,
                patient_name,
                status,
                request_json,
                response_json,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                request_data.get("patient_name", "Patient"),
                status,
                json.dumps(request_data, default=str),
                json.dumps(response_data, default=str)
                if response_data is not None
                else None,
                error_message,
            ),
        )

        return cursor.lastrowid


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="MindShield Backend API",
    description=(
        "MindShield AI stress prediction, "
        "ESP32 sensor monitoring, patient history "
        "and SOS alert system"
    ),
    version="2.5.0",
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


# --------------------------------------------------
# ESP32 SENSOR DATA MODEL
# --------------------------------------------------

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
# STARTUP: DATABASE AND WESAD MODEL
# --------------------------------------------------

@app.on_event("startup")
def startup_tasks():

    global model

    initialize_database()

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
        "version": "2.5.0",
        "status": "running",
        "database": "SQLite",
        "endpoints": [
            "/health",
            "/models",
            "/predict",
            "/predict/wesad",
            "/sos",
            "/api/readings",
            "/api/readings/latest",
            "/api/readings/history",
            "/api/sos/history",
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
        "database": "SQLite",
        "database_path": str(DATABASE_PATH),
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
# ESP32: RECEIVE AND SAVE SENSOR DATA
# --------------------------------------------------

@app.post("/api/readings")
def receive_sensor_reading(data: SensorReading):

    reading = data.model_dump()

    reading["received_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    try:

        record_id = save_sensor_reading(reading)

        logger.info(
            "ESP32 sensor reading saved. ID: %s",
            record_id,
        )

        return {
            "success": True,
            "message": "Sensor reading received and saved.",
            "id": record_id,
            "ml_prediction": "PENDING_FEATURE_PIPELINE",
            "reading": reading,
        }

    except Exception as exc:

        logger.exception(
            "Could not save sensor reading."
        )

        raise HTTPException(
            status_code=500,
            detail="Could not save sensor reading.",
        ) from exc


# --------------------------------------------------
# ESP32: GET LATEST SENSOR DATA
# --------------------------------------------------

@app.get("/api/readings/latest")
def get_latest_reading():

    try:

        with get_db_connection() as connection:

            row = connection.execute(
                """
                SELECT id, reading_json
                FROM sensor_history
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:

            return {
                "success": False,
                "message": "No sensor readings received yet.",
                "reading": None,
            }

        reading = json.loads(row["reading_json"])
        reading["id"] = row["id"]

        return {
            "success": True,
            "reading": reading,
        }

    except Exception as exc:

        logger.exception(
            "Could not retrieve latest reading."
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve latest sensor reading.",
        ) from exc


# --------------------------------------------------
# PATIENT SENSOR HISTORY
# --------------------------------------------------

@app.get("/api/readings/history")
def get_reading_history(
    limit: int = Query(default=300, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
):

    try:

        conditions = []
        parameters = []

        if start_date:

            conditions.append(
                "substr(received_at, 1, 10) >= ?"
            )

            parameters.append(start_date)

        if end_date:

            conditions.append(
                "substr(received_at, 1, 10) <= ?"
            )

            parameters.append(end_date)

        where_clause = (
            "WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        with get_db_connection() as connection:

            total = connection.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM sensor_history
                {where_clause}
                """,
                parameters,
            ).fetchone()["total"]

            rows = connection.execute(
                f"""
                SELECT id, received_at, reading_json
                FROM sensor_history
                {where_clause}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                parameters + [limit, offset],
            ).fetchall()

        readings = []

        for row in rows:

            reading = json.loads(row["reading_json"])

            reading["id"] = row["id"]
            reading["received_at"] = row["received_at"]

            readings.append(reading)

        return {
            "success": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "readings": readings,
        }

    except Exception as exc:

        logger.exception(
            "Could not retrieve sensor history."
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve patient sensor history.",
        ) from exc


# --------------------------------------------------
# SOS SMS ENDPOINT
# --------------------------------------------------

@app.post("/sos")
def trigger_sos(data: SOSRequest):

    logger.info(
        "SOS request received for patient: %s",
        data.patient_name,
    )

    request_data = data.model_dump(mode="json")

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

        save_sos_event(
            request_data=request_data,
            status="PROCESSED",
            response_data=result,
        )

        logger.info(
            "SOS request processed."
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

        try:

            save_sos_event(
                request_data=request_data,
                status="FAILED",
                error_message=str(exc),
            )

        except Exception:

            logger.exception(
                "Could not save failed SOS event."
            )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "SOS SMS sending failed."
        )

        try:

            save_sos_event(
                request_data=request_data,
                status="FAILED",
                error_message=str(exc),
            )

        except Exception:

            logger.exception(
                "Could not save failed SOS event."
            )

        raise HTTPException(
            status_code=502,
            detail=f"SOS SMS failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# PATIENT SOS HISTORY
# --------------------------------------------------

@app.get("/api/sos/history")
def get_sos_history(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):

    try:

        with get_db_connection() as connection:

            total = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM sos_history
                """
            ).fetchone()["total"]

            rows = connection.execute(
                """
                SELECT *
                FROM sos_history
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        events = []

        for row in rows:

            events.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "patient_name": row["patient_name"],
                "status": row["status"],
                "request": json.loads(row["request_json"]),
                "response": (
                    json.loads(row["response_json"])
                    if row["response_json"]
                    else None
                ),
                "error_message": row["error_message"],
            })

        return {
            "success": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "events": events,
        }

    except Exception as exc:

        logger.exception(
            "Could not retrieve SOS history."
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve SOS history.",
        ) from exc
