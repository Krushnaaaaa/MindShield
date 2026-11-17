
import os
import logging

from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException


# ==================================================
# CONFIGURATION
# ==================================================

logger = logging.getLogger("mindshield.twilio")

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env", override=True)


# ==================================================
# GET TWILIO CLIENT
# ==================================================

def get_twilio_client():

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        raise ValueError(
            "Twilio credentials missing. "
            "Check your root .env file."
        )

    return Client(account_sid, auth_token)


# ==================================================
# FORMAT SENSOR VALUES
# ==================================================

def format_value(value, unit=""):

    if value is None:
        return "Not available"

    return f"{value} {unit}".strip()


# ==================================================
# SEND WHATSAPP SOS
# ==================================================

def send_sos_sms(
    patient_name,
    reason,
    latitude,
    longitude,
    stress_mindset="UNKNOWN",
    ai_probability=0.0,
    heart_rate=None,
    hrv=None,
    respiration=None,
    eda=None,
    notes="Auto-triggered by MindShield AI",
    ai_explanation=None,
    warning_seconds=None,
):

    # ----------------------------------------------
    # INITIALIZE TWILIO
    # ----------------------------------------------

    client = get_twilio_client()

    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    to_number = os.getenv("SOS_PHONE_NUMBER")

    if not from_number or not to_number:
        raise ValueError(
            "WhatsApp sender or recipient missing "
            "from the root .env file."
        )

    if not from_number.startswith("whatsapp:"):
        raise ValueError(
            "TWILIO_PHONE_NUMBER must start with whatsapp:"
        )

    if not to_number.startswith("whatsapp:"):
        raise ValueError(
            "SOS_PHONE_NUMBER must start with whatsapp:"
        )

    # ----------------------------------------------
    # VALIDATE LOCATION
    # ----------------------------------------------

    if latitude is None or longitude is None:
        raise ValueError("GPS coordinates are required.")

    if not (-90 <= latitude <= 90):
        raise ValueError("Invalid latitude.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Invalid longitude.")

    location_link = (
        "https://www.openstreetmap.org/"
        f"?mlat={latitude}&mlon={longitude}"
        f"#map=18/{latitude}/{longitude}"
    )

    # ----------------------------------------------
    # AI SEVERITY
    # ----------------------------------------------

    if not 0 <= ai_probability <= 1:
        raise ValueError("AI probability must be between 0 and 1.")

    if ai_probability >= 0.80:
        severity = "HIGH"
    elif ai_probability >= 0.50:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    probability_percent = ai_probability * 100

    # ----------------------------------------------
    # AI EXPLANATION
    # ----------------------------------------------

    explanation_text = ""

    if ai_explanation:

        explanation_text = (
            "\n\n🔍 AI EXPLANATION (TOP DRIVERS):"
        )

        for item in ai_explanation[:5]:

            name = item.get("name", "Unknown")
            value = float(item.get("value", 0))

            direction = "↑" if value > 0 else "↓"

            explanation_text += (
                f"\n{direction} {name}: {value:.3f}"
            )

    # ----------------------------------------------
    # PRE-ATTACK WARNING
    # ----------------------------------------------

    warning_text = ""

    if warning_seconds is not None:

        warning_text = (
            f"\n\n⏱ Pre-Attack Warning: "
            f"~{warning_seconds}s to estimated peak"
        )

    # ----------------------------------------------
    # TIMESTAMP
    # ----------------------------------------------

    timestamp = datetime.now().astimezone().strftime(
        "%d-%m-%Y %I:%M:%S %p %Z"
    )

    # ----------------------------------------------
    # BUILD WHATSAPP MESSAGE
    # ----------------------------------------------

    message_body = f"""
🤖 AUTO-TRIGGERED — MINDSHIELD v2.0

Patient: {patient_name}

🧠 Stress Mindset: {stress_mindset}
⚠️ AI Severity: {severity} (p={ai_probability:.2f})

❤️ Heart Rate: {format_value(heart_rate, "bpm")}
📊 HRV: {format_value(hrv, "ms")}
🌬️ Respiration: {format_value(respiration, "/min")}
⚡ EDA: {format_value(eda, "μS")}

📝 Reason: {reason}
Notes: {notes}
{explanation_text}
{warning_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 Location:
{location_link}

🕐 {timestamp}

This is an experimental AI-generated alert,
not a medical diagnosis.
""".strip()

    
    # ----------------------------------------------
    # SEND WHATSAPP TEMPLATE
    # ----------------------------------------------

    try:
        content_sid = os.getenv("TWILIO_CONTENT_SID")

        if not content_sid:
            raise ValueError(
                "TWILIO_CONTENT_SID is missing from .env"
            )

        message = client.messages.create(
            content_sid=content_sid,
            from_=from_number,
            to=to_number
        )

        logger.info(
            "WhatsApp SOS template accepted. SID: %s | Status: %s",
            message.sid,
            message.status
        )

        return {
            "success": True,
            "channel": "WhatsApp",
            "message_sid": message.sid,
            "message_status": message.status,
            "location_link": location_link,
            "severity": severity,
            "recipient": to_number
        }

    except TwilioRestException as exc:
        logger.exception("Twilio WhatsApp API error.")
        raise RuntimeError(
            f"Twilio error {exc.code}: {exc.msg}"
        ) from exc

    except Exception as exc:
        logger.exception("WhatsApp sending failed.")
        raise RuntimeError(
            f"WhatsApp sending failed: {str(exc)}"
        ) from exc
