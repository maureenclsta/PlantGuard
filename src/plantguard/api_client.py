"""Defensive parsing for the evolving PlantGuard POST /predict contract."""

from __future__ import annotations

from typing import Any


class PredictionResponseError(ValueError):
    """Raised when required prediction fields are missing or malformed."""


def parse_prediction_response(payload: Any) -> dict[str, Any]:
    """Validate required fields and tolerate absent optional response fields."""
    if not isinstance(payload, dict):
        raise PredictionResponseError("The API response must be a JSON object.")
    label = payload.get("predicted_class")
    confidence = payload.get("confidence")
    if not isinstance(label, str) or not label.strip():
        raise PredictionResponseError("Response is missing a valid predicted_class.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise PredictionResponseError("Response is missing a numeric confidence.")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise PredictionResponseError("confidence must be between 0 and 1.")
    status = payload.get("status")
    if status is not None and not isinstance(status, str):
        status = None
    gradcam_available = payload.get("gradcam_available")
    if not isinstance(gradcam_available, bool):
        gradcam_available = False
    gradcam_image = payload.get("gradcam_image") or payload.get("gradcam_base64") or payload.get("gradcam_url")
    return {
        "predicted_class": label.strip(),
        "confidence": confidence,
        "status": status,
        "gradcam_available": gradcam_available,
        "gradcam_image": gradcam_image if isinstance(gradcam_image, str) else None,
    }
