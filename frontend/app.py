"""Streamlit upload UI for the PlantGuard FastAPI response contract."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from urllib.parse import urljoin

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.plantguard.api_client import PredictionResponseError, parse_prediction_response

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {"jpg", "jpeg", "png"}


def load_config(path: Path, fallback: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def decode_gradcam(value: str):
    """Return a Pillow image from a data URI or encoded image; URLs are fetched separately."""
    try:
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        return Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
    except Exception:
        return None


st.set_page_config(page_title="PlantGuard AI", page_icon="🌿", layout="centered")
st.title("🌿 PlantGuard AI")
st.write("Upload a leaf image to receive a prediction among the dataset's verified labels.")

mock_mode = os.getenv("PLANTGUARD_MOCK_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
api_base = os.getenv("PLANTGUARD_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
mapping = load_config(ROOT / "models/class_mapping.json", {"class_to_idx": {}})
confidence_config = load_config(ROOT / "frontend/confidence_config.json", {"threshold": None})

if mock_mode:
    st.warning("MOCK MODE: results are simulated for interface preview and are not model predictions.", icon="⚠️")

upload = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
image = None
if upload is not None:
    suffix = Path(upload.name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_SUFFIXES:
        st.error("Invalid file type. Please upload a JPG, JPEG, or PNG image.")
    else:
        try:
            image = Image.open(upload)
            image.verify()
            upload.seek(0)
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Image preview", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError):
            image = None
            st.error("This image cannot be opened. Please upload a readable JPG, JPEG, or PNG file.")

if st.button("Analyze", type="primary", disabled=image is None):
    try:
        if mock_mode:
            classes = mapping.get("class_to_idx", {})
            label = "grape_healthy" if "grape_healthy" in classes else next(iter(classes), "mock_class")
            response = parse_prediction_response({
                "predicted_class": label, "confidence": 0.83,
                "status": "Healthy" if label.endswith("_healthy") else "Diseased",
                "gradcam_available": False,
            })
        else:
            upload.seek(0)
            http_response = requests.post(
                f"{api_base}/predict",
                files={"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")},
                timeout=90,
            )
            http_response.raise_for_status()
            response = parse_prediction_response(http_response.json())

        st.subheader("Prediction result")
        st.markdown(f"**Predicted class:** `{response['predicted_class']}`")
        st.metric("Confidence", f"{response['confidence']:.1%}")
        status = response.get("status")
        if status is None:
            status = "Healthy" if response["predicted_class"].endswith("_healthy") else "Diseased"
        st.write(f"**Status:** {status}")

        threshold = confidence_config.get("threshold")
        if isinstance(threshold, (int, float)) and response["confidence"] < threshold:
            st.warning("Unable to confidently identify this image. Please upload a clearer image.")
        elif threshold is None:
            st.caption("Confidence warning threshold awaits validation-based calibration.")

        gradcam = response.get("gradcam_image")
        if response.get("gradcam_available"):
            st.markdown("**Grad-CAM explanation**")
            shown = False
            if gradcam:
                if gradcam.startswith(("http://", "https://")):
                    try:
                        image_response = requests.get(urljoin(api_base + "/", gradcam), timeout=30)
                        image_response.raise_for_status()
                        st.image(image_response.content, caption="Grad-CAM", use_container_width=True)
                        shown = True
                    except requests.RequestException as exc:
                        st.info(f"Grad-CAM is available, but could not be loaded: {exc}")
                else:
                    heatmap = decode_gradcam(gradcam)
                    if heatmap:
                        st.image(heatmap, caption="Grad-CAM", use_container_width=True)
                        shown = True
            if not shown:
                st.caption("The backend reports Grad-CAM availability; image field is not present or could not be displayed.")
        else:
            st.info("Grad-CAM image will appear here when the backend provides one.")
        st.caption("AI prediction, not expert diagnosis.")
    except (requests.RequestException, ValueError, PredictionResponseError) as exc:
        st.error(f"Could not analyze this image: {exc}")
    except Exception as exc:
        st.error(f"Unexpected prediction response or connection error: {exc}")

