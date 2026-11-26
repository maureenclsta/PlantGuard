"""Streamlit upload UI for the PlantGuard FastAPI response contract."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
# Streamlit may start with `frontend/` as sys.path[0]; add the repo root so
# reusable modules resolve regardless of the launch directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plantguard.api_client import PredictionResponseError, parse_prediction_response

ALLOWED_SUFFIXES = {"jpg", "jpeg", "png"}


def load_config(path: Path, fallback: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def decode_gradcam(value: str):
    """Return a Pillow image from a data URI or encoded image; URLs load separately."""
    try:
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        return Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
    except Exception:
        return None


st.set_page_config(page_title="PlantGuard AI", page_icon="\U0001F33F", layout="wide")
st.markdown(
    """
    <style>
      :root { --forest:#174b3b; --leaf:#2d7658; --mint:#e8f2ea; --ink:#1e3028; --muted:#66766e; --paper:#f5f7f2; }
      .stApp { background:var(--paper); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(245,247,242,.88); }
      .block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:2rem; }
      #MainMenu, footer { visibility:hidden; }
      .hero { position:relative; overflow:hidden; padding:2rem 2.2rem; border-radius:24px; color:#f5fbf5;
        background:linear-gradient(118deg,#143f33 0%,#1e6048 62%,#3c8963 100%); margin-bottom:1.15rem; }
      .hero:after { content:""; position:absolute; width:250px; height:250px; right:-50px; top:-110px; border:1px solid rgba(255,255,255,.18); border-radius:50%; box-shadow:0 0 0 30px rgba(255,255,255,.045),0 0 0 60px rgba(255,255,255,.035); }
      .eyebrow { color:#b8ddc5; font:700 .73rem/1.2 sans-serif; letter-spacing:.16em; text-transform:uppercase; }
      .hero h1 { color:#fff; margin:.55rem 0 .5rem; font-size:clamp(1.8rem,4vw,2.65rem); line-height:1.08; letter-spacing:-.035em; }
      .hero p { color:#e0eee5; margin:0; max-width:660px; font-size:1.02rem; }
      .hero-pill { display:inline-block; margin-top:1.15rem; padding:.35rem .68rem; border:1px solid rgba(255,255,255,.27); border-radius:99px; color:#eff9f0; font:600 .75rem sans-serif; }
      .section-kicker { color:var(--leaf); font:700 .72rem sans-serif; letter-spacing:.13em; text-transform:uppercase; margin:.25rem 0 .45rem; }
      .panel-title { font-size:1.2rem; font-weight:750; color:var(--ink); margin:0 0 .2rem; }
      .panel-copy { color:var(--muted); font-size:.91rem; margin:0 0 1rem; }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#dce6dd; border-radius:18px; background:#fff; box-shadow:0 8px 24px rgba(24,61,43,.045); }
      [data-testid="stFileUploaderDropzone"] { background:#f5faf5; border:1px dashed #8aac93; border-radius:14px; }
      [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--leaf); background:#eff7f0; }
      [data-testid="stFileUploaderDropzone"] button { border-radius:9px; background:#1e6048 !important; color:#fff !important; border:0 !important; }
      [data-testid="stFileUploaderDropzone"] button p { color:#fff !important; }
      [data-testid="stFileUploaderDropzone"] button:hover { background:#174b3b !important; }
      div.stButton > button[kind="primary"] { background:var(--forest); border:0; border-radius:11px; padding:.62rem 1.25rem; font-weight:700; box-shadow:0 5px 12px rgba(23,75,59,.16); }
      div.stButton > button[kind="primary"]:hover { background:#24674d; border:0; }
      .tip-row { display:flex; gap:.75rem; align-items:flex-start; margin:.9rem 0; }
      .tip-number { min-width:28px; height:28px; display:grid; place-items:center; border-radius:9px; background:var(--mint); color:var(--forest); font:700 .76rem sans-serif; }
      .tip-text { color:var(--ink); font-size:.9rem; } .tip-text small { display:block; color:var(--muted); margin-top:.12rem; }
      .mock-ribbon { border-radius:12px; padding:.72rem 1rem; background:#fff4d9; border:1px solid #f1d591; color:#6a4b08; font-weight:650; margin:.8rem 0 1.2rem; }
      .result-card { background:#fff; border:1px solid #dce6dd; border-radius:18px; padding:1.2rem 1.35rem; min-height:130px; }
      .result-label { color:var(--muted); font:700 .7rem sans-serif; letter-spacing:.12em; text-transform:uppercase; }
      .result-class { color:var(--forest); font-size:1.35rem; font-weight:750; line-height:1.25; overflow-wrap:anywhere; margin-top:.35rem; }
      .status-chip { display:inline-block; margin-top:.65rem; padding:.28rem .62rem; border-radius:99px; background:#e8f2ea; color:#1d6346; font:700 .75rem sans-serif; }
      .confidence-number { color:var(--forest); font-size:2rem; font-weight:800; line-height:1; margin-top:.5rem; }
      .fine-print { color:var(--muted); font-size:.78rem; }
      .footer-note { border-top:1px solid #dce6dd; margin-top:2rem; padding-top:1rem; color:var(--muted); font-size:.82rem; }
      @media (max-width:700px) { .block-container {padding:1rem .8rem;} .hero {padding:1.4rem;border-radius:18px;} }
    </style>
    """,
    unsafe_allow_html=True,
)

mock_mode = os.getenv("PLANTGUARD_MOCK_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
api_base = os.getenv("PLANTGUARD_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
mapping = load_config(ROOT / "models/class_mapping.json", {"class_to_idx": {}})
confidence_config = load_config(ROOT / "frontend/confidence_config.json", {"threshold": None})

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Plant health, made more approachable</div>
      <h1>A clearer first look at your leaf.</h1>
      <p>Upload a photo to explore a PlantGuard AI prediction across the labels verified in this project dataset.</p>
      <span class="hero-pill">41 verified image labels &nbsp;·&nbsp; AI-assisted screening</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if mock_mode:
    st.markdown(
        '<div class="mock-ribbon">MOCK PREVIEW &nbsp;·&nbsp; Example output is simulated and is not a real model prediction.</div>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1.35, 0.85], gap="large")
image = None
upload = None
upload_fingerprint = None

with left:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">01 / Your image</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Choose a leaf photo</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-copy">A clear, well-lit image helps the model read visible details.</div>', unsafe_allow_html=True)
        upload = st.file_uploader("Upload JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if upload is not None:
            suffix = Path(upload.name).suffix.lower().lstrip(".")
            if suffix not in ALLOWED_SUFFIXES:
                st.error("Invalid file type. Please upload a JPG, JPEG, or PNG image.")
            else:
                try:
                    raw_bytes = upload.getvalue()
                    upload_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
                    checked = Image.open(io.BytesIO(raw_bytes))
                    checked.verify()
                    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    st.image(image, caption=upload.name, use_container_width=True)
                except (UnidentifiedImageError, OSError, ValueError):
                    image = None
                    st.error("This image cannot be opened. Please upload a readable JPG, JPEG, or PNG file.")

with right:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">02 / How it works</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Three simple steps</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tip-row"><span class="tip-number">1</span><div class="tip-text">Upload a leaf image<small>Use one clear subject in the frame.</small></div></div>'
            '<div class="tip-row"><span class="tip-number">2</span><div class="tip-text">Run the analysis<small>PlantGuard checks the image against supported labels.</small></div></div>'
            '<div class="tip-row"><span class="tip-number">3</span><div class="tip-text">Review the result<small>See the predicted class and confidence.</small></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="fine-print">Supported formats: JPG · JPEG · PNG</div>', unsafe_allow_html=True)

if upload_fingerprint != st.session_state.get("analyzed_upload"):
    st.session_state.pop("latest_prediction", None)

button_col, _ = st.columns([1, 2])
with button_col:
    analyze = st.button("Analyze leaf", type="primary", disabled=image is None, use_container_width=True)

if analyze and upload is not None and image is not None:
    try:
        if mock_mode:
            classes = mapping.get("class_to_idx", {})
            label = "grape_healthy" if "grape_healthy" in classes else next(iter(classes), "mock_class")
            parsed = parse_prediction_response({
                "predicted_class": label,
                "confidence": 0.83,
                "status": "Healthy" if label.endswith("_healthy") else "Diseased",
                "gradcam_available": False,
            })
        else:
            http_response = requests.post(
                f"{api_base}/predict",
                files={"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")},
                timeout=90,
            )
            http_response.raise_for_status()
            parsed = parse_prediction_response(http_response.json())
        st.session_state["latest_prediction"] = parsed
        st.session_state["analyzed_upload"] = upload_fingerprint
    except (requests.RequestException, ValueError, PredictionResponseError) as exc:
        st.session_state.pop("latest_prediction", None)
        st.error(f"Could not analyze this image: {exc}")
    except Exception as exc:
        st.session_state.pop("latest_prediction", None)
        st.error(f"Unexpected prediction response or connection error: {exc}")

result = st.session_state.get("latest_prediction") if upload_fingerprint == st.session_state.get("analyzed_upload") else None
if result:
    st.markdown('<div class="section-kicker">03 / Analysis</div>', unsafe_allow_html=True)
    st.markdown("## Your prediction")
    status = result.get("status")
    if status is None:
        status = "Healthy" if result["predicted_class"].endswith("_healthy") else "Diseased"
    result_col, confidence_col = st.columns([1.3, 0.7], gap="medium")
    with result_col:
        st.markdown(
            '<div class="result-card"><div class="result-label">Predicted class</div>'
            f'<div class="result-class">{html.escape(result["predicted_class"])}</div>'
            f'<span class="status-chip">{html.escape(status)}</span></div>',
            unsafe_allow_html=True,
        )
    with confidence_col:
        st.markdown(
            '<div class="result-card"><div class="result-label">Model confidence</div>'
            f'<div class="confidence-number">{result["confidence"]:.1%}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(result["confidence"])

    threshold = confidence_config.get("threshold")
    if isinstance(threshold, (int, float)) and result["confidence"] < threshold:
        st.warning("Unable to confidently identify this image. Please upload a clearer image.")
    elif threshold is None and confidence_config.get("status") == "no_threshold_met_criteria":
        st.info("Validation confidence analysis did not identify a reliable warning threshold.")
    elif threshold is None:
        st.caption("Confidence warning threshold awaits validation-based calibration.")

    with st.container(border=True):
        st.markdown('<div class="section-kicker">Visual explanation</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Grad-CAM</div>', unsafe_allow_html=True)
        gradcam = result.get("gradcam_image")
        if result.get("gradcam_available") and gradcam:
            shown = False
            if gradcam.startswith(("http://", "https://")):
                try:
                    heatmap_response = requests.get(urljoin(api_base + "/", gradcam), timeout=30)
                    heatmap_response.raise_for_status()
                    st.image(heatmap_response.content, caption="Grad-CAM explanation", use_container_width=True)
                    shown = True
                except requests.RequestException as exc:
                    st.info(f"Grad-CAM is available, but could not be loaded: {exc}")
            else:
                heatmap = decode_gradcam(gradcam)
                if heatmap:
                    st.image(heatmap, caption="Grad-CAM explanation", use_container_width=True)
                    shown = True
            if not shown:
                st.caption("Grad-CAM is reported as available, but its image could not be displayed.")
        else:
            st.markdown('<div class="panel-copy">A visual explanation will appear here when the connected API provides one.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note"><strong>For guidance, not diagnosis.</strong> PlantGuard AI provides an AI prediction and is not a substitute for expert agricultural advice.</div>',
    unsafe_allow_html=True,
)
