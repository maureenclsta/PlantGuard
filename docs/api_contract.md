# PlantGuard frontend API contract

The Streamlit frontend sends `POST {PLANTGUARD_API_BASE_URL}/predict` as `multipart/form-data` with one image field named `file`. Accepted UI formats are JPG, JPEG, and PNG. The default base URL is `http://127.0.0.1:8000`; override it with `PLANTGUARD_API_BASE_URL`.

## Required JSON response

```json
{
  "predicted_class": "one exact class name from models/class_mapping.json",
  "confidence": 0.0
}
```

`confidence` must be a JSON number in the inclusive range 0 to 1. The frontend displays it as a percentage. A missing or malformed required field produces a user-facing error.

## Optional fields understood by the frontend

- `status`: string such as `Healthy` or `Diseased`. If absent, the frontend derives status only from the verified folder-label convention: a class ending in `_healthy` is Healthy; other verified symptom folders are Diseased.
- `gradcam_available`: boolean. Missing or invalid values are treated as false.
- `gradcam_image`: base64 image/data URI or URL. Compatibility aliases `gradcam_base64` and `gradcam_url` are also accepted while the backend schema is being finalized.

Optional values may be omitted. The frontend shows a Grad-CAM placeholder when no image is returned. Missing status does not prevent displaying the prediction.

## Local interface development

Set `PLANTGUARD_MOCK_MODE=true` to show a prominently marked simulated response without calling the backend. Mock mode is for layout development only; it must never be presented as a real prediction. For real mode, use `PLANTGUARD_MOCK_MODE=false` and set the API base URL. The real model's confidence always comes from the backend response.

Run from the repository root with `streamlit run frontend/app.py`.
