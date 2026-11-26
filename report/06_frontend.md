# Frontend

> **Key takeaways.** The Streamlit upload and result flow is implemented for the evolving FastAPI contract, with mock mode clearly marked and real predictions sourced only from the API.

The interface accepts JPG/JPEG/PNG files, checks that uploads can be decoded, previews the image, and sends it to `POST /predict`. It displays predicted class, confidence, optional status, and an optional Grad-CAM image or placeholder. Missing optional fields do not crash the display. Mock mode is controlled by `PLANTGUARD_MOCK_MODE=true` and prominently labels simulated output. The API base URL is read from `PLANTGUARD_API_BASE_URL`.

The confidence threshold is **NOT YET AVAILABLE** because validation predictions have not yet been generated from the selected checkpoint. `scripts/propose_confidence_threshold.py` derives it from validation only; the UI reads the value from `frontend/confidence_config.json`.

A screenshot was not supplied; insert an application screenshot here after backend integration. The disclaimer reads: AI prediction, not expert diagnosis. See `docs/api_contract.md` for the expected fields.
