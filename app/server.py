"""
FastAPI backend for the DeepFake Detector.
Serves the web UI and provides prediction API endpoints.
"""

import os
import sys
import shutil
import uuid
from pathlib import Path

# Ensure project root is in sys.path (so utils, models, preprocessing are importable)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader

from utils.config import (
    UPLOAD_DIR,
    TEMP_DIR,
    API_HOST,
    API_PORT,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)

# ─── App Setup ────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Detector API",
    description="AI-powered deepfake detection for images and videos using ConvNeXt-Tiny",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Jinja2 templates
jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# ─── Lazy Model Loading ──────────────────────────────────
predictor = None

def get_predictor():
    """Lazy-load the model predictor (loads on first request)."""
    global predictor
    if predictor is None:
        from models.convnext_detector import DeepfakePredictor
        predictor = DeepfakePredictor()
    return predictor


# ─── Routes ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the main web UI."""
    template = jinja_env.get_template("index.html")
    return HTMLResponse(content=template.render())


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict if an uploaded image is real or fake.

    Accepts: image file (JPEG, PNG, BMP, WebP)
    Returns: JSON with label, confidence, Grad-CAM heatmap
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. Allowed: {ALLOWED_IMAGE_EXTENSIONS}",
        )

    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum: {MAX_UPLOAD_SIZE_MB}MB",
        )

    # Save temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Run prediction
        pred = get_predictor()
        result = pred.predict_image(temp_path, return_gradcam=True)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """
    Predict if an uploaded video is real or fake.

    Accepts: video file (MP4, AVI, MOV, MKV, WebM)
    Returns: JSON with verdict, confidence, per-frame analysis
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {ext}. Allowed: {ALLOWED_VIDEO_EXTENSIONS}",
        )

    # Save temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    try:
        # Stream file to disk to handle large files
        with open(temp_path, "wb") as f:
            while chunk := await file.read(8192):
                f.write(chunk)

        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum: {MAX_UPLOAD_SIZE_MB}MB",
            )

        # Run prediction
        pred = get_predictor()
        result = pred.predict_video(temp_path, return_gradcam=True)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "ok", "model": "ConvNeXt-Tiny", "face_detector": "RetinaFace"}


# ─── Run Server ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=API_HOST, port=API_PORT, reload=True)
