"""
GastroLens-AI Backend API
=========================
FastAPI server exposing the trained ResNet50 model for:
  - Single image prediction
  - Batch image prediction
  - Video upload + frame-by-frame prediction
  - Health check

Run locally:
    uvicorn backend.main:app --reload --port 8000

API docs auto-generated at: http://localhost:8000/docs
"""

import os
import tempfile
from collections import Counter
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.model_service import model_service


# =========================================================================
# Lifespan: load model once at startup, free memory on shutdown
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        model_service.load()
    except Exception as e:
        print(f"[Startup] WARNING: Could not load model: {e}")
        # Continue anyway so health check endpoint still works
    yield
    # Shutdown (nothing to clean up currently)


app = FastAPI(
    title="GastroLens-AI API",
    description="Gastrointestinal disease classification (4 classes: "
                "esophagitis, normal, polyp, ulcer)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow frontend on any origin during development
# In production, restrict allow_origins to your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Utility: validate uploaded file is an image
# =========================================================================
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
MAX_VIDEO_FRAMES = 60  # Cap to avoid memory blow-up on long videos


def _validate_image(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )


def _validate_video(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )


# =========================================================================
# Endpoints
# =========================================================================
@app.get("/")
async def root():
    """Welcome message."""
    return {
        "name": "GastroLens-AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check — verifies server is running and model is loaded.
    Returns 200 if healthy, 503 if model not loaded.
    """
    if model_service.model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"},
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": list(model_service.class_labels.values()),
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the class of a single endoscopy image.

    Returns:
        predicted_class: e.g. "polyp"
        predicted_class_index: e.g. 2
        confidence: float 0-1
        probabilities: dict of all class probabilities
    """
    _validate_image(file)

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = model_service.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "filename": file.filename,
        **result,
    }


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict on multiple images in a single request.
    Images are processed in one forward pass for efficiency.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate all files before processing
    for f in files:
        _validate_image(f)

    images_bytes = [await f.read() for f in files]

    try:
        results = model_service.predict_batch(images_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    return {
        "count": len(results),
        "results": [
            {"filename": f.filename, **r}
            for f, r in zip(files, results)
        ],
    }


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    frame_sample_rate: int = Query(
        10,
        ge=1,
        le=100,
        description="Sample every Nth frame (higher = faster, less detail)",
    ),
):
    """
    Analyze a video by sampling frames and running prediction on each.

    Returns:
        total_frames: total frames in the video
        sampled_frames: how many were analyzed
        dominant_class: most frequent prediction across all sampled frames
        frame_results: per-frame predictions
        class_distribution: count of each predicted class

    Notes:
        - Frames sampled every `frame_sample_rate` (default 10)
        - Capped at 60 sampled frames to avoid memory issues
        - For endoscopy videos, 5-10 fps sampling is usually sufficient
    """
    _validate_video(file)

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Import cv2 lazily — only needed for video
    try:
        import cv2
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="OpenCV not installed. Run: pip install opencv-python-headless",
        )

    # Save upload to a temp file (OpenCV needs a file path, not bytes)
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0

        # Sample frames
        frames = []
        frame_indices = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_sample_rate == 0:
                frames.append(frame)
                frame_indices.append(idx)
                if len(frames) >= MAX_VIDEO_FRAMES:
                    break
            idx += 1
        cap.release()

        if not frames:
            raise HTTPException(status_code=400, detail="No frames could be extracted")

        # Predict on all sampled frames in one batch
        frame_results = model_service.predict_frames(frames)

        # Compute dominant class across all frames
        predicted_classes = [r["predicted_class"] for r in frame_results]
        class_counts = Counter(predicted_classes)
        dominant_class, dominant_count = class_counts.most_common(1)[0]

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "filename": file.filename,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "sampled_frames": len(frames),
        "frame_sample_rate": frame_sample_rate,
        "dominant_class": dominant_class,
        "dominant_class_ratio": round(dominant_count / len(frames), 3),
        "class_distribution": dict(class_counts),
        "frame_results": [
            {"frame_index": i, **r}
            for i, r in zip(frame_indices, frame_results)
        ],
    }
