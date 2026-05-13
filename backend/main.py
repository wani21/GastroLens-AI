"""
GastroLens-AI Backend API
=========================
FastAPI server exposing the trained ResNet50 model for:
  - Single image prediction
  - Batch image prediction
  - Video upload + frame-by-frame prediction
  - Health check
"""

import os
import tempfile
from collections import Counter
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.model_service import model_service
from backend.segmentation_model import seg_model, generate_segmentation_mask

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

# =========================================================================
# Lifespan
# =========================================================================


app = FastAPI(
    title="GastroLens-AI API",
    description="Gastrointestinal disease classification (4 classes: "
                "esophagitis, normal, polyp, ulcer)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
MAX_VIDEO_FRAMES = 60

def _validate_image(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

def _validate_video(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{file.content_type}'. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )

@app.get("/")
async def root():
    return {
        "name": "GastroLens-AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
async def health_check():
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
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
    frame_sample_rate: int = Query(10, ge=1, le=100),
):
    _validate_video(file)

    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import cv2
    except ImportError:
        raise HTTPException(status_code=500, detail="OpenCV not installed.")

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

        frame_results = model_service.predict_frames(frames)
        predicted_classes = [r["predicted_class"] for r in frame_results]
        class_counts = Counter(predicted_classes)
        dominant_class, dominant_count = class_counts.most_common(1)[0]

    finally:
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

@app.post("/explain")
async def explain_image(file: UploadFile = File(...), class_index: int = Form(None)):
    _validate_image(file)
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    try:
        result = model_service.generate_gradcam(image_bytes, class_index)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")

@app.post("/predict/video/lstm")
async def predict_video_lstm_endpoint(
    file: UploadFile = File(...),
    frame_sample_rate: int = Query(10, ge=1, le=100)
):
    _validate_video(file)
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = model_service.predict_video_lstm(tmp_path, frame_sample_rate)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM prediction failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    _validate_image(file)
    if seg_model is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")

    image_bytes = await file.read()
    try:
        result = generate_segmentation_mask(image_bytes, seg_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

@app.post("/shap")
async def shap_explain(file: UploadFile = File(...)):
    _validate_image(file)
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    try:
        result = model_service.generate_shap_explanation(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {e}")
