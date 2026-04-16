# GastroLens-AI Backend

FastAPI backend serving the ResNet50 model for gastrointestinal disease classification.

## Features

- **Single image prediction** — `POST /predict`
- **Batch image prediction** — `POST /predict/batch`
- **Video analysis** — `POST /predict/video` (samples frames, predicts each, returns dominant class)
- **Health check** — `GET /health`
- **Interactive docs** — `/docs` (Swagger UI) and `/redoc`

## Prerequisites

- Python 3.11 (same version used for training)
- Trained model at `saved_models/best_model.keras`
- Labels at `saved_models/class_labels.json`

## Setup

From the **project root** (`DL_Project/`):

```powershell
# Activate your virtual environment
.\tf-env\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

## Run the Server

From the **project root**:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Then open:
- API docs (Swagger): http://localhost:8000/docs
- API docs (Redoc): http://localhost:8000/redoc
- Health check: http://localhost:8000/health

## API Endpoints

### `GET /health`
Check if server is up and model is loaded.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": ["esophagitis", "normal", "polyp", "ulcer"]
}
```

### `POST /predict`
Upload a single image, get predicted class and probabilities.

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@test_image.jpg"
```

Response:
```json
{
  "filename": "test_image.jpg",
  "predicted_class": "polyp",
  "predicted_class_index": 2,
  "confidence": 0.987,
  "probabilities": {
    "esophagitis": 0.002,
    "normal": 0.008,
    "polyp": 0.987,
    "ulcer": 0.003
  }
}
```

### `POST /predict/batch`
Upload multiple images in a single request.

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -F "files=@img1.jpg" \
     -F "files=@img2.jpg"
```

### `POST /predict/video`
Upload a video, get frame-by-frame predictions and the dominant class.

```bash
curl -X POST "http://localhost:8000/predict/video?frame_sample_rate=10" \
     -F "file=@endoscopy.mp4"
```

Query params:
- `frame_sample_rate` (1–100, default 10): sample every Nth frame

Response includes:
- `total_frames`, `sampled_frames`, `fps`
- `dominant_class` (most common prediction)
- `class_distribution` (count per class)
- `frame_results` (per-frame predictions)

## Configuration

Override defaults via environment variables:

```powershell
$env:MODEL_PATH = "saved_models/gi_classifier_final.keras"
$env:LABELS_PATH = "saved_models/class_labels.json"
uvicorn backend.main:app --reload
```

## Project Structure

```
backend/
├── __init__.py
├── main.py              # FastAPI app + endpoints
├── model_service.py     # Model loading + inference logic
├── requirements.txt
└── README.md
```

## Notes

- Model is loaded **once at startup** (not per request) for fast inference.
- Videos are sampled and processed in a single batch for efficiency.
- CORS is currently open to all origins — restrict in production.
- Max 60 sampled frames per video to avoid memory issues.
