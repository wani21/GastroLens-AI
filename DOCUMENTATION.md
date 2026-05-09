# GastroLens-AI — Complete Project Documentation

> **A single-source reference covering every aspect of the GastroLens-AI project: vision, architecture, code, setup, and roadmap.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Logic & Use Cases](#2-business-logic--use-cases)
3. [System Architecture](#3-system-architecture)
4. [Folder Structure & Codebase Walkthrough](#4-folder-structure--codebase-walkthrough)
5. [Module-by-Module Breakdown](#5-module-by-module-breakdown)
6. [Database Design](#6-database-design)
7. [APIs & Integrations](#7-apis--integrations)
8. [Setup & Installation Guide](#8-setup--installation-guide)
9. [Current Implementation Status](#9-current-implementation-status)
10. [Future Scope & Roadmap](#10-future-scope--roadmap)
11. [Developer Notes & Best Practices](#11-developer-notes--best-practices)
12. [Glossary](#12-glossary)

---

## 1. Project Overview

### 1.1 What is GastroLens-AI?

**GastroLens-AI** is an AI-powered web application that analyzes **gastrointestinal (GI) endoscopy images and videos** to detect common pathologies. A trained deep learning model reviews each image and tells the user whether it shows a healthy GI tract or one of three disease conditions.

It is a full end-to-end product, composed of three parts:

1. **The AI model** — a deep learning classifier trained on thousands of endoscopy images
2. **The backend** — a Python web service that runs the model and exposes it via an API
3. **The frontend** — a web interface where users upload images or videos and see the results

### 1.2 The Problem It Solves

Endoscopy is the primary way doctors look inside the human digestive system. It produces huge amounts of images and video — a single 30-minute procedure can easily generate thousands of frames. Manually reviewing all of this is:

- **Slow** — it takes doctors many hours to review every frame
- **Fatiguing** — long review sessions lead to human errors and missed findings
- **Inconsistent** — two doctors may interpret the same image differently

GastroLens-AI helps by automatically scanning the imagery and flagging the dominant condition in each frame or clip. This does **not** replace a doctor — it acts as an **AI assistant** that narrows down the review time and reduces the chance of missing something.

### 1.3 Target Users

| User Type | How They Use It |
|-----------|-----------------|
| **Gastroenterologists** | Quickly triage endoscopy recordings, highlight suspicious frames |
| **Hospitals & Clinics** | Screening tool to prioritize patients needing expert review |
| **Medical Researchers** | Batch-analyze large datasets for studies |
| **Medical Students** | Study tool — see AI explanations on their own uploaded samples |
| **Telemedicine Platforms** | Automated first-pass analysis for remote clinics |

### 1.4 Key Features

- Upload a **single image** → get instant prediction + confidence scores
- Upload **multiple images** → batch processing in one call
- Upload a **video** → frame sampling, per-frame prediction, dominant class detection
- **Probability visualization** — shows likelihood of each of the 4 classes, not just the top answer
- **Drag-and-drop interface** with image and video previews
- **Stateless API** — usable by any external system (mobile apps, research tools, EHR systems)
- Detects **4 classes**: `normal`, `polyp`, `ulcer`, `esophagitis`

### 1.5 Value Proposition

- **For doctors**: Reduces review time significantly by pre-flagging suspicious frames
- **For hospitals**: Improves screening throughput and reduces missed diagnoses
- **For patients**: Faster, more consistent reporting
- **For developers**: Clean, documented API — can be embedded into any system
- **For the business**: A defensible data + model advantage — we already have a model at 94.79% accuracy, a full web interface, and a deployable backend

---

## 2. Business Logic & Use Cases

### 2.1 Core Workflows

#### Workflow A — Single Image Analysis

1. A clinician opens the web app on a browser
2. They drag a single endoscopy image (JPG/PNG) into the upload area
3. They click **Analyze**
4. The frontend sends the image to the backend (`POST /predict`)
5. The backend preprocesses the image and runs it through the ResNet50 model
6. The backend returns a JSON with: predicted class, confidence, and probabilities for all 4 classes
7. The frontend displays the top class prominently, plus a bar chart of all probabilities

**Example:** Upload a polyp image → result shows "Predicted: polyp, Confidence: 98.7%, with probability bars for all classes."

#### Workflow B — Batch Image Analysis

1. User uploads multiple images in one API call (`POST /predict/batch`)
2. The backend preprocesses and stacks all images into one batch
3. The model processes them in a **single forward pass** (much faster than one-by-one)
4. The backend returns an array of results, one per image

**Use case:** A clinic has 50 images from a screening session — this runs in about 2-3 seconds.

#### Workflow C — Video Analysis

1. User drags a video file (MP4/AVI/MOV) into the upload area on the Video page
2. They can adjust `frame_sample_rate` — default is **every 10th frame**
3. Click Analyze → backend receives the video
4. Backend saves video to temp file, opens with OpenCV
5. Samples frames at the specified rate (capped at 60 frames to prevent memory blow-up)
6. Sends all sampled frames through the model in one batch
7. Aggregates results:
   - **Dominant class** = the most frequently predicted class across all frames
   - **Class distribution** = count of each class
   - **Per-frame results** = prediction + confidence for each sampled frame
8. Returns everything as JSON
9. Frontend displays dominant class, distribution bars, and expandable per-frame details

**Example:** A 45-second endoscopy clip at 30 FPS = 1,350 frames. With `sample_rate=30`, 45 frames are analyzed. If 40 out of 45 are predicted `polyp`, the dominant class is `polyp` with an 89% ratio.

#### Workflow D — Health Check

A simple `GET /health` request that returns whether the server is running and the model is loaded. Used by deployment platforms (Render, Railway, etc.) and for monitoring.

### 2.2 Real-Life Scenarios

#### Scenario 1 — Screening Clinic Triage
A screening clinic records 50 endoscopies per day. Before a gastroenterologist reviews them, each video is sent through GastroLens-AI. Videos flagged with `polyp` or `ulcer` as dominant get prioritized. Videos predicted as `normal` across all sampled frames go to a secondary review queue.

#### Scenario 2 — Training Hub
A medical school uses GastroLens-AI as a learning tool. Students upload their interpretation exercises and compare their own judgment with the AI's. Probability bars teach students which features are "close calls."

#### Scenario 3 — Rural Telemedicine
A rural clinic without a specialist captures endoscopy images and submits them to GastroLens-AI. The AI provides a first-pass prediction. Any case flagged as suspicious (confidence over 80% for a disease class) is forwarded to a city hospital for expert review.

### 2.3 Decision Logic

The model's decision is a **probability vector** over 4 classes, e.g.:

```
{
  "esophagitis": 0.02,
  "normal":      0.05,
  "polyp":       0.91,
  "ulcer":       0.02
}
```

The **predicted class** is simply the one with the highest probability (`argmax`). The **confidence** is that top probability (0.91 in the example).

### 2.4 Edge Cases

| Case | How It's Handled |
|------|------------------|
| Image format not supported (e.g. TIFF) | Backend returns HTTP 400 with a clear error |
| Grayscale or RGBA image | PIL converts it to RGB automatically |
| Very small image (50×50 pixels) | Resized to 224×224 like any other — prediction still runs |
| Empty file uploaded | Backend returns 400 "Empty file" |
| Video with no extractable frames | Returns 400 "No frames could be extracted" |
| Very long video (>60 sampled frames) | Capped at 60 to prevent out-of-memory |
| Model not loaded on server startup | Health check returns 503; `/predict` also returns 503 |
| Uncertain prediction (all classes near 25%) | Still returns argmax — confidence will be low; UI shows probability bars so user sees the uncertainty |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌──────────────────────┐       HTTP / multipart       ┌──────────────────────┐
│   React Frontend     │ ───────────────────────────► │  FastAPI Backend     │
│   (Vite + Tailwind)  │ ◄─────────────────────────── │  (Python 3.11)       │
│   Port 5173          │         JSON response        │  Port 8000           │
└──────────────────────┘                              └──────────┬───────────┘
                                                                 │
                                                                 │ in-process
                                                                 ▼
                                                      ┌─────────────────────┐
                                                      │  TensorFlow / Keras │
                                                      │  ResNet50 Model     │
                                                      │  (best_model.keras) │
                                                      └─────────────────────┘
```

- **Frontend**: A static React application served by Vite's dev server (for development) or built to static files for production
- **Backend**: A Python FastAPI service that loads the model once at startup and serves predictions
- **Model**: A file on disk (`saved_models/best_model.keras`) loaded into memory by the backend
- **No database** — the app is **stateless** (no user accounts, no history storage)

### 3.2 Tech Stack

#### Machine Learning

| Tool | Purpose |
|------|---------|
| **TensorFlow 2.18** | Deep learning framework |
| **Keras** | High-level model API (part of TensorFlow) |
| **ResNet50** | Pretrained CNN used as the feature extractor |
| **ImageNet weights** | Starting point for transfer learning |
| **scikit-learn** | Class weights, classification report, confusion matrix |
| **NumPy** | Array math |
| **Pillow (PIL)** | Image loading |
| **OpenCV** | Video frame extraction |

#### Backend

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Runtime |
| **FastAPI** | Web framework |
| **Uvicorn** | ASGI server that runs the FastAPI app |
| **python-multipart** | Enables file uploads through FastAPI |

#### Frontend

| Tool | Purpose |
|------|---------|
| **React 19** | UI framework |
| **Vite 8** | Build tool and dev server |
| **Tailwind CSS 3** | Utility-first CSS framework |
| **React Router 7** | Page routing |
| **Axios** | HTTP client |

#### Development Tools

| Tool | Purpose |
|------|---------|
| **Git + GitHub** | Source control, hosted at https://github.com/wani21/GastroLens-AI |
| **Virtual environment (venv)** | Python dependency isolation |
| **ESLint** | JavaScript linting |

### 3.3 Data Flow (End-to-End)

Let's trace a single image analysis request:

```
[1] User drags polyp.jpg into browser (Frontend/ImagePage.jsx)
       │
       ▼
[2] Frontend reads the file into a FormData object
       │
       ▼
[3] axios sends POST http://localhost:8000/predict with multipart/form-data
       │
       ▼
[4] FastAPI receives the request at the /predict endpoint (backend/main.py)
       │
       ▼
[5] Endpoint validates the Content-Type (jpg/png/bmp only)
       │
       ▼
[6] Image bytes read via `await file.read()`
       │
       ▼
[7] ModelService.predict() called (backend/model_service.py)
       │
       ▼
[8] Image converted to RGB, resized to 224x224, reshaped to (1,224,224,3)
       │
       ▼
[9] model.predict(batch) runs the forward pass → probabilities (4,)
       │
       ▼
[10] Results formatted into dict: predicted_class, confidence, probabilities
       │
       ▼
[11] FastAPI returns JSON response
       │
       ▼
[12] Frontend receives JSON, updates state, renders ProbabilityBars and result card
```

Total time: **roughly 50-150 ms** on a CPU for a single image.

### 3.4 Design Decisions and Why They Were Made

#### Decision 1: Transfer Learning with ResNet50

**Why:** Training a deep model from scratch would require millions of medical images — we only had ~7,000. Transfer learning leverages patterns (edges, textures, shapes) already learned by ResNet50 from ImageNet's 1.2 million images. Only the top layers need training, which is:
- Much faster (minutes to hours vs. days/weeks)
- Much more accurate with small datasets
- Less prone to overfitting

#### Decision 2: Two-Phase Training (Freeze → Fine-Tune)

**Why:** Starting with the base frozen prevents the randomly initialized classification head from pushing wild gradients into the pretrained weights. After the head learns, unfreezing the top layers with a tiny learning rate gently adapts them to medical imagery without destroying the useful features.

#### Decision 3: Class Weights (Instead of Oversampling)

**Why:** Our dataset is imbalanced — normal/polyp each have about 2,700 training images, while esophagitis/ulcer have only ~920 each. Class weights tell the loss function to penalize mistakes on rare classes more heavily, restoring balance without duplicating data. Oversampling would have inflated training time and risked overfitting.

#### Decision 4: FastAPI (Instead of Flask)

**Why:**
- **Native async support** — ideal for concurrent file uploads
- **Automatic API documentation** — `/docs` endpoint shows a live Swagger UI
- **Type hints + Pydantic validation** — catches bugs at startup
- **Better performance** than Flask for our use case
- **Modern** — the Python community is moving toward it

#### Decision 5: Model Loaded Once at Startup

**Why:** Loading the 97 MB model file takes several seconds. Doing it per request would make every request ridiculously slow. Instead, FastAPI's `lifespan` hook loads it once, and all requests share the in-memory model.

#### Decision 6: Stateless Backend (No Database)

**Why:** In this version, we don't need to store anything. The user uploads an image, gets a result, done. No history, no accounts. This keeps the architecture simple, easy to deploy, and cheap to host. A database can be added later for user accounts and audit logs.

#### Decision 7: React + Vite + Tailwind

**Why:**
- **React** is the most widely used UI framework, easy to hire for
- **Vite** is 10-100x faster than older bundlers (Create React App) during development
- **Tailwind** eliminates CSS file management and makes the design consistent

#### Decision 8: Categorical (One-Hot) Labels Instead of Integer Labels

**Why:** The built-in Keras `F1Score` metric requires one-hot encoded labels. Switching from `sparse_categorical_crossentropy` (integer labels) to `categorical_crossentropy` (one-hot) was the cleanest fix. This is documented in `train.py`.

#### Decision 9: Sample Video Frames (Not Every Frame)

**Why:** A 1-minute endoscopy video at 30 FPS is 1,800 frames. Running inference on all of them would take minutes and crash memory. Sampling every 10th frame gives 180 samples — plenty to find the dominant pathology, and processes in seconds. The sample rate is user-configurable.

---

## 4. Folder Structure & Codebase Walkthrough

### 4.1 Full Directory Tree

```
DL_Project/
│
├── setup.py                        ← Splits Kvasir dataset into train/val/test
├── train.py                        ← Main ML training pipeline
├── EVALUATION_REPORT.md            ← Model evaluation results (accuracy, per-class metrics)
├── DOCUMENTATION.md                ← THIS file
├── .gitignore                      ← Files Git should never track
│
├── dataset/                        ← (gitignored) raw Kvasir-v2 dataset
│   └── kvasir-dataset-v2/
│       ├── polyps/
│       ├── normal-cecum/
│       ├── normal-pylorus/
│       ├── normal-z-line/
│       ├── dyed-lifted-polyps/
│       ├── dyed-resection-margins/
│       ├── ulcerative-colitis/
│       └── esophagitis/
│
├── data/                           ← (gitignored) processed dataset split
│   ├── train/
│   │   ├── normal/
│   │   ├── polyp/
│   │   ├── ulcer/
│   │   └── esophagitis/
│   ├── val/   (same 4 subfolders)
│   └── test/  (same 4 subfolders)
│
├── saved_models/
│   ├── best_model.keras            ← (gitignored) trained model checkpoint (97 MB)
│   ├── gi_classifier_final.keras   ← (gitignored) final model after fine-tuning
│   ├── class_labels.json           ← Tracked — label mapping for backend
│   └── confusion_matrix.csv        ← (gitignored) raw confusion matrix
│
├── tf-env/                         ← (gitignored) Python virtual environment
│
├── backend/
│   ├── __init__.py                 ← Makes backend a Python package
│   ├── main.py                     ← FastAPI app + all endpoints
│   ├── model_service.py            ← Model loading & inference logic
│   ├── requirements.txt            ← Backend-specific dependencies
│   └── README.md                   ← Backend usage guide
│
└── frontend/
    ├── index.html                  ← HTML entry point
    ├── package.json                ← Node.js dependencies
    ├── vite.config.js              ← Vite bundler config
    ├── tailwind.config.js          ← Tailwind theme (defines the accent color)
    ├── postcss.config.js           ← PostCSS pipeline config
    ├── eslint.config.js            ← JavaScript linting rules
    ├── README.md                   ← Frontend usage guide
    ├── public/                     ← Static assets served as-is
    │   ├── favicon.svg
    │   └── icons.svg
    └── src/
        ├── main.jsx                ← React entry point — mounts App
        ├── App.jsx                 ← Top-level component, defines routes
        ├── index.css               ← Global styles + Tailwind directives
        ├── api/
        │   └── client.js           ← Axios instance + API call wrappers
        ├── components/
        │   ├── Navbar.jsx          ← Top navigation bar
        │   ├── Dropzone.jsx        ← Drag-and-drop file upload component
        │   └── ProbabilityBars.jsx ← Visual bar chart for probabilities
        └── pages/
            ├── LandingPage.jsx     ← Home page with project info
            ├── ImagePage.jsx       ← Image upload + analysis page
            └── VideoPage.jsx       ← Video upload + analysis page
```

### 4.2 Root-Level Files

| File | Role |
|------|------|
| `setup.py` | Splits the raw Kvasir dataset into 70/20/10 train/val/test folders and maps sub-folders to the 4 clinical classes |
| `train.py` | End-to-end training pipeline: loads data, builds the model, trains in two phases, evaluates, saves |
| `EVALUATION_REPORT.md` | Formal report of model performance — overall + per-class metrics + confusion matrix |
| `DOCUMENTATION.md` | This single-source project bible |
| `.gitignore` | Tells Git to ignore `dataset/`, `data/`, `tf-env/`, `node_modules/`, `*.keras`, `*.csv`, `frontend/dist/`, etc. |

### 4.3 Backend Folder

| File | Role |
|------|------|
| `__init__.py` | Empty file that turns `backend/` into an importable Python package (so `from backend.model_service import ...` works) |
| `main.py` | Defines the FastAPI app, registers all endpoints, handles CORS, file validation, and video processing |
| `model_service.py` | A clean wrapper class around the Keras model. Handles loading, preprocessing, batch inference, and result formatting |
| `requirements.txt` | Pinned backend dependencies (FastAPI, Uvicorn, TensorFlow, OpenCV, etc.) |
| `README.md` | Specific backend setup and API usage guide |

### 4.4 Frontend Folder

#### `src/` overview

| File | Role |
|------|------|
| `main.jsx` | Entry point. Boots React and mounts the `<App />` into `#root` |
| `App.jsx` | Defines routes (`/`, `/image`, `/video`), includes `<Navbar />`, and the footer |
| `index.css` | Loads the Inter font and Tailwind's `base`, `components`, and `utilities` layers |

#### `src/api/`

| File | Role |
|------|------|
| `client.js` | Creates an `axios` instance pointing to `VITE_API_URL` (defaults to `http://localhost:8000`). Exports helpers: `predictImage(file)`, `predictVideo(file, rate)`, `checkHealth()` |

#### `src/components/`

| File | Role |
|------|------|
| `Navbar.jsx` | Top bar with logo and links to Home, Image, Video. Uses React Router's `NavLink` for active styling |
| `Dropzone.jsx` | Reusable drag-and-drop upload component. Accepts a file callback, MIME type filter, and optional label |
| `ProbabilityBars.jsx` | Renders a list of probability bars sorted by probability. Highlights the top class with the accent color |

#### `src/pages/`

| File | Role |
|------|------|
| `LandingPage.jsx` | Hero section, stats cards, class descriptions, "How it works" section |
| `ImagePage.jsx` | Handles single image upload, preview, API call, result display with probability bars |
| `VideoPage.jsx` | Handles video upload, sample rate config, API call, shows dominant class and per-frame breakdown |

### 4.5 How Parts Interact (Big Picture)

- **`setup.py`** runs once (offline) to prepare the dataset. Produces `data/`.
- **`train.py`** runs once (or when retraining). Reads `data/`, produces `saved_models/best_model.keras` and `saved_models/class_labels.json`.
- **`backend/main.py`** runs as a server. On startup, it calls `model_service.load()`, which reads the model and labels.
- **`frontend/`** is a separate Node.js app. When the user clicks "Analyze", it calls the backend's `/predict` endpoint.
- **The frontend and backend communicate ONLY via HTTP/JSON** — they are completely decoupled.

---

## 5. Module-by-Module Breakdown

### 5.1 `setup.py` — Dataset Preparation

**Purpose:** Convert the raw Kvasir dataset into a clean train/val/test folder structure that Keras can load easily.

**What it does:**
1. Reads folders from `dataset/kvasir-dataset-v2/`
2. Groups raw sub-folders into 4 clinical classes:
   - `normal` ← normal-cecum + normal-pylorus + normal-z-line
   - `polyp` ← polyps + dyed-lifted-polyps + dyed-resection-margins
   - `ulcer` ← ulcerative-colitis
   - `esophagitis` ← esophagitis
3. Shuffles each class's images with a random seed
4. Splits 70% train / 20% validation / 10% test
5. Copies images into `data/{split}/{class}/`

**Key variables:**
- `SOURCE_DIR = "dataset/kvasir-dataset-v2"`
- `BASE_DIR = "data"`
- `train_ratio = 0.7`, `val_ratio = 0.2`, `test_ratio = 0.1`

**Input:** Folder `dataset/kvasir-dataset-v2/` with class sub-folders
**Output:** `data/train/`, `data/val/`, `data/test/` each with 4 class sub-folders

**Dependencies:** Only standard library (`os`, `shutil`, `random`)

### 5.2 `train.py` — Training Pipeline

**Purpose:** Train the ResNet50 classifier end-to-end.

**What it does (14 sections):**

1. **Configuration** — paths, image size (224×224), batch size (32), phase learning rates
2. **Load datasets** — `keras.utils.image_dataset_from_directory` for train/val/test with `label_mode="categorical"` (one-hot labels)
3. **Save class labels JSON** — maps integer index to human-readable class name
4. **Compute class weights** — uses `sklearn.utils.class_weight.compute_class_weight` with `"balanced"` mode
5. **Data augmentation** — random flip, rotation, zoom, contrast (applied ONLY during training)
6. **Performance optimization** — `.cache().prefetch()` on all datasets
7. **Build model** — ResNet50 (frozen) → augmentation → preprocess_input → GAP → Dropout → Dense(256) → Dropout → Dense(4, softmax)
8. **Compile** — Adam optimizer, categorical_crossentropy loss, metrics: accuracy, precision, recall, F1-score
9. **Callbacks** — EarlyStopping, ModelCheckpoint (saves best to `best_model.keras`), ReduceLROnPlateau
10. **Phase 1 training** — 10 epochs with frozen base, LR=1e-3
11. **Phase 2 fine-tuning** — unfreeze layers 143+, LR=1e-5, 15 epochs
12. **Evaluate on test set** — prints all metrics
13. **Classification report + confusion matrix** — saved to `confusion_matrix.csv`
14. **Save final model** — `gi_classifier_final.keras`

**Output files:**
- `saved_models/best_model.keras` — best checkpoint by val_accuracy (used in backend)
- `saved_models/gi_classifier_final.keras` — model after fine-tuning completes
- `saved_models/class_labels.json` — `{"0": "esophagitis", ...}`
- `saved_models/confusion_matrix.csv` — raw matrix data

**Dependencies:** TensorFlow 2.18, Keras, scikit-learn, NumPy

### 5.3 `backend/model_service.py` — Model Service

**Purpose:** A clean, reusable class for all model operations.

**Class:** `ModelService`

**Key methods:**

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `load()` | Loads the `.keras` file and labels JSON | — | — (sets `self.model`) |
| `preprocess_image(bytes)` | Converts raw image bytes to `(1,224,224,3)` array | `bytes` | `np.ndarray` |
| `preprocess_frame(frame)` | Preprocesses an OpenCV frame (BGR→RGB, resize) | `np.ndarray` (frame) | `np.ndarray` |
| `predict(bytes)` | Single image inference | `bytes` | `dict` |
| `predict_batch(list)` | Batch image inference in one forward pass | `List[bytes]` | `List[dict]` |
| `predict_frames(list)` | Batch frame inference | `List[np.ndarray]` | `List[dict]` |
| `_format_result(probs)` | Builds response dict from raw probability vector | `np.ndarray(4,)` | `dict` |

**Configuration via env variables:**
- `MODEL_PATH` (default: `saved_models/best_model.keras`)
- `LABELS_PATH` (default: `saved_models/class_labels.json`)

**Why a class?** It holds the loaded model as instance state and ensures the model is loaded only once.

**Singleton:** At the bottom, `model_service = ModelService()` creates a single global instance imported by `main.py`.

### 5.4 `backend/main.py` — FastAPI App

**Purpose:** Defines the web API.

**Major sections:**

- **Lifespan handler** — loads the model at startup via `model_service.load()`
- **CORS middleware** — allows all origins (tighten in production)
- **File validation helpers** — `_validate_image()` and `_validate_video()` check MIME type

**Endpoints:**

| Method | Path | Role |
|--------|------|------|
| GET | `/` | Welcome message + links to /docs |
| GET | `/health` | Server/model status |
| POST | `/predict` | Single image prediction |
| POST | `/predict/batch` | Batch image prediction |
| POST | `/predict/video` | Video analysis with frame sampling |

**Error handling:** All endpoints return well-formed HTTP errors:
- 400 — bad request (wrong file type, empty file)
- 500 — internal error (prediction failure)
- 503 — model not loaded

**Auto-docs:** Visit `/docs` for Swagger UI, `/redoc` for alternate docs.

### 5.5 `frontend/src/api/client.js` — API Client

**Purpose:** Centralized HTTP client for all backend calls.

**Exports:**
- `apiClient` — a configured `axios` instance
- `predictImage(file)` — POST to `/predict`
- `predictVideo(file, frameSampleRate)` — POST to `/predict/video?frame_sample_rate=N`
- `checkHealth()` — GET `/health`

**Configuration:**
- Base URL from `import.meta.env.VITE_API_URL`, defaults to `http://localhost:8000`
- Timeout set to 120 seconds (for video inference)

### 5.6 `frontend/src/App.jsx` — Routing

**Purpose:** Top-level React component.

**What it does:**
- Wraps everything in `<BrowserRouter>` for client-side routing
- Renders `<Navbar />` at the top
- Defines three routes: `/`, `/image`, `/video`
- Renders a footer at the bottom

### 5.7 `frontend/src/components/Dropzone.jsx` — File Upload

**Purpose:** Reusable drag-and-drop uploader.

**Props:**
- `onFileSelect(file)` — callback when a file is dropped or selected
- `accept` — MIME filter (`"image/*"` or `"video/*"`)
- `label` — placeholder text
- `file` — currently selected file (for display)

**Features:**
- Click to open file picker
- Drag files over → border highlights with accent color
- Drop → calls `onFileSelect`
- Shows file name and size when a file is picked

### 5.8 `frontend/src/components/ProbabilityBars.jsx` — Result Visualization

**Purpose:** Displays the 4-class probability distribution as colored bars.

**Props:**
- `probabilities` — `{esophagitis: 0.02, normal: 0.05, polyp: 0.91, ulcer: 0.02}`
- `predictedClass` — e.g. `"polyp"` (highlighted)

**Behavior:**
- Sorts entries by probability (highest first)
- Uses the accent color for the top class, gray for others
- Smooth animation on probability changes

### 5.9 `frontend/src/pages/LandingPage.jsx` — Home Page

**Purpose:** Marketing and orientation.

**Sections:**
- **Hero** — project title, tagline, CTA buttons to `/image` and `/video`
- **Metrics cards** — 94.79% Accuracy, 94.68% F1-Score, 4 Classes, 7,302 Images
- **Class cards** — description of each of the 4 classes
- **"How It Works"** — 3-step explanation (Upload → Analyze → Results)

### 5.10 `frontend/src/pages/ImagePage.jsx` — Image Analysis

**Purpose:** User-facing image prediction page.

**State:**
- `file` — selected image
- `preview` — object URL for preview
- `result` — backend response
- `loading` — true while waiting for API
- `error` — error message

**User flow:**
1. Drop image → preview shown
2. Click Analyze → loading spinner
3. Receive result → predicted class + probability bars

### 5.11 `frontend/src/pages/VideoPage.jsx` — Video Analysis

**Purpose:** User-facing video prediction page.

**Extra features on top of ImagePage:**
- `sampleRate` state (default 10)
- Video `<video controls>` element for preview
- Expandable per-frame result list (`<details>`)
- Class distribution bars

---

## 6. Database Design

### Current Status: No Database

The current version of GastroLens-AI is **intentionally stateless**. There is no database, no user accounts, no persistent history. This was a deliberate early-stage choice to keep the architecture simple and deployment cheap.

### Future Database Design (Planned)

When the app needs user accounts, uploaded image history, audit logs, and reporting, a database will be introduced. Here is the proposed schema:

#### Proposed Tables (PostgreSQL)

**users**
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| email | VARCHAR(255) | Unique |
| password_hash | VARCHAR(255) | bcrypt |
| role | VARCHAR(50) | `doctor`, `admin`, `researcher` |
| created_at | TIMESTAMP | |

**predictions**
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| user_id | UUID | FK → users.id |
| file_type | VARCHAR(10) | `image` or `video` |
| file_name | VARCHAR(255) | |
| file_hash | VARCHAR(64) | SHA-256 for deduplication |
| storage_path | VARCHAR(500) | S3 URL or local path |
| predicted_class | VARCHAR(20) | `normal`, `polyp`, `ulcer`, `esophagitis` |
| confidence | FLOAT | 0–1 |
| probabilities | JSONB | Full probability dict |
| created_at | TIMESTAMP | |

**video_frame_results** (for video predictions)
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | |
| prediction_id | UUID | FK → predictions.id |
| frame_index | INTEGER | |
| predicted_class | VARCHAR(20) | |
| confidence | FLOAT | |

**audit_logs**
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | |
| user_id | UUID | Who did it |
| action | VARCHAR(100) | e.g. `prediction_created`, `login` |
| metadata | JSONB | |
| created_at | TIMESTAMP | |

### Migration Strategy

When we're ready:
1. Add SQLAlchemy + Alembic to backend
2. Configure PostgreSQL connection (via `DATABASE_URL` env var)
3. Define models in `backend/models/` directory
4. Run initial migration
5. Add JWT authentication middleware
6. Update endpoints to save predictions automatically

---

## 7. APIs & Integrations

### 7.1 Base URL

- **Local development:** `http://localhost:8000`
- **Interactive docs:** `http://localhost:8000/docs`
- **Production:** (To be deployed — will be added to `.env` as `VITE_API_URL`)

### 7.2 Authentication

Currently **none**. Anyone who can reach the server can call the API. Authentication (JWT-based) will be added alongside the database in a future phase.

### 7.3 Endpoints

#### `GET /`

Welcome info.

**Response:**
```json
{
  "name": "GastroLens-AI API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

#### `GET /health`

Health check for monitoring systems.

**Response (200, healthy):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": ["esophagitis", "normal", "polyp", "ulcer"]
}
```

**Response (503, unhealthy):**
```json
{
  "status": "unhealthy",
  "reason": "Model not loaded"
}
```

#### `POST /predict`

**Request (multipart/form-data):**
- `file` (required) — image file (JPEG, PNG, or BMP)

**Example:**
```bash
curl -X POST http://localhost:8000/predict -F "file=@polyp.jpg"
```

**Response (200):**
```json
{
  "filename": "polyp.jpg",
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

**Error responses:**
- 400 — unsupported content type or empty file
- 503 — model not loaded
- 500 — prediction error

#### `POST /predict/batch`

**Request (multipart/form-data):**
- `files` (required, repeatable) — multiple image files

**Example:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

**Response (200):**
```json
{
  "count": 2,
  "results": [
    {
      "filename": "img1.jpg",
      "predicted_class": "normal",
      "predicted_class_index": 1,
      "confidence": 0.96,
      "probabilities": {...}
    },
    {
      "filename": "img2.jpg",
      "predicted_class": "ulcer",
      "predicted_class_index": 3,
      "confidence": 0.88,
      "probabilities": {...}
    }
  ]
}
```

#### `POST /predict/video`

**Request (multipart/form-data):**
- `file` (required) — video file (MP4, AVI, MOV)
- `frame_sample_rate` (query param, default=10, range 1-100)

**Example:**
```bash
curl -X POST "http://localhost:8000/predict/video?frame_sample_rate=10" \
  -F "file=@endoscopy.mp4"
```

**Response (200):**
```json
{
  "filename": "endoscopy.mp4",
  "total_frames": 1350,
  "fps": 30.0,
  "sampled_frames": 45,
  "frame_sample_rate": 10,
  "dominant_class": "polyp",
  "dominant_class_ratio": 0.844,
  "class_distribution": {
    "polyp": 38,
    "normal": 7
  },
  "frame_results": [
    {
      "frame_index": 0,
      "predicted_class": "polyp",
      "predicted_class_index": 2,
      "confidence": 0.97,
      "probabilities": {...}
    },
    ...
  ]
}
```

### 7.4 External Services & Integrations

**Currently:** None required. The app runs fully offline (after pip installs).

**Used during development:**
- **GitHub** — source control (https://github.com/wani21/GastroLens-AI)
- **Google Fonts** — Inter font loaded in the frontend
- **TensorFlow / Keras weights server** — downloads ResNet50 ImageNet weights on first training run

**Planned future integrations:**
- **AWS S3 or similar** — for uploaded image/video storage
- **SendGrid or Postmark** — for account emails when auth is added
- **Stripe** — if we add paid plans

---

## 8. Setup & Installation Guide

### 8.1 Prerequisites

| Tool | Minimum Version | Why |
|------|-----------------|-----|
| **Python** | 3.11.x (not 3.13+) | TensorFlow 2.18 supports 3.9–3.12 |
| **Node.js** | 18 or newer | For Vite and React |
| **Git** | Any recent version | For cloning the repo |
| **GPU (optional)** | NVIDIA with CUDA | Only for training, not inference. Native Windows TF has no GPU — use WSL2 for GPU |

### 8.2 Clone the Repository

```bash
git clone https://github.com/wani21/GastroLens-AI.git
cd GastroLens-AI
```

### 8.3 Set Up Python Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create venv using Python 3.11
py -3.11 -m venv tf-env

# Activate it
.\tf-env\Scripts\Activate.ps1

# If you get a script execution error:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\tf-env\Scripts\Activate.ps1

# Confirm version
python --version   # should say 3.11.x
```

**On macOS / Linux:**
```bash
python3.11 -m venv tf-env
source tf-env/bin/activate
```

### 8.4 Install Python Dependencies

**For training only:**
```bash
pip install tensorflow==2.18.0 scikit-learn pillow numpy
```

**For the backend server:**
```bash
pip install -r backend/requirements.txt
```

### 8.5 Install Node.js Dependencies (Frontend)

```bash
cd frontend
npm install
cd ..
```

### 8.6 Prepare the Dataset (Only if Training)

1. Download Kvasir v2 from https://datasets.simula.no/kvasir/
2. Extract it into `dataset/kvasir-dataset-v2/`
3. Run:
```bash
python setup.py
```

This creates `data/train/`, `data/val/`, `data/test/`.

### 8.7 Train the Model (Only if Not Using Pre-trained)

```bash
python train.py
```

This takes:
- **~10-15 minutes per epoch on CPU** (expect 4-6 hours total)
- **~2-3 minutes per epoch on NVIDIA GPU via WSL2**

Output: `saved_models/best_model.keras` + `class_labels.json`

### 8.8 Run the Backend Server

From project root:
```bash
uvicorn backend.main:app --reload --port 8000
```

Verify it's running:
- Open http://localhost:8000/docs for the interactive API
- Open http://localhost:8000/health for the health check

### 8.9 Run the Frontend Dev Server

In a **new terminal**:
```bash
cd frontend
npm run dev
```

Open http://localhost:5173

### 8.10 Environment Configuration

#### Frontend
Create `frontend/.env` (gitignored):
```
VITE_API_URL=http://localhost:8000
```

For production deployment, set this to your deployed backend URL.

#### Backend
Environment variables (all optional):
- `MODEL_PATH` — override model location (default: `saved_models/best_model.keras`)
- `LABELS_PATH` — override labels file (default: `saved_models/class_labels.json`)

### 8.11 Common Setup Issues and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'tensorflow'` | Using wrong Python version or venv not activated | Activate venv, confirm Python 3.11 |
| `ModuleNotFoundError: pydantic_core._pydantic_core` | Mixed Python versions in venv | Recreate venv from scratch with Python 3.11 |
| `Could not find a version that satisfies the requirement tensorflow` | Python 3.13 or 3.14 active | Install Python 3.11, recreate venv |
| CORS error in browser console | Frontend and backend on different origins, CORS not set | Already set to `allow_origins=["*"]` in dev |
| Model loads very slowly on every request | Model being loaded per-request | Already fixed via FastAPI lifespan |
| `Set-ExecutionPolicy` blocks venv activation | Windows security policy | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## 9. Current Implementation Status

### 9.1 Completed ✅

#### Machine Learning
- [x] Dataset preprocessing and train/val/test split (`setup.py`)
- [x] Two-phase transfer learning pipeline with ResNet50 (`train.py`)
- [x] Class weights for imbalance handling
- [x] Data augmentation (random flip, rotation, zoom, contrast)
- [x] Metrics: accuracy, precision, recall, F1-score
- [x] Early stopping, model checkpointing, LR reduction on plateau
- [x] Confusion matrix and per-class classification report
- [x] Trained model achieving **94.79% test accuracy** and **94.68% weighted F1**
- [x] Evaluation report document (`EVALUATION_REPORT.md`)

#### Backend
- [x] FastAPI app structure
- [x] Model loading at startup (lifespan)
- [x] Single image prediction endpoint (`/predict`)
- [x] Batch image prediction endpoint (`/predict/batch`)
- [x] Video prediction with frame sampling (`/predict/video`)
- [x] Health check endpoint (`/health`)
- [x] CORS middleware
- [x] File type validation
- [x] Temp file cleanup for videos
- [x] Clean separation: `ModelService` class
- [x] Backend README

#### Frontend
- [x] Vite + React + Tailwind scaffold
- [x] React Router setup (3 pages)
- [x] Navbar component
- [x] Drag-and-drop upload component
- [x] Probability bars visualization
- [x] Landing page with metrics, classes, how-it-works
- [x] Image analysis page with preview + result
- [x] Video analysis page with sample rate config + per-frame breakdown
- [x] Minimalist black/white/accent styling
- [x] Axios client with env-configurable base URL
- [x] Production build working (92.5 KB gzipped)
- [x] Frontend README

#### DevOps / Documentation
- [x] GitHub repository set up with clean commit history
- [x] Sensitive folders properly gitignored (data, model binaries, venv, node_modules)
- [x] `.gitignore` covers all edge cases
- [x] This documentation file

### 9.2 Partially Done ⚠️

- [ ] **Local end-to-end test** — backend and frontend haven't been run together on the main laptop yet. Friend's laptop was used for training only. The flow needs a successful manual test.
- [ ] **OpenCV install** — video endpoint needs `opencv-python-headless`. Code path is ready but not yet exercised on this machine.
- [ ] **README.md at repo root** — currently exists as the original simple README; could be expanded with quick-start.

### 9.3 Known Limitations

- **No GPU on native Windows** — TensorFlow 2.11+ dropped native Windows GPU support. Use WSL2 for GPU training.
- **No authentication** — anyone with the URL can use the API. Fine for local/demo, not for production.
- **No database** — nothing is persisted. If the backend restarts, no data is lost, but no history is kept either.
- **Max 60 sampled frames per video** — longer videos are truncated. Can be increased if memory allows.
- **Esophagitis recall is lower** (0.76) — the model confuses some esophagitis cases with normal tissue. Matches clinical reality — early esophagitis looks very similar to healthy mucosa.
- **CORS wide open** — `allow_origins=["*"]`. Needs tightening in production.
- **Model size (97 MB)** — relatively large; will impact cold-start times on serverless deployments.
- **No offline fallback** — frontend won't work without the backend running.
- **No progress indicator for long video uploads** — just a spinner. Real upload progress could be added.

### 9.4 Known Minor Bugs

- **LF/CRLF line ending warnings on Windows** when committing — harmless, but noisy. Can be fixed with `git config core.autocrlf true`.
- **Hot reload on backend** sometimes fails to pick up changes in `model_service.py` — works after restart.

---

## 10. Future Scope & Roadmap

### 10.1 Short Term (Next 1-2 Weeks)

- [ ] End-to-end local test with a real image and video
- [ ] Deploy backend to Render / Railway / Fly.io
- [ ] Deploy frontend to Vercel / Netlify
- [ ] Configure proper CORS allowlist for production
- [ ] Add loading progress bar for video uploads
- [ ] Polish the root `README.md` with screenshots and quick-start

### 10.2 Medium Term (1-3 Months)

- [ ] **User accounts**: JWT auth, register/login endpoints
- [ ] **PostgreSQL database**: predictions history, audit logs
- [ ] **S3 storage**: uploaded images/videos persisted
- [ ] **Improved esophagitis recall**: targeted augmentation, focal loss, or ensemble with a second model
- [ ] **Bounding box / heatmap visualization**: show *where* the model is looking (Grad-CAM)
- [ ] **PDF report export**: generate a clinician-ready PDF with findings
- [ ] **Admin dashboard**: view overall usage metrics
- [ ] **Docker containers + docker-compose** for easy deployment
- [ ] **Mobile-responsive fixes** (current design is desktop-first)

### 10.3 Long Term (3-12 Months)

- [ ] **Bigger model / more classes**: extend to detect more pathologies (Barrett's, angiodysplasia, etc.)
- [ ] **Video segmentation**: identify the exact timestamps where pathology appears
- [ ] **Multi-language support** in the UI
- [ ] **Patient/case management system**: link predictions to patient records
- [ ] **Integration with DICOM viewers** for hospital use
- [ ] **Regulatory compliance**: HIPAA, GDPR, CE marking for medical devices
- [ ] **Clinical validation study** with a partner hospital
- [ ] **Paid tiers**: free for students, paid for clinics
- [ ] **On-premise deployment option** for hospitals that can't send data to the cloud

### 10.4 Scalability Considerations

Current single-instance setup handles maybe ~20 concurrent requests before slowing down. To scale:

- **Horizontal scaling**: Put the backend behind a load balancer, run N replicas (each loads its own model copy)
- **GPU inference server**: Use TensorFlow Serving or Triton for batched GPU inference
- **Caching**: Cache predictions by image hash (the same image → same answer)
- **CDN for frontend**: Serve static assets from CloudFront / Cloudflare
- **Async job queue**: For video processing, use Celery + Redis so the API returns immediately and results come via webhook or polling

---

## 11. Developer Notes & Best Practices

### 11.1 Coding Standards

#### Python
- Follow **PEP 8** (Python's style guide)
- Use **type hints** on public functions (e.g. `def predict(self, image_bytes: bytes) -> Dict:`)
- **Docstrings** on all modules, classes, and non-trivial functions — use triple quotes
- Split responsibilities: `model_service.py` handles inference, `main.py` handles HTTP

#### JavaScript / React
- Use **functional components** with hooks (no class components)
- Keep components **small and focused** — if it's over ~200 lines, split it
- **Props over state** where possible (prefer stateless components)
- Use **Tailwind utility classes** — avoid writing custom CSS unless necessary
- **Single source of truth for API URL** — everything goes through `api/client.js`

### 11.2 Important Assumptions

1. **The model was trained on Kvasir v2** — performance may degrade on visually very different datasets (different scope, lighting, populations)
2. **Input images are assumed to be endoscopy images** — if someone uploads a cat photo, the model will confidently classify it as one of the 4 classes (garbage in, garbage out)
3. **The backend runs with the model in memory** — cold-start is slow (~5-10 seconds)
4. **Python 3.11 is a hard requirement** for TensorFlow 2.18 compatibility
5. **Backend must be started from the project root** for relative paths (`saved_models/best_model.keras`) to resolve

### 11.3 Common Pitfalls

#### Pitfall 1: Wrong Python Version in venv
Earlier in the project, a venv was accidentally created with Python 3.14 and packages installed — then the user tried to use it with Python 3.11 tools. The fix is always: **recreate the venv from scratch with the correct Python version**.

#### Pitfall 2: Committing the Dataset
The entire Kvasir dataset (~1 GB) once ended up committed to Git, making pushes extremely slow. Always verify `.gitignore` includes:
```
data/
dataset/
saved_models/*.keras
saved_models/*.csv
node_modules/
tf-env/
```

#### Pitfall 3: OneDrive + Git
Storing Git repos inside OneDrive causes slow pushes and occasional conflicts. Ideally, keep the repo in a non-synced folder.

#### Pitfall 4: Mixing Integer and One-Hot Labels
The `F1Score` metric requires one-hot encoded labels. If you switch `label_mode="int"`, you must use `sparse_categorical_crossentropy` and drop the F1Score metric (or wrap it). We standardized on `categorical` throughout.

#### Pitfall 5: Forgetting `preprocess_input`
ResNet50 expects pixel values normalized a specific way. The `preprocess_input` layer is **already baked into the model graph** in `train.py`. Never preprocess images twice before sending them through — once is enough.

#### Pitfall 6: Using the Wrong Model File
There are two model files:
- `best_model.keras` — best val_accuracy checkpoint (**use this for production**)
- `gi_classifier_final.keras` — final state after all fine-tuning (often slightly worse)

The backend defaults to `best_model.keras` — don't change it without reason.

#### Pitfall 7: Running Training with `shuffle=True` on Validation
Always set `shuffle=False` on validation and test sets — otherwise results are inconsistent between epochs.

### 11.4 Testing Philosophy

Currently there are no automated tests. For future additions:
- **Backend**: Use `pytest` + `httpx.AsyncClient` to test each endpoint
- **Model service**: Test `predict()` with a sample image, assert output keys exist
- **Frontend**: Use `vitest` + React Testing Library

### 11.5 Security Notes

- Never commit `.env` files — they are gitignored
- Model files are gitignored to avoid large repo sizes
- Uploaded files are handled in memory or temp dirs, never persisted in the current version
- Backend validates MIME types before processing
- For production: add rate limiting (e.g. `slowapi`) to prevent abuse

### 11.6 Useful Git Commands

```bash
# Check what will be committed
git status

# See the diff
git diff

# See the commit history visually
git log --oneline --graph

# Undo the last unpushed commit (keep files)
git reset --soft HEAD~1

# Force-push after a history rewrite (careful!)
git push --force
```

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **Transfer learning** | Using a model pretrained on one task (ImageNet) as a starting point for a different task |
| **ResNet50** | A 50-layer deep convolutional neural network with "residual connections" that solve the vanishing gradient problem in deep networks |
| **Fine-tuning** | Unfreezing some pretrained layers and training them with a small learning rate on the new task |
| **Transfer learning head** | The small stack of layers added on top of the pretrained base — only this is trained in Phase 1 |
| **Softmax** | Activation function that converts raw scores into probabilities that sum to 1 |
| **Categorical cross-entropy** | Loss function for multi-class classification with one-hot labels |
| **One-hot encoding** | Representing class label 2 (of 4) as `[0, 0, 1, 0]` |
| **Batch size** | How many images are processed together in one forward/backward pass |
| **Epoch** | One full pass through the training dataset |
| **Learning rate** | How big a step the optimizer takes when updating weights |
| **Data augmentation** | Random transformations (flip, rotate, zoom) applied only during training to increase effective dataset size |
| **Overfitting** | When a model memorizes training data but performs poorly on unseen data |
| **Class imbalance** | When some classes have many more examples than others, biasing the model |
| **Class weights** | A way to penalize mistakes on rare classes more heavily to counter imbalance |
| **Confusion matrix** | A table showing actual vs predicted counts for each class pair — reveals which classes get confused |
| **Precision** | Of all the items predicted as class X, how many were actually class X? (1 − false positive rate) |
| **Recall** | Of all actual class X items, how many did the model predict as class X? (1 − false negative rate) |
| **F1-score** | Harmonic mean of precision and recall — a single number balancing both |
| **FastAPI** | A modern Python web framework that uses type hints for automatic validation and documentation |
| **Uvicorn** | An ASGI web server used to run FastAPI apps |
| **ASGI** | Asynchronous Server Gateway Interface — modern Python web server standard, replaces WSGI |
| **Vite** | A fast frontend build tool that replaces older bundlers like webpack |
| **Tailwind CSS** | A CSS framework where you compose styles from utility classes in HTML |
| **JSX** | JavaScript + HTML syntax used in React components |
| **CORS** | Cross-Origin Resource Sharing — browser security rule that blocks requests between different domains unless the server allows them |
| **Multipart form data** | The MIME type used by browsers to upload files |
| **Pydantic** | Python library for data validation used by FastAPI |
| **Endpoint** | A URL + HTTP method combination exposed by a web API |
| **Inference** | Running a trained model on new data to get predictions |
| **Preprocess** | Transforming raw input (an image) into the exact format the model expects |
| **Dominant class** | The most frequently predicted class across many samples (used for video analysis) |

---

## Document End

**Project:** GastroLens-AI
**Repository:** https://github.com/wani21/GastroLens-AI
**Version:** 1.0.0 (as of current commit)
**Document Last Updated:** 2026-04-23

If anything in this document is unclear or outdated, please update it before moving on. This is the single source of truth for the project — keep it that way.
