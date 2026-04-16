"""
Model Service
=============
Handles model loading, image preprocessing, and inference.
Designed to be reusable for both image and video prediction endpoints.
"""

import io
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

# =========================================================================
# Configuration
# =========================================================================
# Paths are relative to the project root. Adjust via environment variable
# MODEL_PATH if needed.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "saved_models", "best_model.keras"),
)
LABELS_PATH = os.environ.get(
    "LABELS_PATH",
    os.path.join(BASE_DIR, "saved_models", "class_labels.json"),
)

IMG_SIZE = (224, 224)  # Must match training input size


class ModelService:
    """Singleton wrapper around the Keras model for efficient inference."""

    def __init__(self, model_path: str = MODEL_PATH, labels_path: str = LABELS_PATH):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_labels: Dict[int, str] = {}

    def load(self) -> None:
        """Load model and labels into memory. Called once at server startup."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Run training first or set MODEL_PATH env variable."
            )

        print(f"[ModelService] Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"[ModelService] Model loaded successfully.")

        # Load class label mapping — JSON stores keys as strings
        with open(self.labels_path, "r") as f:
            raw = json.load(f)
        self.class_labels = {int(k): v for k, v in raw.items()}
        print(f"[ModelService] Labels loaded: {self.class_labels}")

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert raw image bytes -> (1, 224, 224, 3) numpy array ready for
        the model. Handles any input format/size/channel count.
        """
        # Open with PIL, force 3-channel RGB (handles grayscale, RGBA, etc.)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        # NOTE: preprocess_input is already baked into the model graph
        # (see train.py) so we pass raw 0-255 pixel values here.
        return np.expand_dims(arr, axis=0)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single video frame (already a numpy array from OpenCV).
        Expects BGR frame; converts to RGB.
        """
        # OpenCV reads frames as BGR — convert to RGB for consistency
        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def predict(self, image_bytes: bytes) -> Dict:
        """
        Predict on a single image.
        Returns: dict with top prediction + all class probabilities.
        """
        x = self.preprocess_image(image_bytes)
        probs = self.model.predict(x, verbose=0)[0]  # shape: (4,)
        return self._format_result(probs)

    def predict_batch(self, images_bytes: List[bytes]) -> List[Dict]:
        """
        Predict on multiple images in a single forward pass.
        More efficient than calling predict() in a loop.
        """
        # Stack all preprocessed images into one batch
        batch = np.vstack([self.preprocess_image(b) for b in images_bytes])
        probs_batch = self.model.predict(batch, verbose=0)  # shape: (N, 4)
        return [self._format_result(p) for p in probs_batch]

    def predict_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Predict on a list of video frames (numpy arrays).
        Used by the video endpoint.
        """
        batch = np.vstack([self.preprocess_frame(f) for f in frames])
        probs_batch = self.model.predict(batch, verbose=0)
        return [self._format_result(p) for p in probs_batch]

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _format_result(self, probs: np.ndarray) -> Dict:
        """Convert raw probability vector -> structured response."""
        top_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.class_labels[top_idx],
            "predicted_class_index": top_idx,
            "confidence": float(probs[top_idx]),
            "probabilities": {
                self.class_labels[i]: float(p) for i, p in enumerate(probs)
            },
        }


# Single global instance — loaded once at startup
model_service = ModelService()
