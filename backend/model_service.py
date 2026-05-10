import io
import json
import os
from typing import List, Dict

import numpy as np
import tensorflow as tf
from PIL import Image

# =========================================================================
# Configuration
# =========================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "saved_models", "best_model.keras"),
)

LABELS_PATH = os.environ.get(
    "LABELS_PATH",
    os.path.join(BASE_DIR, "saved_models", "class_labels.json"),
)

IMG_SIZE = (224, 224)

# =========================================================================
# LMCL LOSS (Needed for model deserialization)
# =========================================================================

def large_margin_cosine_loss(margin=0.35, scale=64):

    def lmcl_loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        cosine_logits = tf.nn.l2_normalize(y_pred, axis=1)
        cosine_logits = scale * cosine_logits

        margin_cosine_logits = cosine_logits - (margin * y_true)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=margin_cosine_logits
        )

        return tf.reduce_mean(loss)

    lmcl_loss.__name__ = f"lmcl_m{margin}_s{scale}"

    return lmcl_loss

# =========================================================================
# Model Service
# =========================================================================

class ModelService:

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        labels_path: str = LABELS_PATH
    ):

        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_labels: Dict[int, str] = {}

    # ---------------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------------

    def load(self) -> None:

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}"
            )

        print(f"[ModelService] Loading model from {self.model_path}...")

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                "lmcl_m0.35_s64": large_margin_cosine_loss(
                    margin=0.35,
                    scale=64
                )
            },
            compile=False
        )

        print("[ModelService] Model loaded successfully.")

        with open(self.labels_path, "r") as f:
            raw = json.load(f)

        self.class_labels = {
            int(k): v for k, v in raw.items()
        }

        print(f"[ModelService] Labels loaded: {self.class_labels}")

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)

        arr = np.array(img, dtype=np.float32)

        return np.expand_dims(arr, axis=0)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:

        import cv2

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame_rgb).resize(IMG_SIZE)

        arr = np.array(img, dtype=np.float32)

        return np.expand_dims(arr, axis=0)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def predict(self, image_bytes: bytes) -> Dict:

        x = self.preprocess_image(image_bytes)

        probs = self.model.predict(x, verbose=0)[0]

        return self._format_result(probs)

    def predict_batch(self, images_bytes: List[bytes]) -> List[Dict]:

        batch = np.vstack([
            self.preprocess_image(b)
            for b in images_bytes
        ])

        probs_batch = self.model.predict(batch, verbose=0)

        return [self._format_result(p) for p in probs_batch]

    def predict_frames(self, frames: List[np.ndarray]) -> List[Dict]:

        batch = np.vstack([
            self.preprocess_frame(f)
            for f in frames
        ])

        probs_batch = self.model.predict(batch, verbose=0)

        return [self._format_result(p) for p in probs_batch]

    # ---------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------

    def _format_result(self, probs: np.ndarray) -> Dict:

        top_idx = int(np.argmax(probs))

        return {
            "predicted_class": self.class_labels[top_idx],
            "predicted_class_index": top_idx,
            "confidence": float(probs[top_idx]),
            "probabilities": {
                self.class_labels[i]: float(p)
                for i, p in enumerate(probs)
            },
        }

# =========================================================================
# Global singleton
# =========================================================================

model_service = ModelService()