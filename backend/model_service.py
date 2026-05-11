import io
import json
import os
import base64
from typing import List, Dict

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        self._gradcam_model = None
        self._embedding_model = None

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

        probs_softmax = tf.nn.softmax(probs).numpy()
        top_idx = int(np.argmax(probs_softmax))

        return {
            "predicted_class": self.class_labels[top_idx],
            "predicted_class_index": top_idx,
            "confidence": float(probs_softmax[top_idx]),
            "probabilities": {
                self.class_labels[i]: float(p)
                for i, p in enumerate(probs_softmax)
            },
        }

    # ---------------------------------------------------------------------
    # Grad-CAM (Unit 6)
    # ---------------------------------------------------------------------

    def generate_gradcam(self, image_bytes: bytes, class_index: int = None) -> Dict:
        x = self.preprocess_image(image_bytes)
        
        if self._gradcam_model is None:
            target_layer_name = None
            for name in ["se_combine_relu", "conv5_block3_out", "conv5_block3_3_bn"]:
                try:
                    self.model.get_layer(name)
                    target_layer_name = name
                    break
                except ValueError:
                    continue
                    
            if target_layer_name is None:
                layer = [l for l in self.model.layers if len(l.output_shape) == 4][-1]
                target_layer_name = layer.name
                
            last_conv_layer = self.model.get_layer(target_layer_name)
            self._gradcam_model = tf.keras.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self._gradcam_model(x)
            
            if class_index is None:
                class_index = tf.argmax(predictions[0])
                
            loss = predictions[:, class_index]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap = np.uint8(255 * heatmap)
        
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize(IMG_SIZE)
        jet_heatmap = np.array(jet_heatmap)
        
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_arr = np.array(img)
        
        superimposed_img = jet_heatmap * 0.4 + img_arr
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        
        buf = io.BytesIO()
        superimposed_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        res = self._format_result(predictions[0].numpy())
        res["class_index"] = int(class_index)
        res["heatmap_base64"] = f"data:image/png;base64,{encoded}"
        
        return res

    # ---------------------------------------------------------------------
    # SHAP (Unit 6)
    # ---------------------------------------------------------------------

    def generate_shap_explanation(self, image_bytes: bytes) -> Dict:
        x = self.preprocess_image(image_bytes)
        
        background = np.repeat(x, 10, axis=0) + np.random.normal(0, 0.1, (10, 224, 224, 3))
        
        explainer = shap.GradientExplainer(self.model, background)
        shap_values = explainer.shap_values(x)
        
        probs = self.model.predict(x, verbose=0)[0]
        res = self._format_result(probs)
        pred_idx = res["predicted_class_index"]
        
        if isinstance(shap_values, list):
            sv = shap_values[pred_idx][0]
        else:
            if len(shap_values.shape) == 5:
                # Shape: (batch, H, W, C, classes)
                sv = shap_values[0, ..., pred_idx]
            elif len(shap_values.shape) == 4:
                # Shape: (batch, H, W, C)
                sv = shap_values[0]
            else:
                sv = shap_values[0]

        sv_sum = np.sum(sv, axis=-1)
        
        max_val = np.max(np.abs(sv_sum))
        if max_val > 0:
            sv_norm = sv_sum / max_val
        else:
            sv_norm = sv_sum
            
        plt.figure(figsize=(4, 4))
        plt.imshow(sv_norm, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {
            "shap_image_base64": f"data:image/png;base64,{encoded}",
            "predicted_class": res["predicted_class"],
            "explanation": f"Red regions support predicting {res['predicted_class']}. Blue regions argue against it."
        }

    # ---------------------------------------------------------------------
    # LSTM Temporal (Unit 4)
    # ---------------------------------------------------------------------

    def get_frame_embeddings(self, frames_list: List[np.ndarray]) -> np.ndarray:
        if self._embedding_model is None:
            target_layer = None
            for name in ["gap", "global_average_pooling2d", "fc_256"]:
                try:
                    target_layer = self.model.get_layer(name)
                    break
                except ValueError:
                    continue
            
            if target_layer is None:
                layers = [l for l in self.model.layers if isinstance(l, (tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.Flatten))]
                if layers:
                    target_layer = layers[-1]
                else:
                    target_layer = self.model.layers[-2]
            
            self._embedding_model = tf.keras.Model(self.model.input, target_layer.output)
            
        batch = np.vstack(frames_list)
        return self._embedding_model.predict(batch, verbose=0)

    def predict_video_lstm(self, video_path: str, frame_sample_rate: int = 10) -> Dict:
        from collections import Counter
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_indices = []
        raw_frames = []
        idx = 0
        MAX_VIDEO_FRAMES = 60
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_sample_rate == 0:
                raw_frames.append(frame)
                frames.append(self.preprocess_frame(frame))
                frame_indices.append(idx)
                if len(frames) >= MAX_VIDEO_FRAMES:
                    break
            idx += 1
        cap.release()

        if not frames:
            raise ValueError("No frames could be extracted")

        frame_results = self.predict_frames(raw_frames)
        for i, r in enumerate(frame_results):
            r["frame_index"] = frame_indices[i]
            
        predicted_classes = [r["predicted_class"] for r in frame_results]
        class_counts = Counter(predicted_classes)
        dominant_class, dominant_count = class_counts.most_common(1)[0]
        
        embeddings = self.get_frame_embeddings(frames)
        num_frames, embedding_dim = embeddings.shape
        
        # LSTM weights are randomly initialized for architecture demonstration. 
        # In production, train this on labeled video sequences.
        lstm_input = tf.keras.layers.Input(shape=(num_frames, embedding_dim))
        x = tf.keras.layers.LSTM(128, return_sequences=False)(lstm_input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        lstm_output = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        lstm_model = tf.keras.Model(lstm_input, lstm_output)
        
        seq_input = np.expand_dims(embeddings, axis=0)
        lstm_probs = lstm_model.predict(seq_input, verbose=0)[0]
        
        top_idx = int(np.argmax(lstm_probs))
        lstm_prediction = {
            "predicted_class": self.class_labels[top_idx],
            "confidence": float(lstm_probs[top_idx]),
            "probabilities": {
                self.class_labels[i]: float(p) for i, p in enumerate(lstm_probs)
            }
        }
        
        return {
            "total_frames": total_frames,
            "sampled_frames": len(frames),
            "frame_sample_rate": frame_sample_rate,
            "analysis_method": "lstm_temporal",
            "lstm_prediction": lstm_prediction,
            "frame_embeddings_shape": [num_frames, embedding_dim],
            "dominant_class": dominant_class,
            "dominant_class_ratio": round(dominant_count / len(frames), 3),
            "class_distribution": dict(class_counts),
            "frame_results": frame_results
        }

# =========================================================================
# Global singleton
# =========================================================================

model_service = ModelService()