"""
U-Net Segmentation Model for GI Lesion Localization
=====================================================
Syllabus Unit 5: Encoder-Decoder Architecture

The U-Net architecture:
  Encoder: ResNet50 (shared with classifier) — extracts features at 4 scales
  Decoder: 4 upsampling blocks — reconstructs spatial resolution
  Skip connections: connect encoder and decoder at each scale (the key U-Net idea)
  Output: binary mask (224×224×1) where 1=lesion, 0=background

This is the encoder-decoder architecture your faculty specifically requested.

Unit 1 (Neural networks, loss functions) → LMCL loss in train.py [DONE]
Unit 2 (CNN, attention, YOLO)            → SE-ResNet-H in train.py [DONE]
Unit 3 (Quantum CNN)                     → Skipped (hardware not available)
Unit 4 (RNN, LSTM, sequence models)      → LSTM video analysis [TASK 2]
Unit 5 (Encoder-decoder, attention)      → U-Net segmentation [TASK 3]
Unit 6 (XAI, SHAP, Grad-CAM)            → /explain + /shap endpoints [TASKS 1 & 4]
"""

import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# NOTE: This U-Net is initialized with ImageNet pretrained encoder weights
# but untrained decoder weights. To get accurate lesion masks, train it on
# a segmentation dataset (e.g. Kvasir-SEG which has pixel-level polyp masks).
# For this demo, it demonstrates the encoder-decoder architecture concept.
# The encoder (ResNet50) shares the same pretrained features as the classifier.

def build_unet_segmentation_model():
    input_shape = (224, 224, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder (ResNet50)
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Get skip connections
    s1 = base_model.get_layer("conv1_relu").output # (112, 112, 64)
    s2 = base_model.get_layer("conv2_block3_out").output # (56, 56, 256)
    s3 = base_model.get_layer("conv3_block4_out").output # (28, 28, 512)
    s4 = base_model.get_layer("conv4_block6_out").output # (14, 14, 1024)
    bottleneck = base_model.get_layer("conv5_block3_out").output # (7, 7, 2048)

    # Decoder
    # Block 1
    u1 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u1 = tf.keras.layers.concatenate([u1, s4])
    u1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u1)
    u1 = tf.keras.layers.BatchNormalization()(u1)

    # Block 2
    u2 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u1)
    u2 = tf.keras.layers.concatenate([u2, s3])
    u2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u2)
    u2 = tf.keras.layers.BatchNormalization()(u2)

    # Block 3
    u3 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u2)
    u3 = tf.keras.layers.concatenate([u3, s2])
    u3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    u3 = tf.keras.layers.BatchNormalization()(u3)

    # Block 4
    u4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u3)
    u4 = tf.keras.layers.concatenate([u4, s1])
    u4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    u4 = tf.keras.layers.BatchNormalization()(u4)

    # Final upsample to 224x224
    u5 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u4)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u5)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def generate_segmentation_mask(image_bytes: bytes, seg_model: tf.keras.Model):
    # Preprocess
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    x = np.expand_dims(arr, axis=0)

    # Predict mask
    pred_mask = seg_model.predict(x, verbose=0)
    mask = np.squeeze(pred_mask)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Calculate coverage
    coverage_pct = float(np.sum(binary_mask > 0) / (224 * 224) * 100)
    
    # Encode binary mask
    mask_img = Image.fromarray(binary_mask)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    mask_base64_str = f"data:image/png;base64,{mask_base64}"
    
    # Overlay
    orig_img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    red_mask = np.zeros_like(orig_img_cv)
    red_mask[:, :, 2] = binary_mask # Set red channel
    
    overlay = cv2.addWeighted(orig_img_cv, 1.0, red_mask, 0.5, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_img = Image.fromarray(overlay_rgb)
    buf2 = io.BytesIO()
    overlay_img.save(buf2, format="PNG")
    overlay_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    overlay_base64_str = f"data:image/png;base64,{overlay_base64}"
    
    return {
        "mask_base64": mask_base64_str,
        "overlay_base64": overlay_base64_str,
        "lesion_coverage_pct": round(coverage_pct, 2)
    }

# Singleton
seg_model = build_unet_segmentation_model()
