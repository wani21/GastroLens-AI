"""
Gastrointestinal Disease Classification using Transfer Learning (ResNet50)
===========================================================================
This script trains a model to classify GI endoscopy images into 4 classes:
  normal, polyp, ulcer, esophagitis

Strategy:
  Phase 1 - Train only the custom classification head (base frozen)
  Phase 2 - Fine-tune the top layers of ResNet50 along with the head
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
# Why: Centralizing all settings makes experiments easy to tweak.

DATA_DIR = "data"
IMG_SIZE = (224, 224)       # ResNet50 expects 224x224 input
BATCH_SIZE = 32             # Fits comfortably in most GPUs (reduce to 16 if OOM)
NUM_CLASSES = 4
SEED = 42

# Phase 1: frozen base, train head only
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3

# Phase 2: fine-tune top layers of ResNet50
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5           # Much smaller LR to avoid destroying pretrained weights

# ===========================================================================
# 2. LOAD DATASETS
# ===========================================================================
# Why: image_dataset_from_directory is the simplest way to load folder-based
# image datasets. It handles labeling, batching, and shuffling automatically.

print("Loading datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    label_mode="categorical",   # One-hot labels e.g. [0,0,1,0] — required by F1Score
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,          # No need to shuffle validation data
    label_mode="categorical",
)

test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,
    label_mode="categorical",
)

# Print the class names — these are sorted alphabetically by folder name
class_names = train_ds.class_names
print(f"Classes: {class_names}")
# Expected: ['esophagitis', 'normal', 'polyp', 'ulcer']

# ===========================================================================
# 3. SAVE CLASS LABEL MAPPING
# ===========================================================================
# Why: When you deploy this model in a backend, the model outputs a number
# (0, 1, 2, 3) — you need this mapping to convert it back to a human-readable
# class name. Saving it now guarantees consistency between training and serving.

label_map = {i: name for i, name in enumerate(class_names)}
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/class_labels.json", "w") as f:
    json.dump(label_map, f, indent=2)
print(f"Class label mapping saved: {label_map}")

# ===========================================================================
# 4. COMPUTE CLASS WEIGHTS
# ===========================================================================
# Why: The dataset is imbalanced (normal/polyp ~3x more than esophagitis/ulcer).
# Class weights tell the loss function to penalize errors on minority classes
# more heavily, so the model doesn't just predict the majority class.

print("Computing class weights...")

# Extract all labels from the training set
# Labels are one-hot encoded, so argmax converts [0,0,1,0] -> 2
train_labels = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in train_ds])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights = dict(enumerate(weights))
print(f"Class weights: {class_weights}")

# ===========================================================================
# 5. DATA AUGMENTATION
# ===========================================================================
# Why: Augmentation artificially increases training data variety by applying
# random transformations (flips, rotations, zoom). This reduces overfitting
# and helps the model generalize to unseen images.
#
# These layers are ONLY active during training — they pass through unchanged
# during inference/evaluation.

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),        # Up to ±20% of full rotation
    layers.RandomZoom(0.2),            # Up to ±20% zoom
    layers.RandomContrast(0.2),        # Slight contrast variation
], name="data_augmentation")

# ===========================================================================
# 6. PERFORMANCE OPTIMIZATION
# ===========================================================================
# Why: .cache() keeps images in memory after first load (faster epochs).
# .prefetch() loads next batch while GPU processes current one.
# This is standard practice — always do this.

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ===========================================================================
# 7. BUILD THE MODEL
# ===========================================================================
# Why: Transfer learning lets us reuse features (edges, textures, shapes)
# that ResNet50 learned from ImageNet. We only need to train a small
# classification head on top, which is much faster and needs less data
# than training from scratch.

print("Building model...")

# Load ResNet50 WITHOUT its classification head (include_top=False)
base_model = ResNet50(
    weights="imagenet",
    include_top=False,              # Remove the 1000-class ImageNet head
    input_shape=(224, 224, 3),
)

# Freeze all base model layers — we train only our custom head first
base_model.trainable = False

# Build the full model
inputs = keras.Input(shape=(224, 224, 3))

# Step 1: Data augmentation (only active during training)
x = data_augmentation(inputs)

# Step 2: Preprocess inputs for ResNet50 (scales pixels to expected range)
# Why: ResNet50 was trained with specific preprocessing — we must match it
x = keras.applications.resnet50.preprocess_input(x)

# Step 3: Pass through frozen ResNet50 base
# training=False ensures BatchNorm layers stay in inference mode
x = base_model(x, training=False)

# Step 4: Custom classification head
x = layers.GlobalAveragePooling2D()(x)  # Convert feature maps to vector
x = layers.Dropout(0.3)(x)              # Regularization to reduce overfitting
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.summary()

# ===========================================================================
# 8. COMPILE THE MODEL (Phase 1)
# ===========================================================================
# Why:
# - categorical_crossentropy: Standard loss for multi-class with one-hot labels
# - Adam optimizer: Good default, adaptive learning rate
# - Metrics: Accuracy alone can be misleading with imbalanced data,
#   so we also track precision and recall

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(name="f1_score", average="weighted"),
    ],
)

# ===========================================================================
# 9. CALLBACKS
# ===========================================================================
# Why:
# - EarlyStopping: Stops training if validation loss stops improving (prevents
#   overfitting and wasting compute)
# - ModelCheckpoint: Saves the best model based on validation accuracy
# - ReduceLROnPlateau: Lowers learning rate when progress stalls

os.makedirs("saved_models", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,                 # Wait 5 epochs before stopping
        restore_best_weights=True,  # Go back to best epoch's weights
        verbose=1,
    ),
    ModelCheckpoint(
        filepath="saved_models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,                 # Halve the learning rate
        patience=3,                 # Wait 3 epochs before reducing
        min_lr=1e-7,
        verbose=1,
    ),
]

# ===========================================================================
# 10. PHASE 1: TRAIN THE CLASSIFICATION HEAD
# ===========================================================================
# Why: With the base frozen, we only train the small head (~500K params).
# This is fast and prevents the randomly initialized head from sending
# wild gradients into the pretrained base.

print("\n" + "=" * 60)
print("PHASE 1: Training classification head (base frozen)")
print("=" * 60)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ===========================================================================
# 11. PHASE 2: FINE-TUNING
# ===========================================================================
# Why: Now that the head is trained, we unfreeze the top layers of ResNet50
# and train them together with a very small learning rate. This adapts the
# high-level features to our specific medical imaging task.
#
# We only unfreeze the LAST ~30 layers (conv5 block). Unfreezing everything
# risks destroying the useful low-level features and overfitting.

print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning top layers of ResNet50")
print("=" * 60)

# Unfreeze the base model
base_model.trainable = True

# Freeze all layers EXCEPT the last ~30 (conv5_block)
# ResNet50 has 175 layers total
FINE_TUNE_AT = 143  # Unfreeze from layer 143 onwards
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Count trainable parameters
trainable_count = sum(
    tf.keras.backend.count_params(w) for w in model.trainable_weights
)
print(f"Trainable parameters after unfreezing: {trainable_count:,}")

# Recompile with a much smaller learning rate
# Why: Large LR would destroy the pretrained weights. 1e-5 is a safe choice.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(name="f1_score", average="weighted"),
    ],
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ===========================================================================
# 12. EVALUATE ON TEST SET
# ===========================================================================
# Why: The test set was never seen during training. This gives an unbiased
# estimate of how the model will perform on new, unseen data.

print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

results = model.evaluate(test_ds)
print(f"\nTest Loss:      {results[0]:.4f}")
print(f"Test Accuracy:  {results[1]:.4f}")
print(f"Test Precision: {results[2]:.4f}")
print(f"Test Recall:    {results[3]:.4f}")
print(f"Test F1-Score:  {results[4]:.4f}")

# ===========================================================================
# 13. CONFUSION MATRIX & CLASSIFICATION REPORT
# ===========================================================================
# Why: The confusion matrix shows EXACTLY where the model gets confused —
# e.g., does it mix up ulcer with esophagitis? This is essential for:
#   - Presentations / PPTs
#   - Viva / oral exams
#   - Understanding model weaknesses per class
#
# Classification report gives per-class precision, recall, and F1 — much
# more informative than a single overall number.

print("\n" + "=" * 60)
print("PER-CLASS METRICS & CONFUSION MATRIX")
print("=" * 60)

# Get predictions on the full test set
test_labels = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
test_predictions = np.concatenate([
    np.argmax(model.predict(images, verbose=0), axis=1)
    for images, _ in test_ds
])

# Classification report: per-class precision, recall, F1
print("\nClassification Report:")
print(classification_report(test_labels, test_predictions,
                            target_names=class_names))

# Confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(f"{'':>15}", end="")
for name in class_names:
    print(f"{name:>14}", end="")
print()
for i, row in enumerate(cm):
    print(f"{class_names[i]:>15}", end="")
    for val in row:
        print(f"{val:>14}", end="")
    print()

# Save confusion matrix as CSV for use in PPT/reports
np.savetxt("saved_models/confusion_matrix.csv", cm, delimiter=",", fmt="%d",
           header=",".join(class_names), comments="")
print("\nConfusion matrix saved to saved_models/confusion_matrix.csv")

# ===========================================================================
# 14. SAVE THE FINAL MODEL
# ===========================================================================
# Why: Save in .keras format (recommended by TensorFlow). This saves
# architecture + weights + optimizer state, so you can resume training
# or load it for inference in your backend.

model.save("saved_models/gi_classifier_final.keras")
print("\nModel saved to saved_models/gi_classifier_final.keras")
print("Best checkpoint at saved_models/best_model.keras")
print("Class labels at saved_models/class_labels.json")
print("\nDone!")
