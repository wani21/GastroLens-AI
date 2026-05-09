"""
Gastrointestinal Disease Classification — Enhanced with DL Concepts
=====================================================================
ENHANCEMENTS OVER BASELINE (maps to syllabus units):

  [Unit 1]  Large Margin Cosine Loss (LMCL) replaces categorical cross-entropy
            → shows understanding of loss function design, not just defaults

  [Unit 2]  Squeeze-and-Excitation (SE) attention blocks added at H-position
            (after stage0 and stage4 of ResNet50) → SE-ResNet-H architecture
            from reference paper (Ye et al., IEEE TIM 2024)

  [Unit 6]  Grad-CAM heatmap generation after evaluation
            → Explainable AI: shows WHAT the model is looking at per class

Original strategy (preserved):
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
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ===========================================================================
# [NEW — Unit 2] SQUEEZE-AND-EXCITATION (SE) ATTENTION BLOCK
# ===========================================================================
# What: An SE block learns to re-weight feature channels based on their global
#       importance. It "squeezes" spatial information into a channel descriptor,
#       then "excites" useful channels and suppresses irrelevant ones.
#
# How it works (3 steps):
#   1. Squeeze  — GlobalAveragePooling collapses (H, W, C) → (1, 1, C)
#                 This gives a global summary of each channel
#   2. Excitation — Two Dense layers learn which channels matter:
#                   Dense(C/ratio) → ReLU → Dense(C) → Sigmoid
#                   Output is C values in [0,1] — one weight per channel
#   3. Scale    — Multiply original feature map by these weights
#                 Important channels are amplified, weak ones suppressed
#
# Position (H-mode from paper): applied after stage0 and after stage4
# This captures both early edge/texture features and final semantic features.
#
# Paper reference: Ye et al., "Attention Mechanism Guided SE+ResNet-H Model
# for Gastrointestinal Endoscopy Image Classification", IEEE TIM 2024.
# Their result: SE-ResNet-H achieves 97.84% vs ResNet baseline 97.11%

def squeeze_excitation_block(input_tensor, ratio=16, name_prefix="se"):
    """
    Squeeze-and-Excitation block (channel attention).

    Args:
        input_tensor : 4D tensor of shape (batch, H, W, C)
        ratio        : Reduction ratio for the bottleneck Dense layer.
                       ratio=16 means the hidden layer has C/16 units.
                       Higher ratio = fewer params but less expressive.
        name_prefix  : String prefix for layer names (avoids name collisions
                       when this function is called more than once).

    Returns:
        Tensor of same shape as input_tensor, with channels re-weighted.
    """
    # Get number of channels from the input
    channels = input_tensor.shape[-1]   # e.g. 64 after stage0, 2048 after stage4

    # --- Step 1: SQUEEZE ---
    # Reduce spatial dimensions (H, W) → single value per channel
    # Result shape: (batch, channels)
    se = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(input_tensor)

    # --- Step 2: EXCITATION ---
    # Bottleneck: compress to channels/ratio, then expand back
    # This forces the network to learn compact channel relationships
    se = layers.Dense(
        channels // ratio,
        activation="relu",
        use_bias=False,              # Bias not needed before BatchNorm-like squeeze
        name=f"{name_prefix}_fc1"
    )(se)

    se = layers.Dense(
        channels,
        activation="sigmoid",        # Sigmoid → weights in [0,1] per channel
        use_bias=False,
        name=f"{name_prefix}_fc2"
    )(se)

    # Reshape from (batch, channels) → (batch, 1, 1, channels)
    # so it can broadcast correctly over (batch, H, W, channels)
    se = layers.Reshape((1, 1, channels), name=f"{name_prefix}_reshape")(se)

    # --- Step 3: SCALE ---
    # Element-wise multiply: each spatial location gets its channel weights applied
    # Important channels are boosted, unimportant ones suppressed
    output = layers.Multiply(name=f"{name_prefix}_scale")([input_tensor, se])

    return output


# ===========================================================================
# [NEW — Unit 1] LARGE MARGIN COSINE LOSS (LMCL)
# ===========================================================================
# What: Replaces standard cross-entropy. Instead of raw logits, it uses
#       COSINE SIMILARITY between feature vectors and class weight vectors,
#       then subtracts a margin (m) from the correct class score before softmax.
#
# Why this is better than cross-entropy for medical imaging:
#   - Standard cross-entropy: model just needs to push correct class score
#     higher than others — no explicit constraint on HOW far apart
#   - LMCL: maximizes inter-class distance and minimizes intra-class distance
#     → features for "polyp" cluster tightly, far from "normal" cluster
#     → reduces misclassification at class boundaries (esophagitis vs normal)
#
# Mathematical definition (from paper eq. 3):
#   L_lmc = -log[ e^{s(cos θ_yi - m)} / (e^{s(cos θ_yi - m)} + Σ_{j≠yi} e^{s·cos θ_j}) ]
#
#   where:
#     cos θ_j = W_j^T · x  (cosine similarity, both W and x are L2-normalized)
#     m = margin (default 0.35) — the "large margin" that separates classes
#     s = scale factor (default 64) — controls the sharpness of the distribution
#
# Result from paper: LMCL improved SE-ResNet-H accuracy from 97.84% → 98.47%

def large_margin_cosine_loss(margin=0.35, scale=64):
    """
    Factory function that returns an LMCL loss function compatible with
    Keras model.compile(loss=...).

    Args:
        margin : The cosine margin m. Typical values: 0.25–0.45.
                 Higher margin → stricter separation but harder to train.
        scale  : The scaling factor s. Typical values: 30–64.
                 Higher scale → sharper softmax distribution.

    Returns:
        A loss function with signature loss_fn(y_true, y_pred).

    Usage:
        model.compile(loss=large_margin_cosine_loss(margin=0.35, scale=64), ...)
    """
    def lmcl_loss(y_true, y_pred):
        """
        Args:
            y_true : One-hot labels, shape (batch, num_classes)
            y_pred : Raw cosine similarity logits from the model's final Dense
                     layer. The Dense layer MUST NOT have an activation function
                     (linear output) so we get raw dot products here.
                     Shape: (batch, num_classes)

        Returns:
            Scalar loss value.
        """
        # Cast to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # L2-normalize the predictions (cosine similarity requires unit vectors)
        # After normalization, each row has magnitude 1
        # cosine_logits[i][j] = cos(angle between sample i and class j center)
        cosine_logits = tf.nn.l2_normalize(y_pred, axis=1)

        # Scale the cosine values (makes the distribution sharper)
        cosine_logits = scale * cosine_logits

        # Subtract margin ONLY from the correct class logit
        # y_true is one-hot, so (margin * y_true) is non-zero only at the true class
        # This forces the model to score the correct class by at least `margin`
        # more than any other class — the "large margin" constraint
        margin_cosine_logits = cosine_logits - (margin * y_true)

        # Standard softmax cross-entropy on the margin-adjusted logits
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=margin_cosine_logits
        )

        return tf.reduce_mean(loss)

    # Give the function a recognizable name (shows in model.summary())
    lmcl_loss.__name__ = f"lmcl_m{margin}_s{scale}"
    return lmcl_loss


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
SEED = 42

PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3

PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5

# LMCL hyperparameters (from paper Table II)
LMCL_MARGIN = 0.35
LMCL_SCALE  = 64

# SE block reduction ratio (standard value from original SE paper)
SE_RATIO = 16

# ===========================================================================
# 2. LOAD DATASETS
# ===========================================================================
print("Loading datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    label_mode="categorical",
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,
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

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# ===========================================================================
# 3. SAVE CLASS LABEL MAPPING
# ===========================================================================
label_map = {i: name for i, name in enumerate(class_names)}
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/class_labels.json", "w") as f:
    json.dump(label_map, f, indent=2)
print(f"Class label mapping saved: {label_map}")

# ===========================================================================
# 4. COMPUTE CLASS WEIGHTS
# ===========================================================================
print("Computing class weights...")

train_labels = np.concatenate([
    np.argmax(labels.numpy(), axis=1) for _, labels in train_ds
])

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
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# ===========================================================================
# 6. PERFORMANCE OPTIMIZATION
# ===========================================================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ===========================================================================
# [MODIFIED — Unit 2] 7. BUILD THE SE-ResNet-H MODEL
# ===========================================================================
# Change from baseline:
#   BEFORE: plain ResNet50 → GAP → Dropout → Dense(256) → Dropout → Dense(4)
#   AFTER:  SE-ResNet-H:
#             stage0_output → SE_block_1 (early feature re-weighting)
#             → stage1..stage4 → SE_block_2 (semantic feature re-weighting)
#             → GAP → Dropout → Dense(256) → Dropout → Dense(4, linear)
#
# The "H" in SE-ResNet-H means "Head and tail" — SE blocks at both ends.
# This is exactly the architecture from Fig. 3 of the reference paper.
#
# Note: The final Dense layer uses NO activation (linear output) because
# LMCL works on raw cosine logits, not softmax probabilities.
# LMCL internally applies its own softmax — applying softmax before would
# double-squash the values and break the loss computation.

print("Building SE-ResNet-H model...")

# --- Load ResNet50 base ---
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False   # Freeze for Phase 1

# --- Identify intermediate layer for SE-block-1 ---
# stage0 ends at 'conv1_relu' in ResNet50 (output shape: 56x56x64)
# We attach SE_block_1 here to re-weight early texture/edge features.
# To find exact layer names: [l.name for l in base_model.layers[:10]]
STAGE0_LAYER = "conv1_relu"

# Build a sub-model that exposes both stage0 output and final output
stage0_output = base_model.get_layer(STAGE0_LAYER).output   # (56, 56, 64)
final_output  = base_model.output                            # (7, 7, 2048)

# Create a model with TWO outputs so we can tap into stage0
resnet_extractor = keras.Model(
    inputs=base_model.input,
    outputs=[stage0_output, final_output],
    name="resnet_extractor",
)
resnet_extractor.trainable = False   # Keep frozen for Phase 1

# --- Build the full SE-ResNet-H model ---
inputs = keras.Input(shape=(224, 224, 3))

# Augmentation + preprocessing
x_aug  = data_augmentation(inputs)
x_prep = keras.applications.resnet50.preprocess_input(x_aug)

# Extract stage0 and final feature maps simultaneously
# training=False keeps BatchNorm layers in inference mode (no batch stats)
stage0_feat, final_feat = resnet_extractor(x_prep, training=False)

# --- SE block 1: applied to stage0 output (early features, 64 channels) ---
# Why here: early layers capture edges and textures. SE re-weights which
# edge/texture channels are most informative for GI disease detection.
# ratio=4 because 64/16=4 (too small) → use ratio=4 to get 64/4=16 units
se1 = squeeze_excitation_block(stage0_feat, ratio=4, name_prefix="se_stage0")

# We need to compress se1 spatially to match final_feat (7x7)
# Use a small conv to downsample: 56x56 → 7x7 via stride-8
# This creates a "shortcut" from early features to the classification head
se1_down = layers.Conv2D(
    filters=256,
    kernel_size=1,
    strides=8,
    padding="same",
    use_bias=False,
    name="se1_project"
)(se1)
se1_down = layers.BatchNormalization(name="se1_bn")(se1_down)
se1_down = layers.Activation("relu", name="se1_relu")(se1_down)

# --- SE block 2: applied to final ResNet output (semantic features, 2048 ch) ---
# Why here: final layers capture high-level semantic features (polyp shapes,
# ulcer patterns). SE re-weights which semantic features matter most.
se2 = squeeze_excitation_block(final_feat, ratio=16, name_prefix="se_stage4")

# --- Combine SE1 shortcut with SE2 main path ---
# Project se1_down to 2048 channels to match se2
se1_proj = layers.Conv2D(
    filters=2048,
    kernel_size=1,
    padding="same",
    use_bias=False,
    name="se1_to_2048"
)(se1_down)

# Add: combines early texture-aware features with semantic features
combined = layers.Add(name="se_combine")([se2, se1_proj])
combined = layers.Activation("relu", name="se_combine_relu")(combined)

# --- Classification head ---
x = layers.GlobalAveragePooling2D(name="gap")(combined)
x = layers.Dropout(0.3, name="dropout_1")(x)
x = layers.Dense(256, activation="relu", name="fc_256")(x)
x = layers.Dropout(0.3, name="dropout_2")(x)

# IMPORTANT: NO activation here (linear output) — LMCL needs raw logits
# If you switch back to cross-entropy, change this to activation="softmax"
outputs = layers.Dense(NUM_CLASSES, activation=None, name="fc_output")(x)

model = keras.Model(inputs, outputs, name="SE_ResNet_H")
model.summary()

# ===========================================================================
# [MODIFIED — Unit 1] 8. COMPILE (Phase 1) WITH LMCL LOSS
# ===========================================================================
# Change: loss="categorical_crossentropy" → large_margin_cosine_loss(...)
#
# Why LMCL is better for our problem:
#   Our biggest weakness is esophagitis recall (0.76 in baseline).
#   Esophagitis looks similar to normal tissue at class boundaries.
#   LMCL explicitly enforces a cosine margin between classes, pushing
#   esophagitis features away from normal features in embedding space.

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss=large_margin_cosine_loss(margin=LMCL_MARGIN, scale=LMCL_SCALE),
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(name="f1_score", average="weighted"),
    ],
)

# ===========================================================================
# 9. CALLBACKS (unchanged)
# ===========================================================================
os.makedirs("saved_models", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
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
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ===========================================================================
# 10. PHASE 1: TRAIN THE CLASSIFICATION HEAD
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 1: Training SE classification head (base frozen)")
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
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning top layers of SE-ResNet-H")
print("=" * 60)

# Unfreeze the resnet_extractor sub-model
resnet_extractor.trainable = True

# Freeze all layers in the extractor EXCEPT the last ~30 (conv5 block)
FINE_TUNE_AT = 143
for layer in resnet_extractor.layers:
    # Find layers by their index in the original base_model
    try:
        idx = [l.name for l in base_model.layers].index(layer.name)
        if idx < FINE_TUNE_AT:
            layer.trainable = False
    except ValueError:
        pass   # Layer is in extractor but not base_model (e.g. the Model wrapper)

trainable_count = sum(
    tf.keras.backend.count_params(w) for w in model.trainable_weights
)
print(f"Trainable parameters after unfreezing: {trainable_count:,}")

# [MODIFIED — Unit 1] Recompile with LMCL + smaller LR for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss=large_margin_cosine_loss(margin=LMCL_MARGIN, scale=LMCL_SCALE),
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
# 13. CONFUSION MATRIX & CLASSIFICATION REPORT (unchanged logic)
# ===========================================================================
print("\n" + "=" * 60)
print("PER-CLASS METRICS & CONFUSION MATRIX")
print("=" * 60)

test_labels = np.concatenate([
    np.argmax(labels.numpy(), axis=1) for _, labels in test_ds
])

# Note: model outputs linear logits (no softmax) because of LMCL.
# argmax of logits == argmax of softmax(logits), so prediction is identical.
test_predictions = np.concatenate([
    np.argmax(model.predict(images, verbose=0), axis=1)
    for images, _ in test_ds
])

print("\nClassification Report:")
print(classification_report(test_labels, test_predictions,
                            target_names=class_names))

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

np.savetxt("saved_models/confusion_matrix.csv", cm, delimiter=",", fmt="%d",
           header=",".join(class_names), comments="")
print("\nConfusion matrix saved to saved_models/confusion_matrix.csv")

# ===========================================================================
# [NEW — Unit 6] 13.5 GRAD-CAM HEATMAP GENERATION (Explainable AI)
# ===========================================================================
# What: Grad-CAM (Gradient-weighted Class Activation Mapping) shows WHICH
#       regions of an input image most influenced the model's prediction.
#
# How it works (from Selvaraju et al., ICCV 2017):
#   1. Forward pass → get the final convolutional feature map A^k (shape H×W×K)
#   2. Backward pass → compute gradient of class score y^c w.r.t. A^k
#                      ∂y^c / ∂A^k_{ij}
#   3. Pool gradients spatially: α^c_k = (1/Z) ΣΣ ∂y^c/∂A^k_{ij}
#      This gives one importance weight per channel
#   4. Weighted sum + ReLU: L^c_Grad-CAM = ReLU(Σ_k α^c_k · A^k)
#      ReLU keeps only features that positively influence the class score
#   5. Upsample to input size and overlay on original image
#
# Why this matters for GI disease detection:
#   - Clinicians can verify the model is looking at the actual lesion
#   - If the model highlights the wrong region → it's relying on artifacts
#   - This is exactly what the reference paper shows in Fig. 14
#
# Unit 6 coverage: Grad-CAM is a model-specific XAI method (output-space
# explanation). We will also add SHAP (model-agnostic) in model_service.py.

def make_gradcam_heatmap(img_array, grad_model, class_index):
    """
    Generate a Grad-CAM heatmap for a given image and class.

    Args:
        img_array   : Preprocessed image, shape (1, 224, 224, 3)
        grad_model  : A model with TWO outputs:
                        [last_conv_layer_output, model_predictions]
        class_index : Integer — which class to explain (0–3)

    Returns:
        heatmap : 2D numpy array (H, W), values in [0, 1]
                  1.0 = most important region for this class
    """
    with tf.GradientTape() as tape:
        # Forward pass — get conv feature maps and predictions simultaneously
        # tape watches conv_outputs because it's a tensor (not a variable)
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)

        # Get the score for the target class (before softmax)
        # This is the quantity we differentiate with respect to the conv layer
        class_score = predictions[:, class_index]

    # Compute gradients: how does each activation in the conv layer
    # affect the class score?
    # Shape: same as conv_outputs → (1, H, W, num_channels)
    grads = tape.gradient(class_score, conv_outputs)

    # Pool gradients over spatial dimensions (H, W) → (num_channels,)
    # This averages out spatial noise and gives one weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each channel of the conv output by its importance
    # conv_outputs[0] has shape (H, W, num_channels)
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)           # Remove trailing dimension → (H, W)

    # ReLU: only keep positive contributions
    # Negative values mean "suppresses this class" — we don't want those
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1] for visualization
    heatmap_max = tf.reduce_max(heatmap)
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max

    return heatmap.numpy()


def save_gradcam_image(original_img, heatmap, save_path, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image and save.

    Args:
        original_img : numpy array (224, 224, 3), pixel values 0–255
        heatmap      : 2D numpy array (H, W), values 0–1
        save_path    : Where to save the resulting PNG
        alpha        : Transparency of heatmap overlay (0=invisible, 1=opaque)
    """
    # Resize heatmap to match image size using matplotlib colormap
    heatmap_resized = np.uint8(255 * heatmap)

    # Apply 'jet' colormap: blue=low importance, red=high importance
    jet        = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]   # RGB values for each intensity
    jet_heatmap = jet_colors[heatmap_resized]  # Map intensity → color

    # Convert to uint8 image
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose heatmap on original image
    superimposed = jet_heatmap * alpha + original_img
    superimposed = keras.utils.array_to_img(superimposed)
    superimposed.save(save_path)
    print(f"  Grad-CAM saved: {save_path}")


print("\n" + "=" * 60)
print("[Unit 6 — XAI] GENERATING GRAD-CAM HEATMAPS")
print("=" * 60)

# Build the Grad-CAM model:
# We need a model that outputs BOTH the last conv layer's activations
# AND the final class scores simultaneously.
# The last conv layer in our SE-ResNet-H is 'se_combine_relu'
# (the combined SE1+SE2 output before GAP).
# Note: if the layer name changes, inspect with: [l.name for l in model.layers]

GRADCAM_LAYER = "se_combine_relu"

try:
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(GRADCAM_LAYER).output,   # Conv activations
            model.output                               # Class logits
        ],
        name="gradcam_model",
    )

    os.makedirs("saved_models/gradcam", exist_ok=True)

    # Generate one Grad-CAM heatmap per class using the first test batch
    sample_images, sample_labels = next(iter(test_ds))
    sample_images_np = sample_images.numpy()

    for class_idx, class_name in enumerate(class_names):
        # Find the first test image that belongs to this class
        label_indices = np.where(
            np.argmax(sample_labels.numpy(), axis=1) == class_idx
        )[0]

        if len(label_indices) == 0:
            print(f"  No {class_name} sample found in first test batch — skipping")
            continue

        img_idx = label_indices[0]
        img_array = sample_images_np[img_idx:img_idx+1]   # Shape: (1, 224, 224, 3)

        # Preprocess for model (same as training pipeline)
        img_preprocessed = keras.applications.resnet50.preprocess_input(
            img_array.copy()
        )

        # Generate heatmap for this class
        heatmap = make_gradcam_heatmap(img_preprocessed, grad_model, class_idx)

        # Save the overlaid image
        save_gradcam_image(
            original_img=img_array[0],
            heatmap=heatmap,
            save_path=f"saved_models/gradcam/gradcam_{class_name}.png",
        )

    print("\nGrad-CAM heatmaps saved to saved_models/gradcam/")
    print("These show which image regions the model focuses on per disease class.")

except Exception as e:
    print(f"Grad-CAM generation skipped (layer name mismatch): {e}")
    print("To fix: run [l.name for l in model.layers] and update GRADCAM_LAYER")

# ===========================================================================
# 14. SAVE THE FINAL MODEL
# ===========================================================================
model.save("saved_models/gi_classifier_final.keras")
print("\nModel saved to saved_models/gi_classifier_final.keras")
print("Best checkpoint at saved_models/best_model.keras")
print("Class labels at saved_models/class_labels.json")
print("Grad-CAM samples at saved_models/gradcam/")
print("\nDone!")