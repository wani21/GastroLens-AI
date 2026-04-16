# GastroLens-AI — Model Evaluation Report

## 1. Project Summary

GastroLens-AI is a deep learning system for classifying gastrointestinal endoscopy images into four diagnostic categories:

- `esophagitis`
- `normal`
- `polyp`
- `ulcer`

The model was built using **transfer learning** with a **ResNet50** backbone pretrained on ImageNet, following a two-phase training strategy (frozen base + fine-tuning).

---

## 2. Dataset

Source: **Kvasir v2** endoscopy image dataset, reorganized into 4 clinical classes.

| Split | esophagitis | normal | polyp | ulcer | Total |
|-------|-------------|--------|-------|-------|-------|
| Train | 924 | 2,716 | 2,742 | 920 | **7,302** |
| Val   | 365 | 1,070 | 1,095 | 360 | **2,890** |
| Test  | 194 | 568   | 576   | 192 | **1,530** |

Class imbalance was handled using **computed class weights** (balanced mode).

---

## 3. Model Architecture

- **Base model:** ResNet50 (ImageNet weights, `include_top=False`)
- **Custom head:**
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(256, ReLU)
  - Dropout(0.3)
  - Dense(4, Softmax)
- **Total parameters:** 24,113,284
- **Input shape:** 224 × 224 × 3

### Training Strategy

| Phase | Base Model | Trainable Params | Learning Rate | Max Epochs |
|-------|------------|------------------|---------------|------------|
| 1 — Head training | Frozen | 525,572 | 1e-3 | 10 |
| 2 — Fine-tuning | Top 32 layers unfrozen | ~8M | 1e-5 | 15 |

### Data Augmentation
Random horizontal/vertical flip, rotation (±20%), zoom (±20%), contrast (±20%).

### Loss & Optimizer
- Loss: `categorical_crossentropy`
- Optimizer: Adam
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 4. Final Test Metrics

Evaluated on the **held-out test set (1,530 images)** — never seen during training:

| Metric | Score |
|--------|-------|
| **Accuracy** | **94.79%** |
| Weighted Precision | 94.83% |
| Weighted Recall | 94.79% |
| **Weighted F1-score** | **94.68%** |

---

## 5. Confusion Matrix

Rows = actual class, Columns = predicted class.

|                   | esophagitis | normal | polyp | ulcer |
|-------------------|-------------|--------|-------|-------|
| **esophagitis**   | **144**     | 46     | 0     | 0     |
| **normal**        | 13          | **560**| 1     | 4     |
| **polyp**         | 0           | 5      | **562**| 9    |
| **ulcer**         | 0           | 0      | 2     | **189** |

---

## 6. Per-Class Performance

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| esophagitis   | 0.92      | 0.76   | 0.83     | 190     |
| normal        | 0.92      | 0.97   | 0.94     | 578     |
| polyp         | 0.99      | 0.98   | 0.99     | 576     |
| ulcer         | 0.94      | 0.99   | 0.96     | 191     |

---

## 7. Key Observations

### Strengths
- **Strong overall performance** with 94.79% test accuracy.
- **Polyp detection is near-perfect** (F1 = 0.99) — critical for colon cancer screening.
- **Ulcer detection is excellent** (F1 = 0.96) with very few false negatives.
- Transfer learning was highly effective, converging in a small number of epochs.

### Weaknesses
- **Esophagitis recall is lower (0.76)** — 46 of 190 esophagitis images were misclassified as `normal`. This is clinically understandable: early-stage esophagitis can look visually similar to healthy mucosa. However, in a medical context, this is the most concerning error type (false negatives → missed diagnoses).
- **Normal class has a small number of misclassifications** scattered across the other 3 classes (18 total), which is acceptable.

### Interpretation for Clinical Use
- The model is **deployment-ready for assistive screening** but should not replace expert diagnosis.
- Polyps and ulcers are detected reliably; esophagitis cases should ideally be reviewed by a clinician.

---

## 8. Common Mistakes Avoided

1. Used **class weights** to prevent bias toward majority classes (normal/polyp).
2. Started with **frozen base** to avoid destroying pretrained features with untrained head gradients.
3. Used a **very small learning rate (1e-5)** during fine-tuning.
4. Kept the **test set completely isolated** from training/validation.
5. Applied proper **ResNet50 preprocessing** (`preprocess_input`) matching ImageNet training.

---

## 9. Class Label Mapping

Stored in `saved_models/class_labels.json`:

```json
{
  "0": "esophagitis",
  "1": "normal",
  "2": "polyp",
  "3": "ulcer"
}
```

---

## 10. Files Produced

| File | Description |
|------|-------------|
| `saved_models/best_model.keras` | Best checkpoint (highest val_accuracy) — use for inference |
| `saved_models/gi_classifier_final.keras` | Final model after fine-tuning |
| `saved_models/confusion_matrix.csv` | Raw confusion matrix data |
| `saved_models/class_labels.json` | Index-to-label mapping for backend |

---

## 11. Next Steps

- Build a **backend API** (Flask / FastAPI) for model inference.
- Build a **frontend** for image upload and prediction display.
- Optionally: improve esophagitis recall with targeted augmentation or a second-opinion ensemble model.

---

## 12. How to Reproduce

```powershell
# Activate virtual environment
.\tf-env\Scripts\activate

# Run training + evaluation
python train.py
```

The script prints final metrics, the classification report, and confusion matrix, and saves all outputs to `saved_models/`.

---

*Generated as part of the GastroLens-AI project deliverables.*
