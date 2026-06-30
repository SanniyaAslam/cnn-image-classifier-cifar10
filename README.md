# CIFAR-10 Image Classifier — CNN from Scratch

A convolutional neural network built with TensorFlow/Keras that classifies images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using the CIFAR-10 dataset.

Built as part of the OptimusAutomate AI Internship Program.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **78.05%** |
| Test Loss | 0.6665 |
| Total Parameters | 325,418 |
| Epochs Trained | 30 |

Per-class performance (precision / recall / F1):

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| airplane | 0.79 | 0.81 | 0.80 |
| automobile | 0.79 | 0.94 | 0.86 |
| bird | 0.88 | 0.55 | 0.68 |
| cat | 0.75 | 0.56 | 0.64 |
| deer | 0.85 | 0.69 | 0.76 |
| dog | 0.71 | 0.77 | 0.74 |
| frog | 0.68 | 0.89 | 0.77 |
| horse | 0.85 | 0.83 | 0.84 |
| ship | 0.93 | 0.83 | 0.88 |
| truck | 0.70 | 0.93 | 0.80 |

See [`task1_cnn_cifar10.ipynb`](task1_cnn_cifar10.ipynb) for the full executed notebook with training logs, epoch-by-epoch progress, and complete output.

**Observation:** Cats and birds were the hardest classes to classify (lowest recall), likely due to visual similarity with other animals in low-resolution 32×32 images. Automobile and truck had the highest recall but lower precision, suggesting the model sometimes confuses other vehicle-like shapes for these classes.

## Architecture

A 3-block convolutional architecture:

```
Input (32×32×3)
  → Data Augmentation (flip, rotate, zoom, translate)
  → [Conv2D(32) → BatchNorm → ReLU] × 2 → MaxPool → Dropout(0.25)
  → [Conv2D(64) → BatchNorm → ReLU] × 2 → MaxPool → Dropout(0.25)
  → [Conv2D(128) → BatchNorm → ReLU] × 2 → MaxPool → Dropout(0.25)
  → GlobalAveragePooling2D
  → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
  → Dense(10) → Softmax
```

**Design choices:**
- **BatchNorm after every conv layer** — stabilizes and speeds up training by normalizing activations between layers.
- **Dropout at increasing rates** (0.25 → 0.5) — prevents overfitting; higher dropout closer to the output where the model is most prone to memorizing.
- **GlobalAveragePooling instead of Flatten** — drastically reduces parameter count (no giant flatten→dense layer) while keeping accuracy comparable; also more robust to spatial shifts.
- **Data augmentation** (random flip, rotation, zoom, translation) — synthetically expands the training set so the model generalizes better instead of memorizing exact pixel patterns.
- **Adam optimizer with ReduceLROnPlateau** — automatically lowers the learning rate when validation loss stalls, helping the model fine-tune in later epochs.
- **EarlyStopping** — halts training once validation accuracy stops improving, avoiding wasted epochs and overfitting.

## How to Run

```bash
pip install -r requirements.txt
python task1_cnn_cifar10.py
```

Or open in Google Colab (recommended — free GPU, much faster):
1. Upload `task1_cnn_cifar10.py` to Colab
2. Runtime → Change runtime type → GPU
3. Run all cells

CIFAR-10 (~170MB) downloads automatically on first run via `keras.datasets.cifar10`.

## What I Learned

- How convolutional filters detect hierarchical features (edges → shapes → objects)
- Why normalization (both BatchNorm and pixel scaling) matters for training stability
- How to diagnose overfitting by comparing train vs. validation curves
- The tradeoffs between dropout, data augmentation, and model capacity for regularization
- Evaluating a classifier beyond accuracy — confusion matrices, precision/recall, per-class performance

## Tech Stack

- TensorFlow / Keras
- NumPy
- Matplotlib / Seaborn (visualization)
- scikit-learn (evaluation metrics)

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 color images across 10 classes (50,000 train / 10,000 test), created by the Canadian Institute for Advanced Research.
