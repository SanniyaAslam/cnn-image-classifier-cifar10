"""
================================================================================
TASK 1: CNN Image Classification on CIFAR-10
================================================================================
WHAT IS CIFAR-10?
  - A dataset of 60,000 color images (32x32 pixels, RGB = 3 channels)
  - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  - 50,000 training images + 10,000 test images
  - Created by the Canadian Institute for Advanced Research (hence "CI-FAR")

WHY CNN (Convolutional Neural Network)?
  - Regular neural networks (Dense/Fully Connected) treat an image as a flat list
    of numbers. They lose all spatial information (which pixels are neighbors).
  - CNNs use "filters" that slide over the image and detect features like:
      · Edges (early layers)
      · Shapes/curves (middle layers)
      · Object parts like eyes, wheels (deeper layers)
  - This makes CNNs much better at image tasks.
================================================================================
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')            # Use non-interactive backend for file saving
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras — the most popular deep learning framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Scikit-learn for evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds so results are reproducible
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("  TASK 1: CNN Image Classification — CIFAR-10")
print(f"  TensorFlow version: {tf.__version__}")
print("=" * 60)


# ─── STEP 1: LOAD THE DATASET ───────────────────────────────────────────────
"""
WHAT HAPPENS HERE:
  Keras has CIFAR-10 built-in. It downloads it automatically.
  
  x_train: images for training  → shape (50000, 32, 32, 3)
              ↑ 50k images, each 32×32 pixels, 3 color channels (R, G, B)
  y_train: labels for training  → shape (50000, 1)
              ↑ a number 0–9 representing the class

INTERVIEW TIP:
  Always split your data into train / validation / test.
  - Train set   → model learns from this
  - Validation  → you monitor performance during training (tune hyperparameters)
  - Test set    → final evaluation, never touched during training
"""
print("\n[STEP 1] Loading CIFAR-10 dataset...")
# This downloads CIFAR-10 automatically the first time you run it (~170 MB).
# It's cached afterwards, so subsequent runs load instantly from disk.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

# Class names (label 0 = airplane, label 1 = automobile, etc.)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"  Training images:   {x_train.shape}  → {x_train.shape[0]:,} images")
print(f"  Training labels:   {y_train.shape}")
print(f"  Test images:       {x_test.shape}")
print(f"  Pixel value range: {x_train.min()} to {x_train.max()}")


# ─── STEP 2: PREPROCESSING ──────────────────────────────────────────────────
"""
WHY NORMALIZE?
  Raw pixel values range from 0 to 255.
  Neural networks train MUCH better when inputs are small numbers (0.0 to 1.0).
  
  This is because:
  - Large values cause large gradients → unstable training
  - Normalization makes all features contribute equally
  - Simply divide by 255 to scale to [0, 1]

WHAT IS DATA AUGMENTATION?
  Training data is limited. Augmentation creates modified copies of images on-the-fly:
  - Flip horizontally → a dog facing left becomes a dog facing right (still a dog!)
  - Random rotation   → slightly tilted images
  - Random zoom       → slightly zoomed in/out
  
  This helps the model generalize better and reduces overfitting.
  
  WHY DOES OVERFITTING HAPPEN?
  - The model memorizes training data instead of learning general patterns
  - Signs: training accuracy 99%, test accuracy 70% → overfitting!
  - Solutions: dropout, data augmentation, batch normalization, less complexity
"""
print("\n[STEP 2] Preprocessing data...")

# Normalize pixel values from [0, 255] → [0.0, 1.0]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# Flatten labels from shape (50000, 1) → (50000,) — needed for sparse_categorical
y_train = y_train.flatten()
y_test  = y_test.flatten()

print(f"  After normalization: pixel range {x_train.min():.1f} to {x_train.max():.1f}")

# Split 20% of training data as validation set
val_size = int(0.2 * len(x_train))
x_val,   x_train  = x_train[:val_size],  x_train[val_size:]
y_val,   y_train  = y_train[:val_size],  y_train[val_size:]

print(f"  Train: {x_train.shape[0]:,} | Val: {x_val.shape[0]:,} | Test: {x_test.shape[0]:,}")

# Data Augmentation — using Keras augmentation layers
"""
These layers are applied ONLY during training (not during testing).
They randomly transform images each time they're seen, creating variety.
"""
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),          # Flip image left-right randomly
    layers.RandomRotation(0.1),               # Rotate up to ±10% (about ±36°)
    layers.RandomZoom(0.1),                   # Zoom in/out by up to 10%
    layers.RandomTranslation(0.1, 0.1),       # Shift image slightly
], name="data_augmentation")


# ─── STEP 3: BUILD THE CNN MODEL ────────────────────────────────────────────
"""
CNN ARCHITECTURE EXPLAINED:

┌─────────────────────────────────────────────────────────────────────┐
│  Input: 32×32×3 image                                               │
│                                                                      │
│  [Conv2D] → detects features using learnable filters (kernels)      │
│  [BatchNorm] → normalizes layer outputs → faster, stable training   │
│  [ReLU] → activation function: max(0, x) → adds non-linearity       │
│  [MaxPooling] → reduces spatial size (32→16→8) → fewer parameters  │
│  [Dropout] → randomly turns off neurons → prevents overfitting      │
│  ...repeat with more filters (deeper = more complex features)...    │
│                                                                      │
│  [GlobalAvgPooling] → collapses spatial dimensions to 1D vector     │
│  [Dense] → fully connected layer → learns class relationships       │
│  [Output: 10 neurons] → one per class, softmax → probabilities      │
└─────────────────────────────────────────────────────────────────────┘

KEY TERMS FOR INTERVIEWS:
  - Filter/Kernel: small matrix (e.g. 3×3) that slides over the image
  - Feature Map: output of a Conv layer (the "detected features")
  - Stride: how many pixels the filter moves at each step
  - Padding='same': adds zeros around edges so output size = input size
  - Max Pooling: takes the maximum value in a region → keeps strongest feature
  - BatchNorm: normalizes each mini-batch → helps convergence
  - Dropout(0.5): during training, randomly sets 50% of neurons to 0
  - Softmax: converts raw scores to probabilities that sum to 1
"""
print("\n[STEP 3] Building CNN model...")

def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Builds a CNN model using the Functional API.
    
    Architecture: 3 blocks of Conv→BatchNorm→ReLU→MaxPool→Dropout
    Then: GlobalAvgPool → Dense → Output
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # ── Augmentation (only active during training) ──
    x = data_augmentation(inputs)
    
    # ══════════════════════════════════════════════
    # BLOCK 1: 32 filters, detect basic features
    # ══════════════════════════════════════════════
    # Conv2D(32, 3, 3):
    #   32 = number of filters (output feature maps)
    #   3×3 = kernel size (each filter looks at a 3×3 patch)
    #   padding='same' = output same spatial size as input
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    # MaxPooling: reduces 32×32 → 16×16
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    # Dropout: 25% neurons randomly zeroed during training
    x = layers.Dropout(0.25, name='drop1')(x)

    # ══════════════════════════════════════════════
    # BLOCK 2: 64 filters, detect mid-level features
    # ══════════════════════════════════════════════
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    # MaxPooling: reduces 16×16 → 8×8
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='drop2')(x)

    # ══════════════════════════════════════════════
    # BLOCK 3: 128 filters, detect high-level features
    # ══════════════════════════════════════════════
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Activation('relu', name='relu5')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv6')(x)
    x = layers.BatchNormalization(name='bn6')(x)
    x = layers.Activation('relu', name='relu6')(x)
    # MaxPooling: reduces 8×8 → 4×4
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.25, name='drop3')(x)

    # ══════════════════════════════════════════════
    # CLASSIFIER HEAD
    # ══════════════════════════════════════════════
    # GlobalAveragePooling: takes average across spatial dimensions
    # 4×4×128 → 128 (one value per filter) — much lighter than Flatten!
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layer: fully connected, learns class relationships
    x = layers.Dense(256, name='fc1')(x)
    x = layers.BatchNormalization(name='bn7')(x)
    x = layers.Activation('relu', name='relu7')(x)
    x = layers.Dropout(0.5, name='drop4')(x)   # Higher dropout before output
    
    # Output layer: 10 neurons (one per class)
    # Softmax converts scores → probabilities (all 10 sum to 1.0)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='CIFAR10_CNN')
    return model

model = build_cnn_model()
model.summary()


# ─── STEP 4: COMPILE THE MODEL ──────────────────────────────────────────────
"""
COMPILATION = telling the model:
  1. OPTIMIZER: how to update weights
     - Adam is the most popular. It adapts the learning rate automatically.
     - Learning rate (lr=0.001): how big each weight update step is
       Too high → overshoots, never converges
       Too low  → takes forever to train
     
  2. LOSS FUNCTION: what to minimize during training
     - sparse_categorical_crossentropy: used for multi-class classification
       when labels are integers (0–9), not one-hot encoded
     
  3. METRICS: what to display/track
     - accuracy: % of correct predictions on each batch

INTERVIEW COMMON QUESTION: "What's the difference between Adam, SGD, RMSprop?"
  - SGD: basic gradient descent, needs careful lr tuning
  - Adam: adapts lr per parameter, very robust, usually best default choice
  - RMSprop: similar to Adam, often used in RNNs
"""
print("\n[STEP 4] Compiling model...")

# Learning rate schedule: start at 0.001, reduce when val_loss stops improving
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',    # Watch validation loss
    factor=0.5,            # Multiply lr by 0.5 when plateau
    patience=5,            # Wait 5 epochs before reducing
    min_lr=1e-6,           # Don't go below this
    verbose=1
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ─── STEP 5: TRAIN THE MODEL ────────────────────────────────────────────────
"""
TRAINING CONCEPTS:

  EPOCH: one complete pass through the entire training dataset
  BATCH: model doesn't see all 40,000 images at once (too much memory)
         It processes them in small groups (batches) of 64 images
  
  Each epoch = (40,000 / 64) ≈ 625 batches
  Each batch: forward pass → compute loss → backward pass (backpropagation) → update weights

  CALLBACKS (things that happen automatically during training):
  - EarlyStopping: stops training if val_accuracy doesn't improve → saves time
  - ModelCheckpoint: saves the best model to disk → so you can load it later
  - ReduceLROnPlateau: reduces learning rate when training stalls

  BACKPROPAGATION (how the model learns):
  - After each batch, compute how wrong the prediction was (loss)
  - Use chain rule of calculus to figure out how much each weight contributed to the error
  - Adjust weights in the direction that reduces the error (gradient descent)
"""
print("\n[STEP 5] Training model...")
print("  This may take a few minutes...\n")

EPOCHS = 30
BATCH_SIZE = 64

callbacks = [
    # Stop training early if val_accuracy doesn't improve for 10 epochs
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,  # Revert to best weights when stopping
        verbose=1
    ),
    # Save the best model
    keras.callbacks.ModelCheckpoint(
        'best_cifar10_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    ),
    lr_scheduler
]

history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),   # Watch performance on validation set
    callbacks=callbacks,
    verbose=1
)

print("\n  Training complete!")


# ─── STEP 6: EVALUATE ON TEST SET ───────────────────────────────────────────
"""
EVALUATION:
  Now we test on data the model has NEVER seen (the test set).
  This is the true measure of model performance.
  
  METRICS TO KNOW FOR INTERVIEWS:
  - Accuracy: (correct predictions) / (total predictions)
             Works well only when classes are balanced
  
  - Precision: Of all images predicted as "cat", how many were actually cats?
               High precision = few false alarms
  
  - Recall: Of all actual cats, how many did the model find?
            High recall = few misses
  
  - F1-Score: Harmonic mean of precision and recall (2*P*R / (P+R))
              Best single metric when you care about both
  
  - Confusion Matrix: A 10×10 grid showing which classes get confused
                      Diagonal = correct predictions
                      Off-diagonal = mistakes (e.g., cat predicted as dog)
"""
print("\n[STEP 6] Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")

# Get predictions for all test images
y_pred_probs = model.predict(x_test, verbose=0)   # Shape: (10000, 10) — probabilities
y_pred = np.argmax(y_pred_probs, axis=1)           # Take class with highest probability

print("\n  Per-Class Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


# ─── STEP 7: VISUALIZATIONS ─────────────────────────────────────────────────
"""
WHY VISUALIZE?
  - Accuracy/loss curves tell you if the model trained well or overfit
  - Confusion matrix shows which classes are hard to distinguish
  - Sample predictions help you understand model behavior intuitively

  READING TRAINING CURVES:
  - Good training: both train and val accuracy go up together, converge
  - Overfitting:   train accuracy keeps going up, val accuracy plateaus or drops
  - Underfitting:  both accuracies stay low
"""
print("\n[STEP 7] Generating visualizations...")

os.makedirs('plots', exist_ok=True)

# ── Plot 1: Training History ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN Training History — CIFAR-10', fontsize=14, fontweight='bold')

epochs_ran = len(history.history['accuracy'])

# Accuracy plot
axes[0].plot(history.history['accuracy'],     label='Train Accuracy',      color='#2196F3', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF5722', linewidth=2, linestyle='--')
axes[0].set_title('Model Accuracy Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Loss plot
axes[1].plot(history.history['loss'],     label='Train Loss',      color='#2196F3', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='#FF5722', linewidth=2, linestyle='--')
axes[1].set_title('Model Loss Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/01_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_training_history.png")

# ── Plot 2: Confusion Matrix ──────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
# Normalize to percentages (per row = per true class)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Confusion Matrix — CIFAR-10', fontsize=14, fontweight='bold')

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], cbar=True)
axes[0].set_title('Confusion Matrix (Counts)')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')
axes[0].tick_params(axis='x', rotation=45)

# Normalized percentages
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], cbar=True)
axes[1].set_title('Confusion Matrix (% per True Class)')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/02_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_confusion_matrix.png")

# ── Plot 3: Per-Class Accuracy Bar Chart ─────────────────────────────────
per_class_acc = cm_normalized.diagonal()

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#4CAF50' if a >= 70 else '#FF9800' if a >= 50 else '#F44336' for a in per_class_acc]
bars = ax.bar(CLASS_NAMES, per_class_acc, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=test_accuracy*100, color='navy', linestyle='--', linewidth=2, label=f'Overall: {test_accuracy*100:.1f}%')
ax.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Class')
ax.set_ylim([0, 110])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/03_per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_per_class_accuracy.png")

# ── Plot 4: Sample Predictions ───────────────────────────────────────────
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=13, fontweight='bold')

# Pick 32 random test images
indices = np.random.choice(len(x_test), 32, replace=False)
for i, idx in enumerate(indices):
    ax = axes[i // 8][i % 8]
    ax.imshow(x_test[idx])
    true_label = CLASS_NAMES[y_test[idx]]
    pred_label = CLASS_NAMES[y_pred[idx]]
    confidence = y_pred_probs[idx][y_pred[idx]] * 100
    color = 'green' if y_test[idx] == y_pred[idx] else 'red'
    ax.set_title(f'T: {true_label}\nP: {pred_label}\n{confidence:.0f}%',
                 fontsize=7, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('plots/04_sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_sample_predictions.png")

# ── Plot 5: CIFAR-10 Sample Images ───────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('CIFAR-10 Dataset — One Sample Per Class', fontsize=13, fontweight='bold')

for class_idx, class_name in enumerate(CLASS_NAMES):
    # Find first image of this class in training set
    sample_idx = np.where(y_train_orig := y_train)[0]
    # Just grab index class_idx*1000 as a representative sample
    ax = axes[class_idx // 5][class_idx % 5]
    ax.imshow(x_train[class_idx * 1000])
    ax.set_title(f'{class_idx}: {class_name}', fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('plots/05_dataset_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_dataset_samples.png")


# ─── STEP 8: SUMMARY ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
print(f"  Test Loss:      {test_loss:.4f}")
print(f"  Epochs trained: {epochs_ran}")
print(f"  Model saved to: best_cifar10_model.keras")
print(f"  Plots saved to: plots/")
print("=" * 60)

# Count parameters
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\n  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"\n  INTERVIEW NOTE: Each parameter is a weight the model learns.")
print(f"  More parameters = more capacity but also more risk of overfitting.")

print("\n  All done! Task 1 complete. ✓")
