# =============================================================================
# ISP560 - MACHINE LEARNING | GROUP PROJECT
# Title  : AI Assistant Usage in Student Life — BPNN Optimizer Comparison
# Team   : DATAMAZING (Group CDCS2595B)
# Members:
#   - Muhammad Khairul Izzat Bin Khalid  (2024649896)
#   - Muhammad Imran Bin Roszaide        (2024446276)
#   - Nazif Muhsin Bin Mohd Zahir        (2024271454)
#   - Nik Mohamad Aniq Irfan Bin Nik Mahadi
#   - Muhammad Ameer Mubaraq Bin Adnan   (2023862318)
# =============================================================================

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, Adagrad, Nadam, Adadelta, SGD

from imblearn.over_sampling import SMOTE

import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# STEP 2: LOAD AND PREPROCESS DATASET
# =============================================================================
df = pd.read_csv("ai_assistant_usage_student_life.csv")

print(f"\nDataset shape: {df.shape}")
print(f"Target distribution:\n{df['UsedAgain'].value_counts()}")

# Preprocessing
df['UsedAgain'] = df['UsedAgain'].astype(int)
df.drop(['SessionID', 'SessionDate'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['StudentLevel', 'Discipline', 'TaskType', 'FinalOutcome'])

X = df.drop('UsedAgain', axis=1)
y = df['UsedAgain']

# Scale features
X_scaled = StandardScaler().fit_transform(X)

# Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE to handle class imbalance (training set only)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nTraining set size (after SMOTE): {X_train_sm.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# =============================================================================
# STEP 3: DEFINE MODEL ARCHITECTURE & FOCAL LOSS
# =============================================================================

def focal_loss(gamma=2., alpha=0.85):
    """Focal loss to handle class imbalance."""
    def loss(y_true, y_pred):
        bce = binary_crossentropy(y_true, y_pred)
        pt = K.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss

def build_model(optimizer, activation='relu'):
    """
    Build BPNN model.
    Architecture: Input -> Dense(64) -> Dropout(0.3) -> Dense(32) -> Dropout(0.3) -> Output
    """
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation=activation),
        Dropout(0.3),
        Dense(32, activation=activation),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
    return model

# Training configuration
EPOCHS = 50
BATCH_SIZE = 32
CLASS_WEIGHT = {0: 1.70, 1: 0.70}

# =============================================================================
# STEP 4: TRAIN MODELS — ALL OPTIMIZERS
# =============================================================================
# Each member contributed experiments with different optimizers:
#   Member 1: Adam, AdamW, RMSprop   (ReLU activation)
#   Member 2: Adam, AdamW, RMSprop   (ELU activation)
#   Member 3: Adagrad, Nadam, Adadelta (ReLU activation)
#   Member 4: Adagrad, Nadam, Adadelta (ELU activation)
#   Member 5: SGD, Nadam              (ReLU & ELU activation)

print("\n" + "="*60)
print("TRAINING MODELS — OPTIMIZER COMPARISON")
print("="*60)

optimizers = {
    'Adam':     Adam(learning_rate=0.001),
    'AdamW':    AdamW(learning_rate=0.001),
    'RMSprop':  RMSprop(learning_rate=0.001),
    'Adagrad':  Adagrad(learning_rate=0.001),
    'Nadam':    Nadam(learning_rate=0.001),
    'Adadelta': Adadelta(learning_rate=0.001),
    'SGD':      SGD(learning_rate=0.001),
}

histories = {}
models = {}

for name, opt in optimizers.items():
    print(f"\nTraining with {name}...")
    model = build_model(opt, activation='relu')
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.2,
        class_weight=CLASS_WEIGHT
    )
    histories[name] = history
    models[name] = model
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  {name} — Test Accuracy: {acc:.4f}")

# =============================================================================
# STEP 5: EVALUATE ALL MODELS
# =============================================================================
print("\n" + "="*60)
print("FINAL TEST ACCURACY COMPARISON")
print("="*60)

results = {}
for name, model in models.items():
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = acc
    print(f"  {name:12s}: {acc:.4f}")

best_optimizer = max(results, key=results.get)
print(f"\nBest Optimizer: {best_optimizer} ({results[best_optimizer]:.4f})")

# =============================================================================
# STEP 6: CLASSIFICATION REPORT & CONFUSION MATRIX (BEST MODEL)
# =============================================================================
print(f"\n--- Classification Report: {best_optimizer} ---")
best_model = models[best_optimizer]
y_probs = best_model.predict(X_test)
y_pred = (y_probs > 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred, target_names=["Will NOT Use Again", "Will Use Again"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix — {best_optimizer}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# =============================================================================
# STEP 7: VISUALIZATION — ACCURACY & LOSS CURVES
# =============================================================================
fig, axes = plt.subplots(len(optimizers), 2, figsize=(12, len(optimizers) * 3))
fig.suptitle("Training vs Validation — All Optimizers", fontsize=14, fontweight='bold')

for idx, (name, history) in enumerate(histories.items()):
    # Accuracy
    axes[idx, 0].plot(history.history['accuracy'], label='Train')
    axes[idx, 0].plot(history.history['val_accuracy'], label='Val')
    axes[idx, 0].set_title(f'{name} — Accuracy')
    axes[idx, 0].set_ylim(0, 1)
    axes[idx, 0].legend()
    axes[idx, 0].grid(True)

    # Loss
    axes[idx, 1].plot(history.history['loss'], label='Train')
    axes[idx, 1].plot(history.history['val_loss'], label='Val')
    axes[idx, 1].set_title(f'{name} — Loss')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# Accuracy comparison bar chart
plt.figure(figsize=(10, 5))
bars = plt.bar(results.keys(), results.values(), color=['#2196F3' if k != best_optimizer else '#4CAF50' for k in results])
plt.title("Test Accuracy by Optimizer")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.show()

print("\nDone. Output files: confusion_matrix.png, training_curves.png, accuracy_comparison.png")
