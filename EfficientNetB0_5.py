
# ── 1. IMPORTS ───────────────────────────────────────────────
import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")



# ── 2. SEEDS ─────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



# ── 3. GPU ───────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] {len(gpus)} GPU(s) detected and configured.")
else:
    print("[INFO] No GPU found — running on CPU.")



# ── 4. CONFIG ────────────────────────────────────────────────
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-3
DROPOUT_RATE = 0.4
NUM_CLASSES  = 5

CSV_PATH   = "/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading.csv"
IMAGE_DIR  = "/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading/DR_grading"
MODEL_PATH = "/kaggle/working/efficientnetb0_ddr_best.h5"
CLASS_NAMES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]



# ── 5. DATA PREP ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(df.head())

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["diagnosis"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=SEED
)

for split in [train_df, val_df, test_df]:
    split["diagnosis"] = split["diagnosis"].astype(str)

print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")



# ── 6. GENERATORS ────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

gen_kwargs = dict(
    directory=IMAGE_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    validate_filenames=False,
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df, shuffle=True, seed=SEED, **gen_kwargs
)
valid_gen = val_test_datagen.flow_from_dataframe(
    dataframe=val_df, shuffle=False, **gen_kwargs
)
test_gen = val_test_datagen.flow_from_dataframe(
    dataframe=test_df, shuffle=False, **gen_kwargs
)

print(f"[INFO] Train samples : {train_gen.samples}")
print(f"[INFO] Valid samples : {valid_gen.samples}")
print(f"[INFO] Test  samples : {test_gen.samples}")



# ── 7. MODEL ─────────────────────────────────────────────────
def build_model(input_shape=(224, 224, 3)):
    backbone = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling=None,
    )

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = backbone(inputs, training=True)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(DROPOUT_RATE / 2, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    return Model(inputs, outputs, name="EfficientNetB0_DDR")

model = build_model()
model.summary()



# ── 8. COMPILE ───────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
    ],
)



# ── 9. CALLBACKS (Phase 1) ───────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
]


# ── 9. TRAIN ──────────────────────────────────────────────────
print("\n[INFO] Starting training …\n")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=valid_gen,
    callbacks=callbacks,
    verbose=1,
)
print("\n[INFO] Training complete.")


# ── FINE-TUNING PHASE ────────────────────────────────────────
FT_LR         = 1e-4
FT_EPOCHS     = 20
UNFREEZE_FROM = 100
FT_MODEL_PATH = "/kaggle/working/efficientnetb0_ddr_finetuned.h5"

# Find and selectively unfreeze backbone
backbone = None
for layer in model.layers:
    if "efficientnetb0" in layer.name.lower():
        backbone = layer
        break

backbone.trainable = True
for i, layer in enumerate(backbone.layers):
    layer.trainable = (i >= UNFREEZE_FROM)

print(f"[INFO] Unfreezing layers {UNFREEZE_FROM}–{len(backbone.layers)-1} of backbone")

model.compile(
    optimizer=Adam(learning_rate=FT_LR),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
    ],
)

ft_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=7, mode="max",
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3,
                      min_lr=1e-7, verbose=1),
    ModelCheckpoint(filepath=FT_MODEL_PATH, monitor="val_accuracy",
                    mode="max", save_best_only=True, verbose=1),
]

print("\n[INFO] Starting fine-tuning …\n")
ft_history = model.fit(
    train_gen,
    epochs=FT_EPOCHS,
    validation_data=valid_gen,
    callbacks=ft_callbacks,
    verbose=1,
)
print("\n[INFO] Fine-tuning complete.")


# ── 11. TRAINING CURVES ──────────────────────────────────────
def plot_history(h1, h2):
    for metric in ["loss", "accuracy"]:
        p1  = h1.history[metric]
        p2  = h2.history[metric]
        vp1 = h1.history[f"val_{metric}"]
        vp2 = h2.history[f"val_{metric}"]

        full_train = p1 + p2
        full_val   = vp1 + vp2
        split_ep   = len(p1)

        plt.figure(figsize=(10, 4))
        plt.plot(full_train, label=f"Train {metric}", lw=2)
        plt.plot(full_val,   label=f"Val {metric}",   lw=2, ls="--")
        plt.axvline(split_ep - 1, color="red", ls=":", lw=1.5,
                    label="Fine-tune start")
        plt.title(f"{metric.capitalize()} — Phase 1 + Fine-Tune", fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel(metric.capitalize())
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"/kaggle/working/curves_{metric}.png", dpi=150, bbox_inches="tight")
        plt.show()

    print("[INFO] Training curves saved.")

plot_history(history, ft_history)   # ← fixed: was fine_tune_history




# ── 12. EVALUATE ON TEST SET ─────────────────────────────────
print("\n[INFO] Evaluating on test set …")
test_gen.reset()
test_results = model.evaluate(test_gen, verbose=1)

metric_names = ["loss", "accuracy", "top2_acc"]
print("\n── Test Metrics ──────────────────────────────")
for name, value in zip(metric_names, test_results):
    print(f"  {name:15s}: {value:.4f}")


# ── 13. PREDICTIONS ──────────────────────────────────────────
test_gen.reset()
y_prob = model.predict(test_gen, verbose=1)   # shape: (N, 5)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_gen.classes


# ── 14. CONFUSION MATRIX ─────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Row %)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Confusion matrix saved.")

plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)


# ── 15. CLASSIFICATION REPORT ────────────────────────────────
print("\n── Classification Report ─────────────────────")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ── 16. ROC CURVE (one-vs-rest, per class) ───────────────────
def plot_roc_multiclass(y_true, y_prob, class_names):
    n_classes  = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — One-vs-Rest (per DR grade)",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/kaggle/working/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] ROC curve saved.")

plot_roc_multiclass(y_true, y_prob, CLASS_NAMES)


# ── 17. SUMMARY TABLE ────────────────────────────────────────
summary_df = pd.DataFrame({
    "Metric": metric_names,
    "Value":  [round(v, 4) for v in test_results],
})
print("\n── Final Summary ─────────────────────────────")
print(summary_df.to_string(index=False))


# ── 18. SAVE MODEL ───────────────────────────────────────────
model.save(FT_MODEL_PATH)
print(f"\n[INFO] Fine-tuned model saved to: {FT_MODEL_PATH}")
print("[INFO] All done. ✓")
