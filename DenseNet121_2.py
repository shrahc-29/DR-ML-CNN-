# ============================================================
# DDR Dataset — Binary Image Classification
# Model: DenseNet121 (from scratch, weights=None)
# Framework: TensorFlow / Keras
# ============================================================

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
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import (
    Precision,
    Recall,
    AUC,
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ── 2. REPRODUCIBILITY ───────────────────────────────────────
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── 3. GPU CONFIGURATION ─────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")

if gpus:

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"[INFO] {len(gpus)} GPU(s) detected and configured.")

else:
    print("[INFO] No GPU found — running on CPU.")


    # ── 4. HYPERPARAMETERS ───────────────────────────────────────
IMG_SIZE     = (224, 224)

BATCH_SIZE   = 32

EPOCHS       = 20

LR           = 1e-4

DROPOUT_RATE = 0.4

NUM_CLASSES  = 1


# ── 5. DATASET PATHS ─────────────────────────────────────────
CSV_PATH = "/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading.csv"

IMAGES_DIR = "/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading/DR_grading"

MODEL_PATH = "/kaggle/working/densenet121_ddr_best.h5"


# ── 6. LOAD CSV ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

print("\n[INFO] Dataset shape:")
print(df.shape)

print("\n[INFO] First 5 rows:")
print(df.head())


# ── 7. CHECK COLUMN NAMES ────────────────────────────────────
print("\n[INFO] Columns:")
print(df.columns)

# ============================================================
# CHANGE THESE COLUMN NAMES IF REQUIRED
# ============================================================
img_col = "id_code"
lbl_col = "diagnosis"


# ── 8. CREATE BINARY LABELS ──────────────────────────────────
df["label"] = df[lbl_col].apply(
    lambda x: "No_DR" if x == 0 else "DR"
)

print("\n[INFO] Binary class distribution:")
print(df["label"].value_counts())


# ── 9. CREATE FULL IMAGE PATHS ───────────────────────────────
df["filepath"] = df[img_col].apply(
    lambda x: os.path.join(IMAGES_DIR, x)
)

# ── 10. REMOVE MISSING FILES ─────────────────────────────────
df = df[df["filepath"].apply(os.path.exists)]

print(f"\n[INFO] Valid images found: {len(df)}")


# ── 11. TRAIN / VALID / TEST SPLIT ───────────────────────────
train_df, temp_df = train_test_split(

    df,

    test_size=0.30,

    stratify=df["label"],

    random_state=SEED,
)

val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    stratify=temp_df["label"],

    random_state=SEED,
)

print("\n[INFO] Split sizes:")
print(f"Train : {len(train_df)}")
print(f"Valid : {len(val_df)}")
print(f"Test  : {len(test_df)}")


# ── 12. IMAGE DATA GENERATORS ────────────────────────────────
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

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ── 13. DATAFRAME GENERATORS ─────────────────────────────────
train_gen = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    x_col="filepath",

    y_col="label",

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True,

    seed=SEED,
)

valid_gen = val_test_datagen.flow_from_dataframe(

    dataframe=val_df,

    x_col="filepath",

    y_col="label",

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False,
)

test_gen = val_test_datagen.flow_from_dataframe(

    dataframe=test_df,

    x_col="filepath",

    y_col="label",

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False,
)

print(f"\n[INFO] Train samples : {train_gen.samples}")

print(f"[INFO] Valid samples : {valid_gen.samples}")

print(f"[INFO] Test samples  : {test_gen.samples}")

print(f"\n[INFO] Class indices :")

print(train_gen.class_indices)

# ── 14. MODEL: DenseNet121 FROM SCRATCH ──────────────────────
def build_model(input_shape=(224, 224, 3)):

    backbone = DenseNet121(

        include_top=False,

        weights='imagenet',

        input_shape=input_shape,

        pooling=None,
    )

    inputs = keras.Input(
        shape=input_shape,
        name="input_image"
    )

    x = backbone(inputs, training=True)

    # GLOBAL AVERAGE POOLING
    x = layers.GlobalAveragePooling2D(
        name="gap"
    )(x)

    # CLASSIFICATION HEAD
    x = layers.BatchNormalization(
        name="bn_head"
    )(x)

    x = layers.Dropout(
        DROPOUT_RATE,
        name="dropout_1"
    )(x)

    x = layers.Dense(

        256,

        activation="relu",

        kernel_initializer="he_normal",

        name="dense_256",

    )(x)

    x = layers.BatchNormalization(
        name="bn_dense"
    )(x)

    x = layers.Dropout(
        DROPOUT_RATE / 2,
        name="dropout_2"
    )(x)

    outputs = layers.Dense(

        NUM_CLASSES,

        activation="sigmoid",

        name="output",

    )(x)

    model = Model(
        inputs,
        outputs,
        name="DenseNet121_DDR"
    )

    return model


model = build_model()

model.summary()

# ── 15. COMPILE MODEL ────────────────────────────────────────
model.compile(

    optimizer=Adam(
        learning_rate=LR
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc"),
    ],
)

# ── 16. CALLBACKS ────────────────────────────────────────────
callbacks = [

    EarlyStopping(

        monitor="val_auc",

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

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1,
    ),
]


# ── 17. TRAIN MODEL ──────────────────────────────────────────
print("\n[INFO] Starting training...\n")

history = model.fit(

    train_gen,

    epochs=EPOCHS,

    validation_data=valid_gen,

    callbacks=callbacks,

    verbose=1,
)

print("\n[INFO] Training complete.")

# ============================================================
# OPTION 1 — CONTINUE TRAINING / FINE-TUNING
# (FOR MODEL ALREADY TRAINED WITH weights=None)
# ============================================================

print("\n[INFO] Starting continued fine-tuning...\n")

# ── UNFREEZE ENTIRE MODEL ────────────────────────────────────
model.trainable = True

# OPTIONAL:
# Fine-tune only deeper layers
# (recommended for stability)

for layer in model.layers[:-30]:
    layer.trainable = False

# ── RECOMPILE WITH LOWER LEARNING RATE ───────────────────────
model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc"),
    ],
)

# ── CALLBACKS FOR FINE-TUNING ────────────────────────────────
fine_tune_callbacks = [

    EarlyStopping(

        monitor="val_auc",

        patience=5,

        mode="max",

        restore_best_weights=True,

        verbose=1,
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-7,

        verbose=1,
    ),

    ModelCheckpoint(

        filepath="/kaggle/working/densenet121_continued_best.h5",

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1,
    ),
]

# ── CONTINUE TRAINING ────────────────────────────────────────
fine_tune_history = model.fit(

    train_gen,

    epochs=20,

    validation_data=valid_gen,

    callbacks=fine_tune_callbacks,

    verbose=1,
)

print("\n[INFO] Continued fine-tuning complete.")

# ── COMBINE TRAINING HISTORIES ───────────────────────────────
for key in history.history.keys():

    history.history[key].extend(
        fine_tune_history.history[key]
    )

# ── SAVE FINAL MODEL ─────────────────────────────────────────
FINAL_MODEL_PATH = "/kaggle/working/densenet121_final_continued.h5"

model.save(FINAL_MODEL_PATH)

print(f"\n[INFO] Final model saved to:")
print(FINAL_MODEL_PATH)


# ── 18. TRAINING CURVES ──────────────────────────────────────
def plot_training_history(history):

    metrics = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "auc",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    axes = axes.flatten()

    for idx, metric in enumerate(metrics):

        ax = axes[idx]

        ax.plot(
            history.history[metric],
            label=f"Train {metric}",
            lw=2,
        )

        ax.plot(
            history.history[f"val_{metric}"],
            label=f"Val {metric}",
            lw=2,
            linestyle="--",
        )

        ax.set_title(
            metric.capitalize(),
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel("Epoch")

        ax.set_ylabel(metric.capitalize())

        ax.legend()

        ax.grid(alpha=0.3)

    axes[-1].set_visible(False)

    plt.suptitle(
        "DenseNet121 — Training History (DDR Dataset)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    plt.savefig(
        "/kaggle/working/training_history.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("[INFO] Training history plot saved.")


plot_training_history(history)

# ── 19. EVALUATE MODEL ───────────────────────────────────────
print("\n[INFO] Evaluating on test set...\n")

test_results = model.evaluate(
    test_gen,
    verbose=1
)

metric_names = [
    "loss",
    "accuracy",
    "precision",
    "recall",
    "auc",
]

print("\n── Test Metrics ──────────────────────────────")

for name, value in zip(metric_names, test_results):

    print(f"{name:12s}: {value:.4f}")


# ── 20. PREDICTIONS ──────────────────────────────────────────
test_gen.reset()

y_prob = model.predict(
    test_gen,
    verbose=1
).ravel()

y_pred = (y_prob >= 0.5).astype(int)

y_true = test_gen.classes


# ── 21. CONFUSION MATRIX ─────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=class_names,

        yticklabels=class_names,

        linewidths=0.5,

        ax=ax,
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12,
    )

    ax.set_ylabel(
        "True Label",
        fontsize=12,
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        "/kaggle/working/confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("[INFO] Confusion matrix saved.")


plot_confusion_matrix(
    y_true,
    y_pred,
    list(train_gen.class_indices.keys())
)



# ── 22. CLASSIFICATION REPORT ────────────────────────────────
print("\n── Classification Report ─────────────────────")

report = classification_report(

    y_true,

    y_pred,

    target_names=list(train_gen.class_indices.keys())
)

print(report)


# ── 23. ROC CURVE ────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(

        fpr,

        tpr,

        lw=2.5,

        label=f"ROC Curve (AUC = {roc_auc:.4f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],

        lw=1.5,

        linestyle="--",

        label="Random Classifier",
    )

    ax.fill_between(
        fpr,
        tpr,
        alpha=0.1,
    )

    ax.set_xlim([0.0, 1.0])

    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=12,
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=12,
    )

    ax.set_title(
        "Receiver Operating Characteristic (ROC) Curve",
        fontsize=13,
        fontweight="bold",
    )

    ax.legend(
        loc="lower right",
        fontsize=11,
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "/kaggle/working/roc_curve.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print(f"[INFO] ROC-AUC = {roc_auc:.4f}")

    return roc_auc


roc_auc_score = plot_roc_curve(
    y_true,
    y_prob
)

# ── 24. FINAL SUMMARY ────────────────────────────────────────
summary_df = pd.DataFrame({

    "Metric":
        metric_names + ["ROC-AUC (sklearn)"],

    "Value":
        [round(v, 4) for v in test_results]
        + [round(roc_auc_score, 4)],
})

print("\n── Final Summary ─────────────────────────────")

print(summary_df.to_string(index=False))

# ── 25. SAVE FINAL RESULTS ───────────────────────────────────
summary_df.to_csv(
    "/kaggle/working/final_metrics.csv",
    index=False
)

print("\n[INFO] Final metrics saved.")

print("\n[INFO] All done ")
