import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers
from tensorflow.keras.applications import (
    InceptionResNetV2, VGG16, VGG19, DenseNet121, MobileNetV2, EfficientNetV2L
)
from tensorflow.keras.applications import (
    inception_resnet_v2, vgg16, vgg19, densenet, mobilenet_v2, efficientnet_v2
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow:", tf.__version__)

# Kaggle paths
CSV_PATH = "/kaggle/input/ddrdataset/DR_grading.csv"
IMAGES_DIR = "/kaggle/input/ddrdataset/DR_grading/DR_grading"
OUT_DIR = "/kaggle/working"

# Labels (human readable, optional)
CLASSES = ['No-DR', 'Mild', 'Moderate', 'Severe', 'Proliferative-DR']
NUM_CLASSES = 5
BATCH_SIZE = 32
SEED = 42

# Add stage configs and fine-tune depth
MODEL_CONFIGS = {
    'InceptionResNetV2': {
        'base': InceptionResNetV2, 'preprocess': inception_resnet_v2.preprocess_input,
        'input_size': (299, 299),
        'fine_tune_at': 100, 'stage1_epochs': 25, 'stage2_epochs': 25,
        'stage1_lr': 1e-3, 'stage2_lr': 1e-5
    },
    'VGG16': {
        'base': VGG16, 'preprocess': vgg16.preprocess_input,
        'input_size': (224, 224),
        'fine_tune_at': 15, 'stage1_epochs': 25, 'stage2_epochs': 25,
        'stage1_lr': 1e-3, 'stage2_lr': 1e-5
    },
    'VGG19': {
        'base': VGG19, 'preprocess': vgg19.preprocess_input,
        'input_size': (224, 224),
        'fine_tune_at': 20, 'stage1_epochs': 25, 'stage2_epochs': 25,
        'stage1_lr': 1e-3, 'stage2_lr': 1e-5
    },
    'DenseNet121': {
        'base': DenseNet121, 'preprocess': densenet.preprocess_input,
        'input_size': (224, 224),
        'fine_tune_at': 300, 'stage1_epochs': 20, 'stage2_epochs': 20,
        'stage1_lr': 1e-3, 'stage2_lr': 1e-5
    },
    'MobileNetV2': {
        'base': MobileNetV2, 'preprocess': mobilenet_v2.preprocess_input,
        'input_size': (224, 224),
        'fine_tune_at': 100, 'stage1_epochs': 20, 'stage2_epochs': 20,
        'stage1_lr': 1e-3, 'stage2_lr': 1e-5
    },
    'EfficientNetV2L': {
        'base': EfficientNetV2L, 'preprocess': efficientnet_v2.preprocess_input,
        'input_size': (480, 480),
        'fine_tune_at': 400, 'stage1_epochs': 15, 'stage2_epochs': 15,
        'stage1_lr': 1e-3, 'stage2_lr': 1e-5
    }
}

# Heuristics for CSV columns
POSSIBLE_IMG_COLS = ['image', 'image_name', 'filename', 'file', 'img', 'image_id', 'id_code', 'id', 'name']
POSSIBLE_LBL_COLS = ['DR_grade', 'level', 'diagnosis', 'label', 'class', 'grade', 'dr_level']

def guess_columns(df):
    lower2orig = {c.lower(): c for c in df.columns}
    img_col = next((lower2orig[c] for c in POSSIBLE_IMG_COLS if c in lower2orig), None)
    lbl_col = next((lower2orig[c] for c in POSSIBLE_LBL_COLS if c in lower2orig), None)
    if img_col is None or lbl_col is None:
        raise ValueError(f"Could not detect image/label columns, CSV columns: {list(df.columns)}")
    return img_col, lbl_col

def build_dataframe(csv_path, images_dir):
    df = pd.read_csv(csv_path, dtype=str)
    img_col, lbl_col = guess_columns(df)
    df[lbl_col] = df[lbl_col].str.strip()
    try:
        df[lbl_col] = df[lbl_col].astype(int)
    except Exception:
        mapping = {k: i for i, k in enumerate(sorted(df[lbl_col].unique()))}
        df[lbl_col] = df[lbl_col].map(mapping).astype(int)
    uniq = sorted(df[lbl_col].unique())
    if uniq and uniq[0] == 1 and uniq[-1] == 5:
        df[lbl_col] = df[lbl_col] - 1
    def to_path(name):
        name = str(name).strip()
        path = os.path.join(images_dir, name)
        if os.path.isfile(path):
            return path
        base, ext = os.path.splitext(name)
        if ext == '':
            cand = os.path.join(images_dir, base + ".jpg")
            if os.path.isfile(cand):
                return cand
        return None
    df['filepath'] = df[img_col].apply(to_path)
    before = len(df)
    df = df.dropna(subset=['filepath']).reset_index(drop=True)
    after = len(df)
    if after == 0:
        raise RuntimeError("No image files matched CSV entries; check names/paths/extensions")
    print(f"Matched files: {after} / {before}")
    return df, img_col, lbl_col

def make_generators(df_train, df_val, lbl_col, target_size, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filepath',
        y_col=lbl_col,
        target_size=target_size,
        color_mode='rgb',
        class_mode='raw',   # integer labels streamed as-is (sparse loss)
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col='filepath',
        y_col=lbl_col,
        target_size=target_size,
        color_mode='rgb',
        class_mode='raw',
        batch_size=batch_size,
        shuffle=False
    )
    return train_gen, val_gen

def create_frozen_model(model_name='VGG19', num_classes=5):
    cfg = MODEL_CONFIGS[model_name]
    in_h, in_w = cfg['input_size']
    inputs = tf.keras.Input(shape=(in_h, in_w, 3))
    x = layers.Lambda(lambda z: z * 255.0)(inputs)
    x = layers.Lambda(cfg['preprocess'])(x)
    # Name the base so we can retrieve it later
    base = cfg['base'](weights='imagenet', include_top=False, input_shape=(in_h, in_w, 3), name='backbone')
    base.trainable = False
    # Keep BN in inference mode during both stages
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def attach_unfreeze_method(model, backbone_name='backbone'):
    # Adds model.unfreeze_layers(fine_tune_at)
    def _unfreeze_layers(fine_tune_at):
        base = model.get_layer(backbone_name)
        base.trainable = True
        n = len(base.layers)
        k = max(0, min(int(fine_tune_at), n))
        for i, layer in enumerate(base.layers):
            # Keep BatchNorm frozen; fine-tune other layers from k onward
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = (i >= k)
        print(f"Unfroze backbone from layer index {k} of {n} (BatchNorm layers remain frozen).")
    model.unfreeze_layers = _unfreeze_layers

# Backbone choice
model_name = 'VGG19'  # try: 'VGG16','VGG19','DenseNet121','MobileNetV2','EfficientNetV2L'
cfg = MODEL_CONFIGS[model_name]
target_size = cfg['input_size']

# Build dataframe from Kaggle CSV and folder
df, img_col, lbl_col = build_dataframe(CSV_PATH, IMAGES_DIR)

# Stratified split on labels
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df[lbl_col])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df[lbl_col])
print(f"Rows -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# Generators (streaming, no preload)
train_gen, val_gen = make_generators(train_df, val_df, lbl_col, target_size, BATCH_SIZE)

# Build model
print(f"🚀 Building {model_name} (frozen backbone)")
model = create_frozen_model(model_name, NUM_CLASSES)
attach_unfreeze_method(model)  # add model.unfreeze_layers(...)

# --- STAGE 1: Feature Extraction ---
print("🎯 STAGE 1: Feature Extraction")
model.compile(
    optimizer=Adam(learning_rate=cfg['stage1_lr']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history_stage1 = model.fit(
    train_gen,
    epochs=cfg['stage1_epochs'],
    validation_data=val_gen,
    verbose=1
)


# --- STAGE 2: Fine-tuning ---
print("🎯 STAGE 2: Fine-tuning")
model.unfreeze_layers(cfg['fine_tune_at'])
# Re-compile is required after changing trainable flags
model.compile(
    optimizer=Adam(learning_rate=cfg['stage2_lr']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history_stage2 = model.fit(
    train_gen,
    epochs=cfg['stage2_epochs'],
    validation_data=val_gen,
    verbose=1
)
print("🎉 MULTISTAGE TRAINING COMPLETED!")

# --- TRAINING CURVE PLOTTING ---
def plot_history(history, title_prefix="Model"):
    h = history.history
    acc_key = 'accuracy' if 'accuracy' in h else next((k for k in h if not k.startswith('val') and k.endswith('accuracy')), None)
    val_acc_key = 'val_accuracy' if 'val_accuracy' in h else next((k for k in h if k.startswith('val') and k.endswith('accuracy')), None)
    loss_key, val_loss_key = 'loss', 'val_loss'
    epochs = range(1, len(h[loss_key]) + 1)
    acc = h.get(acc_key, None); val_acc = h.get(val_acc_key, None)
    loss = h[loss_key]; val_loss = h[val_loss_key]

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    if acc is not None and val_acc is not None:
        plt.plot(epochs, acc, label='Train Acc')
        plt.plot(epochs, val_acc, label='Val Acc')
        plt.title(f'{title_prefix} Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No accuracy metric in history", ha='center', va='center')
        plt.title(f'{title_prefix} Accuracy (N/A)')
        plt.axis('off')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"{title_prefix.replace(' ', '_').lower()}_training_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved curves to: {fig_path}")

# Plot both stages
plot_history(history_stage1, title_prefix=f"{model_name} Stage 1 (Frozen)")
plot_history(history_stage2, title_prefix=f"{model_name} Stage 2 (Fine-tune)")

# --- Evaluation on test split (streaming) ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col=lbl_col,
    target_size=target_size,
    color_mode='rgb',
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False
)
y_true = test_gen.labels.astype(int)
y_prob = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'{model_name} - Confusion Matrix')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.show()


# Save to Kaggle working directory
save_path = os.path.join(OUT_DIR, f"{model_name}_dr_kaggle_multistage.h5")
model.save(save_path)
print(f"Saved model to: {save_path}")