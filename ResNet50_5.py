import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("GPU:", tf.config.list_physical_devices('GPU'))


CSV_PATH = "/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading.csv"
IMAGES_DIR = "/kaggle/input/datasets/mariaherrerot/ddrdataset/DR_grading/DR_grading"
OUT_DIR = "/kaggle/working/"


df = pd.read_csv(CSV_PATH)

print("Columns:", df.columns)
df.head()


possible_img_cols = ['image', 'image_name', 'id_code', 'filename', 'img']
possible_lbl_cols = ['DR_grade', 'level', 'diagnosis', 'grade', 'label']

img_col = next(c for c in possible_img_cols if c in df.columns)
lbl_col = next(c for c in possible_lbl_cols if c in df.columns)

print("Image column:", img_col)
print("Label column:", lbl_col)


df['filepath'] = df[img_col].apply(
    lambda x: os.path.join(IMAGES_DIR, str(x))
)

df[lbl_col] = df[lbl_col].astype(int)

print("Total images:", len(df))


train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df[lbl_col], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df[lbl_col], random_state=42
)

print(len(train_df), len(val_df), len(test_df))


BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
).flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col=lbl_col,
    target_size=TARGET_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    val_df,
    x_col='filepath',
    y_col=lbl_col,
    target_size=TARGET_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE
)


NUM_CLASSES = 5

inputs = tf.keras.Input(shape=(224,224,3))

x = tf.keras.layers.Lambda(lambda z: z * 255.0)(inputs)
x = tf.keras.layers.Lambda(resnet.preprocess_input)(x)

base = ResNet50(weights='imagenet', include_top=False)
base.trainable = False

x = base(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


history1 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)


base.trainable = True

for layer in base.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history2 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)


test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col=lbl_col,
    target_size=TARGET_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = test_gen.labels.astype(int)
y_pred = np.argmax(model.predict(test_gen), axis=1)

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))


# Combine histories
train_acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.show()


train_loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

classes = ['No-DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import accuracy_score

test_acc = accuracy_score(y_true, y_pred)
print("Test Accuracy:", test_acc)