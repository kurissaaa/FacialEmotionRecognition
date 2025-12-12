import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.applications import MobileNet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns

# Paths
train_data_dir = r"D:\CSC583 PROJECT CODE\datasets (3)\datasets\train"
validation_data_dir = r"D:\CSC583 PROJECT CODE\datasets (3)\datasets\test"
model_save_path = r"D:\CSC582 PROJECT NEW\Emotion_detection_with_CNN-main\model\mobilenet_model.keras"

# Updated Class Labels
class_labels = ['angry', 'disgust', 'happy', 'neutral', 'surprise']
num_classes = len(class_labels)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
])

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical",
    shuffle=True,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_data_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical",
    shuffle=False,
)

# Apply data augmentation
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Load MobileNet with pre-trained weights
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Add custom classification layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_loss", mode="min")

# Train the model
epochs = 20
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stopping, checkpoint],
)

# Fine-tune the base model
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_epochs = 10
history_fine_tune = model.fit(
    train_ds,
    epochs=fine_tune_epochs,
    validation_data=val_ds,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stopping, checkpoint],
)

# Evaluate the model
predictions = model.predict(val_ds)
y_pred = np.argmax(predictions, axis=1)
y_true = np.concatenate([y.numpy().argmax(axis=1) for _, y in val_ds])

# Confusion Matrix and Metrics
conf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
