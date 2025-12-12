import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns

# Ignore warnings from Keras
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Directories for training and validation data
train_data_dir = r"D:\CSC583 PROJECT CODE\datasets-sop\datasets\train"  # Update with your training dataset path
validation_data_dir = r"D:\CSC583 PROJECT CODE\datasets-sop\datasets\test" # Update with your validation dataset path

# Updated Class labels
class_labels = ['angry', 'disgust', 'happy', 'neutral', 'surprise']

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest",
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode="grayscale",
    target_size=(48, 48),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode="grayscale",
    target_size=(48, 48),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

# Improved CNN Model
model = Sequential([
    Input(shape=(48, 48, 1)),  # Correct input shape for grayscale images
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(len(class_labels), activation="softmax"),  # Updated for 5 classes
])

# Compile the model with a learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

# Callbacks for Early Stopping and Model Checkpoint
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    filepath=r"D:\CSC582 PROJECT NEW\Emotion_detection_with_CNN-main\model\best_model.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

# Training the model
epochs = 50
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    #callbacks=[early_stopping, checkpoint],
)

# Save the final model
model_path = r"D:\CSC582 PROJECT NEW\Emotion_detection_with_CNN-main\model\final_emotion_model.keras"
model.save(model_path)

# Predictions and true labels
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
y_true = validation_generator.classes  # True class indices

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plotting Confusion Matrix
df_cm = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Calculate Precision, Recall, F1-Score
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot Training and Validation Accuracy/Loss
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()
