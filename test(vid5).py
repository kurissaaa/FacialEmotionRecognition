import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import tensorflow.keras.backend
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.optimizers import Adam



# Ignore warnings from Keras
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Directories for training and validation data
train_data_dir = r"D:\CSC583 PROJECT CODE\datasets (3)\datasets\train"  # Update with your training dataset path
validation_data_dir = r"D:\CSC583 PROJECT CODE\datasets (3)\datasets\test"  # Update with your validation dataset path

# Updated Class labels
class_labels = ['angry', 'disgust', 'happy', 'neutral', 'surprise']

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                        validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode="grayscale",
    target_size=(48, 48),
    subset='training',
    batch_size=64,
    class_mode="categorical",
    #shuffle=True,
)

validation_generator = validation_datagen.flow_from_directory(
    train_data_dir,
    color_mode="grayscale",
    target_size=(48, 48),
    batch_size=64,
    class_mode="categorical",
    subset='validation',
    #shuffle=False,
)

test_dataset = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(48,48),
    class_mode="categorical",
    subset='validation',
    batch_size=64,)

base_model=tf.keras.applications.EfficientNetB0(
    input_shape=(48,48,3),
    include_top=False,
    weights="imagenet")

for layer in base_model_layers[:-4]:
    layer.trainable=False
    

# Improved CNN Model
model=Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(7,activation='softmax'))

print(model.summary())

# Compile the model with a learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
        f1_score,
]

lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 20,verbose = 1,factor = 0.50, min_lr = 1e-10)

mcp = ModelCheckpoint(r"D:\CSC582 PROJECT NEW\Emotion_detection_with_CNN-main\model\emotion_model.h5")

es = EarlyStopping(verbose=1, patience=20)

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=METRICS)

history=model.fit(train_dataset,validation_data=valid_dataset,epochs = 50,verbose = 1,callbacks=[lrd,mcp,es])


  


