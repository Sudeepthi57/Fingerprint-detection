import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ============================================
# PATHS
# ============================================
DATASET_PATH = r"C:\Users\sande\Downloads\archive (7)\SOCOFing\SOCOFing"
MODEL_PATH = "fingerprint_model.h5"

# ============================================
# LOAD MODEL IF ALREADY TRAINED
# ============================================
if os.path.exists(MODEL_PATH):
    print("✅ Model found. Loading existing model...")
    model = load_model(MODEL_PATH)
    TRAIN = False
else:
    print("❌ Model not found. Training new model...")
    TRAIN = True

# ============================================
# IMAGE DATA GENERATOR
# ============================================


train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ============================================
# BUILD MODEL (ONLY IF REQUIRED)
# ============================================
if TRAIN:
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # ============================================
    # CALLBACKS
    # ============================================
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    # ============================================
    # TRAIN MODEL
    # ============================================
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[early_stop, checkpoint]
    )

    # ============================================
    # PLOT ACCURACY & LOSS
    # ============================================
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

# ============================================
# PREDICT FUNCTION
# ============================================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)[0][0]

    if pred > 0.5:
        return "🟥 Altered Image"
    else:
        return "🟩 Real Image"

# ============================================
# EXAMPLE PREDICTION
# ============================================
print("\nPrediction 1:", 
      predict_image(r"C:\Users\sande\Downloads\archive (7)\SOCOFing\SOCOFing\Real\1__M_Left_index_finger.BMP"))

print("Prediction 2:", 
      predict_image(r"C:\Users\sande\Downloads\archive (7)\SOCOFing\SOCOFing\Altered\Altered-Easy\1__M_Left_index_finger_CR.BMP"))


