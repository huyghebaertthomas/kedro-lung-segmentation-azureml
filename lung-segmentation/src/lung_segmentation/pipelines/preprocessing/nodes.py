import PIL

import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split

def resize_images(images: Dict[str, PIL.Image.Image]) -> Dict[str, PIL.Image.Image]:
    for filename, image in images.items():
        images[filename] = image.resize((256,256))
    return images

def convert_to_grayscale(images: Dict[str, PIL.Image.Image]) -> Dict[str, PIL.Image.Image]:
    for filename, image in images.items():
        images[filename] = image.convert('L')
    return images

def normalize_images(images: Dict[str, PIL.Image.Image]) -> Dict[str, np.ndarray]:
    for filename, image in images.items():
        images[filename] = np.array(image)/255.0
    return images

def group_to_numpy(images: Dict[str, np.ndarray]) -> np.ndarray:
    return np.array(list(images.values()))

def feature_target_split(X_dataset: np.ndarray, y_dataset: np.ndarray, split_ratio: float) -> Dict[str, np.ndarray]:
    print(f"Train test split ratio: {split_ratio}")
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=split_ratio, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

#############################################################################################################


def unet(input_shape=(256, 256, 1)):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    # Bottleneck
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    # Decoder
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = layers.Conv2D(128, 2, activation='relu', padding='same')(up4)
    merge4 = layers.Concatenate(axis=3)([conv2, up4])
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.Conv2D(64, 2, activation='relu', padding='same')(up5)
    merge5 = layers.Concatenate(axis=3)([conv1, up5])
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    # Model compilation
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(datasets: Dict[str, np.ndarray], model_params: Dict[str, Any]) -> keras.models.Model:
    try:
        batch_size = model_params["batch_size"]
        epochs = model_params["epochs"]
    except KeyError as e:
        raise KeyError(f"Missing key {e} in model_params. Please provide a valid argument.")
    X_train, X_test, y_train, y_test = datasets["X_train"], datasets["X_test"], datasets["y_train"], datasets["y_test"]
    model = unet()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model