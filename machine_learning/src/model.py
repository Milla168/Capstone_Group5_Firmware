# src/model.py
"""
This module defines the neural network architecture for crochet stitch detection.
It provides a 1D Convolutional Neural Network (CNN) designed for binary classification
of IMU sensor time-series data.

Architecture:
    - 3 Conv1D layers with batch normalization and pooling
    - Global average pooling for dimensionality reduction
    - Dense layers with dropout for classification
    - Sigmoid activation for binary output

Functions:
    build_model: Constructs and returns the 1D-CNN Keras model

"""


import tensorflow as tf
from tensorflow.keras import layers, models
from configs.config import WINDOW_LENGTH, NUM_CHANNELS
import numpy as np



@tf.keras.utils.register_keras_serializable()
class StandardScalerLayer(layers.Layer):

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def call(self, inputs):
        return (inputs - self.mean) / self.std

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self.mean.tolist(),  
            "std":  self.std.tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):         
        config["mean"] = np.array(config["mean"], dtype=np.float32)
        config["std"]  = np.array(config["std"],  dtype=np.float32)
        return cls(**config)



def build_model(
    window_length=WINDOW_LENGTH,
    num_channels=NUM_CHANNELS,
    mean=None,
    std=None
):
    """
    Builds a 1D-CNN binary classifier for crochet stitch detection
    with optional embedded normalization.

    Args:
        window_length (int): Number of time steps
        num_channels (int): Number of sensor channels
        mean (np.ndarray): Feature-wise mean for normalization
        std (np.ndarray): Feature-wise std for normalization

    Returns:
        tf.keras.Model
    """

    inputs = layers.Input(shape=(window_length, num_channels))
    x = inputs

    if mean is not None and std is not None:
        x = StandardScalerLayer(mean, std, name="normalization")(x)

    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    return model