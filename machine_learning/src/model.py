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


def build_model(window_length=WINDOW_LENGTH, num_channels=NUM_CHANNELS):
    """
    Builds a 1D-CNN binary classifier for crochet stitch detection

        Args:
            window_length (int): Number of time steps in each input window
            num_channels (int): Number of sensor channels (features) per time step

        Returns:
            tensorflow.keras.Model: Compiled Sequential model with Conv1D layers,
                batch normalization, pooling, and dense layers for binary classification
    """
    model = models.Sequential([
        layers.Input(shape=(window_length, num_channels)),

        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model