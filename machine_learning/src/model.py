# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from configs.config import WINDOW_LENGTH, NUM_CHANNELS


def build_model(window_length=WINDOW_LENGTH, num_channels=NUM_CHANNELS):
    """1D-CNN binary classifier for crochet stitch detection."""
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