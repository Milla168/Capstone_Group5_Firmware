# src/train.py
"""
This module handles model training and evaluation for the crochet stitch detection pipeline.
It provides functions for:

1. Compiling and training the Keras model with class balancing
2. Applying callbacks for early stopping, learning rate reduction, and checkpointing
3. Evaluating model performance with classification metrics

Functions:
    train: Train the model with configured hyperparameters and callbacks
    evaluate: Evaluate model and display classification report and confusion matrix

"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from configs.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, KERAS_DIR



def train(model, X_train, y_train, X_val, y_val):
    """
    Trains the model with early stopping, learning rate reduction, and checkpointing

        Args:
            model (tensorflow.keras.Model): Compiled Keras model to train
            X_train (np.ndarray): Training data of shape (num_windows, window_length, num_channels)
            y_train (np.ndarray): Training labels of shape (num_windows,)
            X_val (np.ndarray): Validation data of shape (num_windows, window_length, num_channels)
            y_val (np.ndarray): Validation labels of shape (num_windows,)

        Returns:
            tensorflow.keras.callbacks.History: Training history containing loss and metrics per epoch
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weights = {0: weights[0], 1: weights[1]}
    print(f"Class weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(KERAS_DIR / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    return history



def evaluate(model, X, y, threshold=0.5, split_name=""):
    """
    Evaluates model performance and prints classification metrics

        Args:
            model (tensorflow.keras.Model): Trained Keras model to evaluate
            X (np.ndarray): Input data of shape (num_windows, window_length, num_channels)
            y (np.ndarray): Ground truth labels of shape (num_windows,)
            threshold (float): Classification threshold for converting probabilities to binary predictions
            split_name (str): Name of the data split for display purposes (e.g., "train", "validation", "test")

        Returns:
            tuple: (y_prob, y_pred) where y_prob is raw prediction probabilities
                and y_pred is binary predictions based on threshold
    """

    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    header = f"--- {split_name.upper()} EVALUATION (threshold={threshold}) ---"
    print(f"\n{header}")
    print(classification_report(y, y_pred, target_names=['Non-Stitch', 'Stitch']))

    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    return y_prob, y_pred