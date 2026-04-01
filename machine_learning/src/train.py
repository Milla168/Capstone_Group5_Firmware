# src/train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from configs.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, KERAS_DIR


def train(model, X_train, y_train, X_val, y_val):
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