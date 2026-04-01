# scripts/run_pipeline.py
"""
This file is a script that runs the ML model pipeline.
It takes methods from other files and follows the sequence:

1. Load the train, validation, and test datasets
2. Preprocess each dataset
3. Train the ML mode
4. Save the best performing model
5. Evaluate the models performance

To run this script, use command:

python run_script.py

"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import joblib
import os
from configs.config import ANNOTATED_DIR, KERAS_DIR, TFLITE_DIR, SCALER_DIR
from src.preprocess import preprocess_split, normalize
from src.model import build_model
from src.train import train, evaluate
from src.visualize import plot_history, plot_confusion_matrix


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

for d in [KERAS_DIR, TFLITE_DIR, SCALER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Preprocess
_, _, X_train, y_train = preprocess_split(ANNOTATED_DIR / "train")
_, _, X_val,   y_val   = preprocess_split(ANNOTATED_DIR / "validation")
_, _, X_test,  y_test  = preprocess_split(ANNOTATED_DIR / "test")

# Normalize, fit on train only
X_train, X_val, X_test, scaler = normalize(X_train, X_val, X_test)

# Save scaler
joblib.dump(scaler, SCALER_DIR / 'scaler.pkl')
np.save(SCALER_DIR / 'scaler_mean.npy',  scaler.mean_)
np.save(SCALER_DIR / 'scaler_scale.npy', scaler.scale_)
np.save(SCALER_DIR / 'X_train_sample.npy', X_train[:500])
print("Scaler saved")

# Train
model = build_model()
model.summary()
history = train(model, X_train, y_train, X_val, y_val)
plot_history(history)

# Load best checkpoint
model = tf.keras.models.load_model(str(KERAS_DIR / 'best_model.keras'))

# Evaluate
evaluate(model, X_val,  y_val,  split_name="validation")
evaluate(model, X_test, y_test, split_name="test")

# _, y_pred_test = evaluate(model, X_test, y_test, split_name="test")
# plot_confusion_matrix(y_test, y_pred_test)



# raw_signals = {
#     col: {'Raw': data_raw[col].values}
#     for col in SENSOR_COLUMNS
# }

# filtered_signals = {
#     col: {'Filtered': data_filtered[col].values}
#     for col in SENSOR_COLUMNS
# }

# plot_signals(time, raw_signals, title="Raw IMU Signals")
# plot_signals(time, filtered_signals, title="Filtered IMU Signals")
# plot_raw_vs_filtered(time, data_raw, data_filtered, SENSOR_COLUMNS)
# plot_zoomed_segment(time, data_raw, data_filtered, SENSOR_COLUMNS, 6000, 6300)
# plot_windows_zoomed(
#     data_filtered,
#     signal_col='ax_g',
#     window_length=WINDOW_LENGTH,
#     stride=STRIDE,
#     start_time_s=6000,
#     end_time_s=6300
# )

