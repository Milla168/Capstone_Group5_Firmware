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
from configs.config import ANNOTATED_DIR, KERAS_DIR, TFLITE_DIR, SCALER_DIR, SENSOR_COLUMNS, WINDOW_LENGTH, STRIDE
from src.preprocess import preprocess_split, normalize
from src.model import build_model
from src.train import train, evaluate
from src.visualize import plot_history, plot_confusion_matrix, plot_signals, plot_raw_vs_filtered, plot_signal_segment, plot_windows_segment


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

for d in [KERAS_DIR, TFLITE_DIR, SCALER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- PREPROCESS -------------
df_raw_train, df_filtered_train, X_train, y_train = preprocess_split(ANNOTATED_DIR / "train")
df_raw_val, df_filtered_val, X_val, y_val = preprocess_split(ANNOTATED_DIR / "validation")
df_raw_test, df_filtered_test, X_test, y_test = preprocess_split(ANNOTATED_DIR / "test")

df_raw_graph, df_filtered_graph, X_graph, y_graph = preprocess_split(ANNOTATED_DIR / "graphing")

# Normalize, fit on train only
X_train, X_val, X_test, scaler = normalize(X_train, X_val, X_test)

# Save scaler
joblib.dump(scaler, SCALER_DIR / 'scaler.pkl')
np.save(SCALER_DIR / 'scaler_mean.npy',  scaler.mean_)
np.save(SCALER_DIR / 'scaler_scale.npy', scaler.scale_)
np.save(SCALER_DIR / 'X_train_sample.npy', X_train[:500])
print("Scaler saved")

# ----------- TRAINING -------------
model = build_model()
model.summary()
history = train(model, X_train, y_train, X_val, y_val)
plot_history(history)

# Load best checkpoint
model = tf.keras.models.load_model(str(KERAS_DIR / 'best_model.keras'))

# -------- EVALUATE -------------
y_prob_val, y_pred_val = evaluate(model, X_val,  y_val,  split_name="validation")
y_prob_test, y_pred_test = evaluate(model, X_test, y_test, split_name="test")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_test)
plot_confusion_matrix(y_val, y_pred_val)


# ----------- GRAPHING SAMPLES ------------
# Create visualizations using dataset specific for graphs
# time = df_filtered_graph['time_ms'].values / 1000.0

# zoom_start = 46
# zoom_end = 76

# print(f"Visualizing segment: {zoom_start:.2f}s to {zoom_end:.2f}s")

# # Plot raw vs filtered for this segment
# plot_signal_segment(time, df_raw_graph, df_filtered_graph, SENSOR_COLUMNS, 
#                     start_s=zoom_start, end_s=zoom_end)

# window_start = 46
# window_end = 76

# plot_windows_segment(
#     df_filtered_graph,
#     signal_col='ax_g',
#     window_length=WINDOW_LENGTH,
#     stride=STRIDE,
#     start_time_s=window_start,
#     end_time_s=window_end
# )
