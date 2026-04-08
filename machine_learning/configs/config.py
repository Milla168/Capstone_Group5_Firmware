# configs/config.py
from pathlib import Path

# Base paths
BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DATA_DIR  = DATA_DIR / "raw"
ANNOTATED_DIR = DATA_DIR / "annotated"
MODELS_DIR    = BASE_DIR / "models"
KERAS_DIR     = MODELS_DIR / "keras"
TFLITE_DIR    = MODELS_DIR / "tflite"
SCALER_DIR    = MODELS_DIR / "scaler"
KERAS_MODEL = KERAS_DIR / "best_model.keras"
TFLITE_MODEL = TFLITE_DIR / "model.tflite"
C_HEADER_FILE = TFLITE_DIR / "model_data.h"

# Sensor
SAMPLING_RATE   = 100
SENSOR_COLUMNS  = ['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps']
NUM_CHANNELS    = len(SENSOR_COLUMNS)

# Preprocessing
CUTOFF_HZ     = 8
WINDOW_LENGTH = 200
STRIDE        = 100

# Training
EPOCHS        = 15
BATCH_SIZE    = 32
LEARNING_RATE = 0.001