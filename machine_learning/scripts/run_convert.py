# scripts/run_convert.py
"""
Script to convert the trained Keras model to TensorFlow Lite and
generate a C header file for ESP32 deployment.
"""


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.convert import convert_to_tflite, tflite_to_c_array



# Convert Keras to TFLite
convert_to_tflite(quantize=True)

# Step 2: Generate C header file
tflite_to_c_array()