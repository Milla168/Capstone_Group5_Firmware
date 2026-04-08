# src/convert.py
"""
Helper methods for converting a Keras model to TFLite and generating
a C header file for TinyML deployment on ESP32.
"""

import tensorflow as tf
from pathlib import Path
from configs.config import KERAS_MODEL, TFLITE_MODEL, C_HEADER_FILE
from src.model import StandardScalerLayer



def convert_to_tflite(quantize=False):
    """
    Converts a Keras model to TensorFlow Lite format.

    Args:
        quantize (bool): If True, applies full integer quantization

    Returns:
        None
    """
    keras_model = tf.keras.models.load_model(
        KERAS_MODEL,
        compile=False,
        custom_objects={"StandardScalerLayer": StandardScalerLayer}
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if quantize:
        # full integer quantization for microcontrollers
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    TFLITE_MODEL.write_bytes(tflite_model)
    print(f"\nTFLite model saved to {TFLITE_MODEL}")



def tflite_to_c_array(array_name="model_data"):
    """
    Converts a .tflite file to a C header file containing a byte array.

    Args:
        array_name (str): Name of the C array

    Returns:
        None
    """

    C_HEADER_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TFLITE_MODEL, "rb") as f:
        tflite_bytes = f.read()

    with open(C_HEADER_FILE, "w") as f:
        f.write(f"Generated from {TFLITE_MODEL.name}\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")

        for i, byte in enumerate(tflite_bytes):
            if i % 12 == 0:
                f.write("    ")
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")

        f.write("\n};\n")
        f.write(f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n")

    print(f"\nC header file saved to {C_HEADER_FILE}")