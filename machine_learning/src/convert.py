# src/convert.py
"""
Converts a trained Keras model to TFLite and generates deployment artifacts
for ESP32/TinyML.

Three quantization modes are supported:
  - float32   : No quantization. Largest file (~180 KB), easiest to debug.
  - dynamic   : Weights quantized to INT8, activations stay float32 at runtime.
                Good balance — no calibration data needed (~60 KB).
  - int8       : Full integer quantization. Requires a small calibration dataset.
                Smallest and fastest on MCUs (~62 KB), but input/output are INT8
                so the ESP32 must apply the reported scale/zero_point.

Key fix from original:
  The original convert.py tried to use a custom StandardScalerLayer that was
  commented out of model.py. Normalization is now done outside the model
  (mean/std saved as .npy files), so no custom objects are needed when loading.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load a saved Keras model. No custom objects needed since normalization
    is handled externally (mean.npy / std.npy)."""
    return tf.keras.models.load_model(str(model_path), compile=False)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_tflite(model, mode="dynamic", calibration_data=None):
    """
    Convert a Keras model to a TFLite flatbuffer.

    Args:
        model: tf.keras.Model
        mode:  "float32"  - no quantization
               "dynamic"  - dynamic-range quantization (recommended default)
               "int8"     - full INT8 quantization (requires calibration_data)
        calibration_data: np.ndarray of shape (N, WINDOW_LENGTH, NUM_CHANNELS),
                          already normalised. Required only for mode="int8".

    Returns:
        bytes: TFLite flatbuffer
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if mode == "float32":
        pass  # no extra settings needed

    elif mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Weights are INT8; activations quantized at inference time.
        # Input/output remain float32, so the ESP32 side is simpler.

    elif mode == "int8":
        if calibration_data is None:
            raise ValueError(
                "mode='int8' requires calibration_data (a sample of normalised "
                "training windows so the converter can determine activation ranges)."
            )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_gen():
            n = min(200, len(calibration_data))
            for i in range(n):
                yield [calibration_data[i : i + 1].astype(np.float32)]

        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'float32', 'dynamic', or 'int8'.")

    return converter.convert()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_tflite(tflite_bytes, path):
    """Write the TFLite flatbuffer to a .tflite file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tflite_bytes)
    print(f"Saved .tflite ({len(tflite_bytes)/1024:.1f} KB) -> {path}")


def save_c_header(tflite_bytes, path, array_name="model_data"):
    """
    Write a C header file containing the model as a byte array.
    Include this header in your ESP32 Arduino/ESP-IDF project.

    Usage in C++:
        #include "model_data.h"
        // model_data[]     - the flatbuffer bytes
        // model_data_len   - number of bytes
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hex_vals = [f"0x{b:02x}" for b in tflite_bytes]
    rows = []
    for i in range(0, len(hex_vals), 12):
        rows.append("  " + ", ".join(hex_vals[i : i + 12]))

    content = (
        f"#ifndef MODEL_DATA_H\n"
        f"#define MODEL_DATA_H\n\n"
        f"// Generated TFLite model - include in your ESP32 sketch\n"
        f"alignas(8) const unsigned char {array_name}[] = {{\n"
        + ",\n".join(rows)
        + f"\n}};\n\n"
        f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n\n"
        f"#endif  // MODEL_DATA_H\n"
    )
    path.write_text(content)
    print(f"Saved C header -> {path}")


# ---------------------------------------------------------------------------
# Inspection helper
# ---------------------------------------------------------------------------

def print_quantization_params(tflite_bytes):
    """
    Print the scale and zero_point for the model's input and output tensors.

    For mode='int8' you MUST use these values on the ESP32 to convert your
    normalised float window into the INT8 values the model expects:

        int8_val = (int8_t) round(float_val / input_scale) + input_zero_point

    And to interpret the INT8 output as a probability [0, 1]:

        probability = (int8_output - output_zero_point) * output_scale
        prediction  = probability >= 0.5 ? STITCH : NON_STITCH
    """
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    print("\n--- TFLite Tensor Info ---")
    print(f"  Input  dtype={inp['dtype'].__name__:6s}  shape={list(inp['shape'])}  "
          f"scale={inp['quantization'][0]:.8f}  zero_point={inp['quantization'][1]}")
    print(f"  Output dtype={out['dtype'].__name__:6s}  shape={list(out['shape'])}  "
          f"scale={out['quantization'][0]:.8f}  zero_point={out['quantization'][1]}")

    if inp['dtype'] == np.int8:
        print("\n  ESP32 pre-processing (per sample, per channel):")
        print("    1. Subtract mean[ch]  (from mean.npy)")
        print("    2. Divide by std[ch]  (from std.npy)")
        print(f"    3. int8 = (int8_t)(norm_val / {inp['quantization'][0]:.6f}"
              f" + {inp['quantization'][1]})")
        print(f"\n  ESP32 post-processing:")
        print(f"    prob = (int8_output - ({out['quantization'][1]})) "
              f"* {out['quantization'][0]:.6f}")
        print(f"    prediction = prob >= 0.5 ? 1 (STITCH) : 0 (NON-STITCH)")


# # src/convert.py
# """
# Helper methods for converting a Keras model to TFLite and generating
# a C header file for TinyML deployment on ESP32.
# """

# import tensorflow as tf
# from pathlib import Path
# from configs.config import KERAS_MODEL, TFLITE_MODEL, C_HEADER_FILE
# from src.model import StandardScalerLayer



# def convert_to_tflite(quantize=False):
#     keras_model = tf.keras.models.load_model(
#         KERAS_MODEL,
#         compile=False,
#         custom_objects={"StandardScalerLayer": StandardScalerLayer}
#     )

#     converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

#     if quantize:
#         # full integer quantization for microcontrollers
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]

#         # Representative dataset generator
#         def representative_data_gen():
#             # Replace this with a few windows from your training set
#             import numpy as np
#             for _ in range(100):
#                 dummy_input = np.random.rand(1, 200, 6).astype(np.float32)
#                 yield [dummy_input]

#         converter.representative_dataset = representative_data_gen

#         # Force all ops to INT8
#         converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#         converter.inference_input_type = tf.int8
#         converter.inference_output_type = tf.int8

#     tflite_model = converter.convert()
#     TFLITE_MODEL.write_bytes(tflite_model)
#     print(f"\nTFLite model saved to {TFLITE_MODEL}")



# def tflite_to_c_array(array_name="model_data"):
#     """
#     Converts a .tflite file to a C header file containing a byte array.

#     Args:
#         array_name (str): Name of the C array

#     Returns:
#         None
#     """

#     C_HEADER_FILE.parent.mkdir(parents=True, exist_ok=True)

#     with open(TFLITE_MODEL, "rb") as f:
#         tflite_bytes = f.read()

#     with open(C_HEADER_FILE, "w") as f:
#         f.write(f"//Generated from {TFLITE_MODEL.name}\n")
#         f.write(f"const unsigned char {array_name}[] = {{\n")

#         for i, byte in enumerate(tflite_bytes):
#             if i % 12 == 0:
#                 f.write("    ")
#             f.write(f"0x{byte:02x}, ")
#             if (i + 1) % 12 == 0:
#                 f.write("\n")

#         f.write("\n};\n")
#         f.write(f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n")

#     print(f"\nC header file saved to {C_HEADER_FILE}")