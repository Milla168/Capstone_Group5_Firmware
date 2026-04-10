# scripts/run_convert.py
"""
Converts the trained Keras model to TFLite and generates:
  - model.tflite       (the TFLite flatbuffer)
  - model_data.h       (C header for ESP32)
  - quantization_params.txt  (scale/zero_point notes for ESP32 firmware)

Run from the project root:
    python scripts/run_convert.py [--mode dynamic]

Modes:
    float32   No quantization. ~180 KB. Easiest to debug.
    dynamic   Dynamic-range quant. ~60 KB. Recommended default.
              Input/output stay float32 — simpler ESP32 code.
    int8      Full INT8. ~62 KB. Requires calibration data (X_cal.npy).
              Slightly more complex ESP32 firmware (scale/zero_point needed).
"""

import sys
import os
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import KERAS_MODEL, TFLITE_DIR, SCALER_DIR
from src.convert import (
    load_model,
    convert_to_tflite,
    save_tflite,
    save_c_header,
    print_quantization_params,
)


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite for ESP32")
    parser.add_argument(
        "--mode",
        choices=["float32", "dynamic", "int8"],
        default="dynamic",
        help="Quantization mode (default: dynamic)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {KERAS_MODEL} ...")
    model = load_model(KERAS_MODEL)
    model.summary()

    # ------------------------------------------------------------------
    # 2. Load calibration data (only needed for int8)
    # ------------------------------------------------------------------
    cal_data = None
    if args.mode == "int8":
        # run_pipeline.py saves X_train_cal.npy (already normalised)
        cal_path = SCALER_DIR / "X_train_cal.npy"
        if not cal_path.exists():
            print(
                f"\nERROR: Calibration file not found at {cal_path}\n"
                "Add this to run_pipeline.py after normalising X_train:\n"
                "    np.save(SCALER_DIR / 'X_train_cal.npy', X_train[:200])\n"
                "Or use --mode dynamic which needs no calibration data."
            )
            sys.exit(1)
        cal_data = np.load(cal_path)
        print(f"Loaded {len(cal_data)} calibration windows from {cal_path}")

    # ------------------------------------------------------------------
    # 3. Convert
    # ------------------------------------------------------------------
    print(f"\nConverting with mode='{args.mode}' ...")
    tflite_bytes = convert_to_tflite(model, mode=args.mode, calibration_data=cal_data)

    # ------------------------------------------------------------------
    # 4. Save artifacts
    # ------------------------------------------------------------------
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    save_tflite(tflite_bytes, TFLITE_DIR / "model.tflite")
    save_c_header(tflite_bytes, TFLITE_DIR / "model_data.h")

    # ------------------------------------------------------------------
    # 5. Print quantization params + write notes file
    # ------------------------------------------------------------------
    print_quantization_params(tflite_bytes)

    # Load mean/std so we can write them into the notes file
    mean_path = SCALER_DIR / "mean.npy"
    std_path  = SCALER_DIR / "std.npy"
    mean_str = str(np.load(mean_path).tolist()) if mean_path.exists() else "not found"
    std_str  = str(np.load(std_path).tolist())  if std_path.exists()  else "not found"

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    notes = f"""ESP32 Deployment Notes
======================
Quantization mode : {args.mode}
Model file        : model.tflite
C header          : model_data.h

Normalization (apply BEFORE feeding the model)
-----------------------------------------------
mean = {mean_str}
std  = {std_str}

For each window of shape (200, 6):
    norm[t][ch] = (raw[t][ch] - mean[ch]) / std[ch]

Input tensor
------------
dtype      : {inp['dtype'].__name__}
shape      : {list(inp['shape'])}
scale      : {inp['quantization'][0]}
zero_point : {inp['quantization'][1]}
"""

    if inp['dtype'] == np.int8:
        notes += f"""
For int8 mode, quantize each normalised float value:
    int8_val = (int8_t) clamp(round(norm_val / {inp['quantization'][0]:.8f}) + {inp['quantization'][1]}, -128, 127)

Output tensor
-------------
dtype      : {out['dtype'].__name__}
shape      : {list(out['shape'])}
scale      : {out['quantization'][0]}
zero_point : {out['quantization'][1]}

De-quantize output to probability:
    probability = ((float)int8_output - {out['quantization'][1]}) * {out['quantization'][0]:.8f}
    prediction  = probability >= 0.5 ? 1 (STITCH) : 0 (NON-STITCH)
"""
    else:
        notes += """
Output tensor
-------------
dtype  : float32
Output is already a probability in [0, 1].
    prediction = output >= 0.5 ? 1 (STITCH) : 0 (NON-STITCH)
"""

    notes_path = TFLITE_DIR / "esp32_deployment_notes.txt"
    notes_path.write_text(notes)
    print(f"\nDeployment notes saved -> {notes_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()



# # scripts/run_convert.py
# import sys
# import os
# import numpy as np
# import tensorflow as tf

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from configs.config import KERAS_MODEL, TFLITE_DIR, SCALER_DIR
# from src.convert import load_model, convert_to_tflite, save_tflite, save_c_header

# # Load model
# print("Loading model...")
# model = load_model(KERAS_MODEL)

# # Load calibration data (optional, for full int8)
# cal_data = None
# cal_path = SCALER_DIR / 'X_train_sample.npy'
# if cal_path.exists():
#     cal_data = np.load(cal_path)
#     print(f"Loaded {len(cal_data)} calibration samples")

# # Convert
# print("Converting to TFLite...")
# tflite_model = convert_to_tflite(model, quantize=True, representative_data=cal_data)
# print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

# # Save
# save_tflite(tflite_model, TFLITE_DIR / "model.tflite")
# save_c_header(tflite_model, TFLITE_DIR / "model_data.h")

# print(f"\nSaved to: {TFLITE_DIR}")

# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()[0]
# output_details = interpreter.get_output_details()[0]

# print("\n--- ESP32 Quantization Parameters ---")
# print(f"Input shape: {input_details['shape']}")
# print(f"Input scale: {input_details['quantization'][0]}")
# print(f"Input zero_point: {input_details['quantization'][1]}")
# print(f"Output scale: {output_details['quantization'][0]}")
# print(f"Output zero_point: {output_details['quantization'][1]}")

# print("Done!")





# # scripts/run_convert.py
# """
# Script to convert the trained Keras model to TensorFlow Lite and
# generate a C header file for ESP32 deployment.
# """


# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.convert import convert_to_tflite, tflite_to_c_array



# # Convert Keras to TFLite
# convert_to_tflite(quantize=False)

# # Step 2: Generate C header file
# tflite_to_c_array()