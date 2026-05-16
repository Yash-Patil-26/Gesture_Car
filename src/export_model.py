# src/export_model.py
# ─────────────────────────────────────────────────────────────
# Convert trained sklearn Random Forest → ONNX format.
# ONNX runs in browsers via onnxruntime-web.
#
# Run: python src/export_model.py
# Output: web/model.onnx + web/labels.json
# ─────────────────────────────────────────────────────────────

import os
import sys
import pickle
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_FILE, ENCODER_FILE, BASE_DIR

WEB_DIR    = os.path.join(BASE_DIR, "docs")
ONNX_FILE  = os.path.join(WEB_DIR, "model.onnx")
LABELS_FILE= os.path.join(WEB_DIR, "labels.json")


def main():
    print("Converting model to ONNX...")

    # Install skl2onnx if needed
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("Installing skl2onnx...")
        os.system("pip install skl2onnx onnx")
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

    os.makedirs(WEB_DIR, exist_ok=True)

    # Load model
    with open(MODEL_FILE,   'rb') as f: model   = pickle.load(f)
    with open(ENCODER_FILE, 'rb') as f: encoder = pickle.load(f)

    # Convert to ONNX
    # Input: float32 array of shape [1, 63]
    initial_type = [('float_input', FloatTensorType([None, 63]))]
    onnx_model   = convert_sklearn(
        model,
        initial_types = initial_type,
        target_opset  = 12
    )

    with open(ONNX_FILE, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    # Save label mapping
    labels = {str(i): cls for i, cls in enumerate(encoder.classes_)}
    with open(LABELS_FILE, 'w') as f:
        json.dump({
            "labels":    list(encoder.classes_),
            "id_to_label": labels
        }, f, indent=2)

    size_kb = os.path.getsize(ONNX_FILE) / 1024
    print(f"ONNX model saved → {ONNX_FILE}  ({size_kb:.0f} KB)")
    print(f"Labels saved     → {LABELS_FILE}")
    print(f"Classes: {list(encoder.classes_)}")
    print("\nNext: deploy web/ folder to Render.com")


if __name__ == "__main__":
    main()