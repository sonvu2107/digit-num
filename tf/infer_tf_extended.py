"""
infer_tf_extended.py - Inference CNN 13 classes (0-9 + A, B, C)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from preprocess import preprocess_digit_from_bgr

# Mapping labels
LABEL_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C"]
NUM_CLASSES = 13

# Đường dẫn model mới
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_digit_letter_tf.keras")

# Load model
_model = None
if os.path.exists(MODEL_PATH):
    _model = keras.models.load_model(MODEL_PATH)
    print(f"[infer_tf_extended] Model loaded ({NUM_CLASSES} classes)")
else:
    print(f"[infer_tf_extended] Model not found: {MODEL_PATH}")


def predict_extended(img_bgr, auto_invert=False):
    """
    Dự đoán chữ số/chữ cái từ ảnh BGR.
    
    Args:
        img_bgr: ảnh BGR từ GUI hoặc camera
        auto_invert: True nếu ảnh có nền sáng (ảnh giấy)
    
    Returns:
        (label_name, probs) hoặc (None, None)
        - label_name: "0"-"9" hoặc "A", "B", "C"
        - probs: numpy array (13,) xác suất từng class
    """
    if _model is None:
        return None, None
    
    digit = preprocess_digit_from_bgr(img_bgr, auto_invert=auto_invert)
    if digit is None:
        return None, None
    
    x = digit.reshape(1, 28, 28, 1)
    probs = _model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    
    return LABEL_NAMES[pred_idx], probs


def predict_extended_from_array(digit_array):
    """
    Dự đoán từ array 28x28 đã preprocess.
    
    Args:
        digit_array: numpy (28,28) float32 [0,1]
    
    Returns:
        (label_name, probs) hoặc (None, None)
    """
    if _model is None:
        return None, None
    
    x = digit_array.reshape(1, 28, 28, 1)
    probs = _model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    
    return LABEL_NAMES[pred_idx], probs


def get_top_predictions(probs, top_k=3):
    """
    Lấy top-k dự đoán từ probs array.
    
    Args:
        probs: numpy array (13,) xác suất
        top_k: số lượng kết quả trả về
    
    Returns:
        List[(label_name, probability), ...]
    """
    indices = np.argsort(probs)[::-1][:top_k]
    return [(LABEL_NAMES[i], float(probs[i])) for i in indices]
