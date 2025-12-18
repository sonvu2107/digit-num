"""
infer_tf.py - Inference CNN với TensorFlow (rút gọn)
"""

import os
import sys

# Thêm thư mục cha vào path để import preprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from preprocess import preprocess_digit_from_bgr

# Đường dẫn đến model (từ thư mục gốc)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_digit_tf.keras")

# Load model
_model = None
if os.path.exists(MODEL_PATH):
    _model = keras.models.load_model(MODEL_PATH)
    print("[infer_tf] Model loaded")


def predict_with_cnn_tf(img_bgr):
    """Dự đoán chữ số từ ảnh BGR. Returns: (pred, probs) hoặc (None, None)."""
    if _model is None:
        return None, None
    
    digit = preprocess_digit_from_bgr(img_bgr)
    if digit is None:
        return None, None
    
    x = digit.reshape(1, 28, 28, 1)
    probs = _model.predict(x, verbose=0)[0]
    return int(np.argmax(probs)), probs


def predict_with_cnn_tf_from_array(digit_array):
    """Dự đoán chữ số từ array 28x28 đã preprocess.
    
    Args:
        digit_array: numpy (28,28) float32 [0,1] - đã preprocess sẵn
    
    Returns:
        (pred, probs) hoặc (None, None)
    """
    if _model is None:
        return None, None
    
    x = digit_array.reshape(1, 28, 28, 1)
    probs = _model.predict(x, verbose=0)[0]
    return int(np.argmax(probs)), probs
