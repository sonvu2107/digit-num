"""
infer_tf.py - Inference CNN với TensorFlow (rút gọn)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from preprocess import preprocess_digit_from_bgr

# Load model
_model = None
if os.path.exists("models/cnn_digit_tf.keras"):
    _model = keras.models.load_model("models/cnn_digit_tf.keras")
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
