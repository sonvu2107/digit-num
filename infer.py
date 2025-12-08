"""
infer.py - Wrapper functions để gọi các model đã train

Hỗ trợ 3 model:
    1. CNN (PyTorch) - Khuyến nghị: nhanh và chính xác nhất
    2. SVM + HOG features
    3. MLP + pixel features

Input: ảnh BGR từ GUI (280x280, nền đen, chữ trắng)
Output: (prediction, probabilities) với prediction là số 0-9
"""

import os
import numpy as np
from joblib import load as joblib_load
from skimage.feature import hog

from preprocess import preprocess_digit_from_bgr

# === Load SVM và MLP models ===
svm_model = joblib_load("models/svm_hog.joblib")
mlp_model = joblib_load("models/mlp_digit.joblib")

# === Load CNN model (PyTorch) ===
cnn_model = None
try:
    import torch
    import torch.nn as nn
    
    class DigitCNN(nn.Module):
        """CNN architecture phải khớp với train_cnn.py"""
        def __init__(self):
            super(DigitCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(64 * 3 * 3, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = self.dropout1(x)
            x = x.view(-1, 64 * 3 * 3)
            x = self.dropout2(self.relu(self.fc1(x)))
            return self.fc2(x)
    
    if os.path.exists("models/cnn_digit.pth"):
        cnn_model = DigitCNN()
        cnn_model.load_state_dict(
            torch.load("models/cnn_digit.pth", map_location='cpu', weights_only=True)
        )
        cnn_model.eval()
        print("[infer] CNN model loaded")
except ImportError:
    print("[infer] PyTorch not installed, CNN unavailable")
except Exception as e:
    print(f"[infer] CNN load error: {e}")

# === HOG Parameters (phải khớp với train_svm.py) ===
HOG_ORIENTATIONS = 12
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'


def hog_features_single(img_28):
    """Extract HOG features từ một ảnh 28x28.
    
    Args:
        img_28: (28, 28) float32 [0,1], chữ đen nền trắng
    
    Returns:
        (1, D) feature vector cho SVM
    """
    img_uint8 = (img_28 * 255).astype("uint8")
    feat = hog(img_uint8,
               orientations=HOG_ORIENTATIONS,
               pixels_per_cell=HOG_PIXELS_PER_CELL,
               cells_per_block=HOG_CELLS_PER_BLOCK,
               block_norm=HOG_BLOCK_NORM)
    return feat.reshape(1, -1)


def predict_with_svm(img_bgr):
    """Dự đoán bằng SVM + HOG.
    
    Args:
        img_bgr: ảnh BGR từ GUI
    
    Returns:
        (pred, probs): prediction (0-9) và probability array
    """
    digit_28 = preprocess_digit_from_bgr(img_bgr)
    if digit_28 is None:
        return None, None

    feat = hog_features_single(digit_28)
    pred = int(svm_model.predict(feat)[0])
    
    probs = None
    if hasattr(svm_model, "predict_proba"):
        try:
            probs = svm_model.predict_proba(feat)[0]
        except:
            pass
    
    return pred, probs


def predict_with_mlp(img_bgr):
    """Dự đoán bằng MLP + pixel features.
    
    Args:
        img_bgr: ảnh BGR từ GUI
    
    Returns:
        (pred, probs): prediction (0-9) và probability array
    """
    digit_28 = preprocess_digit_from_bgr(img_bgr)
    if digit_28 is None:
        return None, None

    x_flat = digit_28.reshape(1, -1)  # Flatten 28x28 -> 784
    
    probs = None
    if hasattr(mlp_model, "predict_proba"):
        probs = mlp_model.predict_proba(x_flat)[0]

    pred = int(mlp_model.predict(x_flat)[0])
    return pred, probs


def predict_with_cnn(img_bgr):
    """Dự đoán bằng CNN (PyTorch).
    
    Args:
        img_bgr: ảnh BGR từ GUI
    
    Returns:
        (pred, probs): prediction (0-9) và probability array
    """
    if cnn_model is None:
        return None, None
    
    digit_28 = preprocess_digit_from_bgr(img_bgr)
    if digit_28 is None:
        return None, None
    
    import torch
    import torch.nn.functional as F
    
    # Convert: (28,28) -> (1, 1, 28, 28)
    x = torch.from_numpy(digit_28).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logits = cnn_model(x)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred = int(logits.argmax(dim=1).item())
    
    return pred, probs
