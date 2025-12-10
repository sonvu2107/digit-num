"""
dataset_main.py - Load dataset chữ số viết tay từ folder

Hỗ trợ:
    1. dataset-main (40k train, 10k test) - dataset gốc
    2. mnist_png (60k train, 10k test) - MNIST chuẩn
    3. dataset_extra - mẫu tự vẽ từ GUI (tự động merge vào train)

Cấu trúc folder:
    dataset-main/           mnist_png/           dataset_extra/
        train/                  training/            0/*.png
            0/*.png                 0/*.png          1/*.png
            ...                     ...              ...
        test/                   testing/
            0/*.png                 0/*.png

Output:
    - X: numpy array (N, 28, 28) float32 [0,1]
    - y: numpy array (N,) int64, nhãn 0-9
"""

import numpy as np
import cv2
import glob
import os


# Mapping tên folder cho các dataset khác nhau
# invert: True nếu ảnh gốc là nền đen chữ trắng (cần đảo để thành nền trắng chữ đen)
DATASET_CONFIG = {
    "dataset-main": {"train": "train", "test": "test", "invert": False},
    "mnist_png": {"train": "training", "test": "testing", "invert": True},
}

# Thư mục chứa mẫu tự vẽ
EXTRA_DATASET_DIR = "dataset_extra"


def load_extra_samples():
    """Load thêm mẫu từ dataset_extra/ (mẫu tự vẽ từ GUI).
    
    Returns:
        X_extra, y_extra: ảnh và nhãn từ dataset_extra, hoặc (None, None) nếu không có
    """
    if not os.path.exists(EXTRA_DATASET_DIR):
        return None, None
    
    imgs = []
    labels = []
    
    for digit in range(10):
        digit_dir = os.path.join(EXTRA_DATASET_DIR, str(digit))
        if not os.path.exists(digit_dir):
            continue
            
        pattern = os.path.join(digit_dir, "*.png")
        files = glob.glob(pattern)
        
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (28, 28))
            imgs.append(img)
            labels.append(digit)
    
    if len(imgs) == 0:
        return None, None
    
    X = np.array(imgs, dtype="float32") / 255.0
    y = np.array(labels, dtype="int64")
    return X, y


def load_dataset_from_folder(root="dataset-main", max_per_class=None, include_extra=True):
    """Load toàn bộ dataset từ folder structure.
    
    Args:
        root: đường dẫn thư mục gốc ("dataset-main" hoặc "mnist_png")
        max_per_class: giới hạn số ảnh mỗi class (None = không giới hạn)
        include_extra: có load thêm dataset_extra không (mặc định: True)
    
    Returns:
        X_train, y_train, X_test, y_test
        - X: (N, 28, 28) float32 [0,1], nền trắng chữ đen
        - y: (N,) int64
    """
    # Lấy config tên folder
    config = DATASET_CONFIG.get(root, {"train": "train", "test": "test", "invert": False})
    need_invert = config.get("invert", False)
    
    def load_split(split_name):
        """Load một split (train hoặc test)."""
        imgs = []
        labels = []
        split_dir = os.path.join(root, config[split_name])
        
        for digit in range(10):
            class_dir = os.path.join(split_dir, str(digit))
            pattern = os.path.join(class_dir, "*.png")
            files = glob.glob(pattern)
            
            # Giới hạn số ảnh nếu cần
            if max_per_class is not None:
                files = files[:max_per_class]
            
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (28, 28))
                
                # Đảo màu nếu cần (nền đen chữ trắng -> nền trắng chữ đen)
                if need_invert:
                    img = 255 - img
                
                imgs.append(img)
                labels.append(digit)
        
        X = np.array(imgs, dtype="float32") / 255.0
        y = np.array(labels, dtype="int64")
        return X, y

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    print(f"Loaded dataset: {root}")
    print(f"  Train: {X_train.shape} {y_train.shape}")
    print(f"  Test : {X_test.shape} {y_test.shape}")
    
    # Load thêm dataset_extra nếu có
    if include_extra:
        X_extra, y_extra = load_extra_samples()
        if X_extra is not None and len(X_extra) > 0:
            X_train = np.concatenate([X_train, X_extra], axis=0)
            y_train = np.concatenate([y_train, y_extra], axis=0)
            print(f"  + Extra samples: {len(X_extra)} (merged into train)")
            print(f"  = Total train: {len(X_train)}")
    
    return X_train, y_train, X_test, y_test

