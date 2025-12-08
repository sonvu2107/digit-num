"""
dataset_main.py - Load dataset chữ số viết tay từ folder

Hỗ trợ 2 loại dataset:
    1. dataset-main (40k train, 10k test) - dataset gốc
    2. mnist_png (60k train, 10k test) - MNIST chuẩn

Cấu trúc folder:
    dataset-main/           mnist_png/
        train/                  training/
            0/*.png                 0/*.png
            ...                     ...
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


def load_dataset_from_folder(root="dataset-main", max_per_class=None):
    """Load toàn bộ dataset từ folder structure.
    
    Args:
        root: đường dẫn thư mục gốc ("dataset-main" hoặc "mnist_png")
        max_per_class: giới hạn số ảnh mỗi class (None = không giới hạn)
    
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
    
    return X_train, y_train, X_test, y_test
