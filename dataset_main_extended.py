"""
dataset_main_extended.py - Load dataset CHỮ SỐ + CHỮ CÁI (0-9 + A, B, C)

Hỗ trợ:
    1. mnist_png (60k train, 10k test) - chữ số 0-9
    2. dataset_extra - mẫu tự vẽ từ GUI (0-9 + A, B, C)
    3. dataset_letters - chữ cái A, B, C (từ EMNIST)

QUAN TRỌNG: Tất cả ảnh được chuẩn hoá về:
    - 28x28 pixels
    - Nền ĐEN, chữ TRẮNG (auto-invert nếu cần)
    - float32 [0,1]

Output:
    - X: numpy array (N, 28, 28) float32 [0,1]
    - y: numpy array (N,) int64, nhãn 0-12 (0-9: digits, 10: A, 11: B, 12: C)
"""

import numpy as np
import cv2
import glob
import os


# === MAPPING ===
# 0-9: chữ số
# 10: A, 11: B, 12: C
NUM_CLASSES = 13
LABEL_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABEL_NAMES)}


# Thư mục dataset
MNIST_DIR = "mnist_png"
LETTERS_DIR = "dataset_letters"
EXTRA_DIR = "dataset_extra"


# === HÀM PREPROCESS THỐNG NHẤT ===
def preprocess_img(img, invert=None):
    """
    Chuẩn hoá ảnh về cùng chuẩn:
      - 28x28
      - Nền đen, chữ trắng
      - float32 [0,1]
    
    Args:
        img: ảnh grayscale (numpy array)
        invert: 
            - None: tự quyết định theo mean brightness
            - True/False: ép invert hoặc không
    
    Returns:
        ảnh đã chuẩn hoá (28,28) float32 [0,1], hoặc None nếu lỗi
    """
    if img is None:
        return None

    # Resize chuẩn 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Tự động quyết định invert nếu không chỉ định:
    # Nếu nền sáng (mean cao) -> khả năng là nền trắng chữ đen => invert
    if invert is None:
        if img.mean() > 127:
            img = 255 - img
    else:
        if invert:
            img = 255 - img

    # Normalize về [0,1]
    img = img.astype("float32") / 255.0
    return img


def load_mnist_png():
    """Load MNIST PNG dataset (chữ số 0-9)."""
    if not os.path.exists(MNIST_DIR):
        print(f"[SKIP] Không tìm thấy: {MNIST_DIR}")
        return None, None, None, None
    
    def load_split(split_name):
        folder_map = {"train": "training", "test": "testing"}
        split_dir = os.path.join(MNIST_DIR, folder_map[split_name])
        
        imgs, labels = [], []
        for digit in range(10):
            class_dir = os.path.join(split_dir, str(digit))
            files = glob.glob(os.path.join(class_dir, "*.png"))
            
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Preprocess thống nhất (auto-invert nếu cần)
                img = preprocess_img(img, invert=None)
                if img is None:
                    continue
                    
                imgs.append(img)
                labels.append(digit)  # 0-9
        
        X = np.array(imgs, dtype="float32")  # Đã normalize trong preprocess_img
        y = np.array(labels, dtype="int64")
        return X, y
    
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    
    print(f"MNIST: train={len(X_train)}, test={len(X_test)}")
    return X_train, y_train, X_test, y_test


def load_letters():
    """Load EMNIST Letters dataset (A, B, C)."""
    if not os.path.exists(LETTERS_DIR):
        print(f"[SKIP] Không tìm thấy: {LETTERS_DIR}")
        return None, None, None, None
    
    def load_split(split_name):
        split_dir = os.path.join(LETTERS_DIR, split_name)
        if not os.path.exists(split_dir):
            return np.array([]), np.array([])
        
        imgs, labels = [], []
        for letter in ["A", "B", "C"]:
            class_dir = os.path.join(split_dir, letter)
            if not os.path.exists(class_dir):
                continue
            
            files = glob.glob(os.path.join(class_dir, "*.png"))
            label_idx = LABEL_TO_INDEX[letter]  # A=10, B=11, C=12
            
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Preprocess thống nhất (auto-invert nếu cần)
                img = preprocess_img(img, invert=None)
                if img is None:
                    continue
                    
                imgs.append(img)
                labels.append(label_idx)
        
        if len(imgs) == 0:
            return np.array([]), np.array([])
        
        X = np.array(imgs, dtype="float32")  # Đã normalize trong preprocess_img
        y = np.array(labels, dtype="int64")
        return X, y
    
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    
    print(f"Letters (A,B,C): train={len(X_train)}, test={len(X_test)}")
    return X_train, y_train, X_test, y_test


def load_extra_samples():
    """Load mẫu tự vẽ từ dataset_extra/ (cả số và chữ)."""
    if not os.path.exists(EXTRA_DIR):
        return None, None
    
    imgs, labels = [], []
    
    # Load tất cả các class (0-9 và A, B, C)
    for class_name in LABEL_NAMES:
        class_dir = os.path.join(EXTRA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
        
        files = glob.glob(os.path.join(class_dir, "*.png"))
        label_idx = LABEL_TO_INDEX[class_name]
        
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Preprocess thống nhất (auto-invert nếu cần)
            img = preprocess_img(img, invert=None)
            if img is None:
                continue
                
            imgs.append(img)
            labels.append(label_idx)
    
    if len(imgs) == 0:
        return None, None
    
    X = np.array(imgs, dtype="float32")  # Đã normalize trong preprocess_img
    y = np.array(labels, dtype="int64")
    print(f"Extra samples: {len(X)}")
    return X, y


def debug_preview_samples(X_train, y_train, num_samples=12):
    """Hiển thị preview ảnh để kiểm tra polarity."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    indices = np.random.choice(len(X_train), num_samples, replace=False)
    
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        ax.imshow(X_train[idx], cmap="gray")
        ax.set_title(f"{LABEL_NAMES[y_train[idx]]} (mean={X_train[idx].mean():.2f})")
        ax.axis("off")
    
    plt.suptitle("Debug: Kiểm tra polarity (nền đen = mean thấp)", fontsize=11)
    plt.tight_layout()
    plt.savefig("debug_dataset_preview.png", dpi=100)
    plt.show()
    print("Saved: debug_dataset_preview.png")


def load_all_datasets(include_extra=True, debug_preview=False):
    """Load và kết hợp TẤT CẢ dataset: MNIST + Letters + Extra.
    
    Args:
        include_extra: có load thêm dataset_extra không
        debug_preview: hiển thị preview ảnh để kiểm tra polarity
    
    Returns:
        X_train, y_train, X_test, y_test
        - X: (N, 28, 28) float32 [0,1]
        - y: (N,) int64, nhãn 0-12
    """
    print("=" * 60)
    print("  LOADING ALL DATASETS (Digits + Letters)")
    print("=" * 60)
    
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    
    # 1. Load MNIST (0-9)
    data = load_mnist_png()
    if data[0] is not None:
        all_X_train.append(data[0])
        all_y_train.append(data[1])
        all_X_test.append(data[2])
        all_y_test.append(data[3])
    
    # 2. Load Letters (A, B, C)
    data = load_letters()
    if data[0] is not None and len(data[0]) > 0:
        all_X_train.append(data[0])
        all_y_train.append(data[1])
        if len(data[2]) > 0:
            all_X_test.append(data[2])
            all_y_test.append(data[3])
    
    # 3. Load Extra samples
    if include_extra:
        X_extra, y_extra = load_extra_samples()
        if X_extra is not None:
            all_X_train.append(X_extra)
            all_y_train.append(y_extra)
    
    # Kết hợp tất cả
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_test = np.concatenate(all_X_test, axis=0)
    y_test = np.concatenate(all_y_test, axis=0)
    
    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    print("-" * 60)
    print(f"TOTAL COMBINED (13 classes: 0-9 + A,B,C):")
    print(f"  Train: {X_train.shape} ({len(X_train):,} samples)")
    print(f"  Test:  {X_test.shape} ({len(X_test):,} samples)")
    
    # Thống kê mỗi class
    print("\nClass distribution (train):")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(y_train == i)
        print(f"  {name}: {count:,}")
    
    # Debug: kiểm tra mean để xác nhận polarity đúng
    print("\n[DEBUG] Mean brightness per source (should be LOW = dark background):")
    print(f"  Overall train mean: {X_train.mean():.3f}")
    
    print("=" * 60)
    
    # Preview nếu cần
    if debug_preview:
        debug_preview_samples(X_train, y_train)
    
    return X_train, y_train, X_test, y_test


def get_label_name(index):
    """Chuyển index (0-12) thành tên label (0-9, A, B, C)."""
    return LABEL_NAMES[index]


if __name__ == "__main__":
    # Test với debug preview
    X_train, y_train, X_test, y_test = load_all_datasets(debug_preview=True)
