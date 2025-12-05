"""
train_svm_aug.py

Train SVM với Data Augmentation để cải thiện độ chính xác
khi nhận diện chữ viết tay từ GUI.

Augmentation bao gồm:
- Xoay nhẹ (±10°)
- Dịch chuyển (±2 pixel)
- Scale (0.9-1.1)
- Thêm morphological dilation để tạo nét dày hơn
"""

import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
from skimage.feature import hog
from tqdm import tqdm
import time
import cv2

from dataset_main import load_dataset_from_folder

# === THAM SỐ HOG ===
HOG_ORIENTATIONS = 12
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'


def augment_image(img, num_augments=3):
    """Tạo các biến thể augmented từ một ảnh.
    
    Tham số:
    - img: ảnh gốc 28x28 float32 [0,1], nền trắng (1.0), chữ đen (~0)
    - num_augments: số ảnh augmented tạo ra
    
    Trả về: list các ảnh augmented (bao gồm ảnh gốc)
    """
    augmented = [img]  # Luôn giữ ảnh gốc
    
    # Convert sang uint8 để xử lý
    img_uint8 = (img * 255).astype(np.uint8)
    
    for _ in range(num_augments):
        aug = img_uint8.copy()
        
        # Random rotation ±10°
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (28, 28), borderValue=255)
        
        # Random translation ±2 pixels
        tx = np.random.randint(-2, 3)
        ty = np.random.randint(-2, 3)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        aug = cv2.warpAffine(aug, M, (28, 28), borderValue=255)
        
        # Random scale 0.9-1.1 (resize rồi crop/pad về 28x28)
        scale = np.random.uniform(0.9, 1.1)
        new_size = int(28 * scale)
        aug_resized = cv2.resize(aug, (new_size, new_size), interpolation=cv2.INTER_AREA)
        
        # Pad hoặc crop về 28x28
        if new_size > 28:
            start = (new_size - 28) // 2
            aug = aug_resized[start:start+28, start:start+28]
        else:
            aug = np.full((28, 28), 255, dtype=np.uint8)
            start = (28 - new_size) // 2
            aug[start:start+new_size, start:start+new_size] = aug_resized
        
        # Random dilation (50% chance) - làm nét dày hơn
        if np.random.random() > 0.5:
            # Đảo màu trước khi dilate (vì chữ đen)
            inv = 255 - aug
            kernel = np.ones((2, 2), np.uint8)
            inv = cv2.dilate(inv, kernel, iterations=1)
            aug = 255 - inv
        
        # Convert về float32 [0,1]
        aug_float = aug.astype(np.float32) / 255.0
        augmented.append(aug_float)
    
    return augmented


def hog_features(images, desc="Extracting HOG"):
    """Tạo HOG feature từ ảnh dataset."""
    feats = []
    for img in tqdm(images, desc=desc, unit="img", ncols=80):
        img = (img * 255).astype("uint8")
        
        feat = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM
        )
        feats.append(feat)
    return np.array(feats, dtype="float32")


def main():
    print("=" * 60)
    print("   TRAIN SVM WITH DATA AUGMENTATION")
    print("=" * 60)
    
    # === BƯỚC 1: Load dataset ===
    print("\n[1/5] Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset_from_folder("dataset-main")
    print(f"      Train: {X_train.shape[0]:,} images")
    print(f"      Test:  {X_test.shape[0]:,} images")

    # === BƯỚC 2: Data Augmentation ===
    print("\n[2/5] Applying data augmentation...")
    NUM_AUGMENTS = 2  # Tạo 2 ảnh augmented cho mỗi ảnh gốc (total 3x)
    
    X_aug = []
    y_aug = []
    
    for i in tqdm(range(len(X_train)), desc="Augmenting", ncols=80):
        aug_images = augment_image(X_train[i], num_augments=NUM_AUGMENTS)
        X_aug.extend(aug_images)
        y_aug.extend([y_train[i]] * len(aug_images))
    
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    
    print(f"      Augmented train: {X_aug.shape[0]:,} images ({NUM_AUGMENTS+1}x)")

    # === BƯỚC 3: Extract HOG features ===
    print("\n[3/5] Extracting HOG features...")
    X_train_hog = hog_features(X_aug, desc="Train HOG")
    X_test_hog  = hog_features(X_test, desc="Test HOG ")
    print(f"      Train HOG shape: {X_train_hog.shape}")
    print(f"      Test HOG shape:  {X_test_hog.shape}")

    # === BƯỚC 4: Train SVM ===
    print("\n[4/5] Training SVM (RBF kernel)...")
    print("      Parameters: C=10.0, gamma='scale', probability=True")
    print("      This may take several minutes with augmented data...\n")
    
    clf = svm.SVC(
        kernel='rbf',
        gamma='scale',
        C=10.0,
        class_weight='balanced',
        probability=True,
        verbose=False
    )
    
    start_time = time.time()
    clf.fit(X_train_hog, y_aug)
    train_time = time.time() - start_time
    print(f"      Training completed in {train_time:.1f} seconds")

    # === BƯỚC 5: Evaluate ===
    print("\n[5/5] Evaluating model...")
    
    y_pred_test = []
    for i in tqdm(range(0, len(X_test_hog), 1000), desc="Test pred ", ncols=80):
        batch = X_test_hog[i:i+1000]
        y_pred_test.extend(clf.predict(batch))
    y_pred_test = np.array(y_pred_test)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print("\n" + "=" * 60)
    print("                        RESULTS")
    print("=" * 60)
    print(f"  Test Accuracy:  {acc_test*100:.2f}%")
    
    # Hiển thị confusion matrix tóm tắt
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n  Per-digit accuracy:")
    print("  " + "-" * 40)
    for i in range(10):
        correct = cm[i, i]
        total = cm[i].sum()
        bar = "█" * int(correct/total * 20)
        print(f"    Digit {i}: {bar:<20} {correct:>4}/{total:<4} ({correct/total*100:.1f}%)")

    # Save model
    os.makedirs("models", exist_ok=True)
    dump(clf, "models/svm_hog.joblib")
    print("\n" + "=" * 60)
    print(f"  Model saved to: models/svm_hog.joblib")
    print("=" * 60)


if __name__ == "__main__":
    main()
