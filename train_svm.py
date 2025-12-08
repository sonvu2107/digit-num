"""
train_svm.py - Train SVM với HOG features

Hỗ trợ 2 dataset:
    - dataset-main: 40k train, 10k test
    - mnist_png: 60k train, 10k test

Sử dụng:
    python train_svm.py                    # Train với dataset-main
    python train_svm.py mnist_png          # Train với MNIST

HOG Parameters:
    - orientations=12: số hướng gradient (30° mỗi bin)
    - pixels_per_cell=(4,4): cell 4x4 pixels
    - cells_per_block=(2,2): block 2x2 cells
    
SVM: RBF kernel, C=10.0, gamma='scale'

Lưu ý: SVM RBF chậm với dataset lớn (O(n²)), 
       với MNIST 60k có thể mất 10-30 phút.
"""

import os
import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
from skimage.feature import hog
from tqdm import tqdm
import time

from dataset_main import load_dataset_from_folder

# === THAM SỐ HOG ===
HOG_ORIENTATIONS = 12        # Số hướng gradient (360° / 12 = 30° mỗi bin)
HOG_PIXELS_PER_CELL = (4, 4) # Mỗi cell 4x4 pixels -> 7x7 cells cho ảnh 28x28
HOG_CELLS_PER_BLOCK = (2, 2) # Block 2x2 cells để L2 normalize
HOG_BLOCK_NORM = 'L2-Hys'    # L2 norm + clipping (Hys = Hysteresis)


def hog_features(images, desc="Extracting HOG"):
    """Extract HOG features từ batch ảnh.
    
    HOG mô tả hình dạng qua phân bố gradient:
        - Chia ảnh thành cells
        - Mỗi cell tính histogram của hướng gradient
        - Normalize theo block để robust với lighting
    
    Args:
        images: (N, 28, 28) float32 [0,1]
        desc: mô tả cho progress bar
    
    Returns:
        (N, D) float32, D = số HOG features
    """
    feats = []
    for img in tqdm(images, desc=desc, unit="img", ncols=80):
        img_uint8 = (img * 255).astype("uint8")
        
        feat = hog(img_uint8,
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm=HOG_BLOCK_NORM)
        feats.append(feat)
    
    return np.array(feats, dtype="float32")


def main():
    # === Parse arguments ===
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset-main"
    
    print("=" * 60)
    print("       TRAIN SVM MODEL - HOG Features")
    print("=" * 60)
    print(f"\nDataset: {dataset}")
    
    # === Load dataset ===
    print("\n[1/4] Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset_from_folder(dataset)
    print(f"      Train: {X_train.shape[0]:,} images")
    print(f"      Test:  {X_test.shape[0]:,} images")

    # === Extract HOG ===
    print("\n[2/4] Extracting HOG features...")
    X_train_hog = hog_features(X_train, desc="Train HOG")
    X_test_hog = hog_features(X_test, desc="Test HOG ")
    print(f"      HOG feature dim: {X_train_hog.shape[1]}")

    # === Train SVM ===
    print("\n[3/4] Training SVM (RBF kernel)...")
    print("      C=10.0, gamma='scale', probability=True")
    
    clf = svm.SVC(
        kernel='rbf',
        gamma='scale',
        C=10.0,
        class_weight='balanced',  # Cân bằng class nếu data imbalanced
        probability=True,         # Cho phép predict_proba()
        verbose=False
    )
    
    start_time = time.time()
    clf.fit(X_train_hog, y_train)
    print(f"      Completed in {time.time() - start_time:.1f}s")

    # === Evaluate ===
    print("\n[4/4] Evaluating...")
    
    # Predict theo batch để hiển thị progress
    y_pred_test = []
    for i in tqdm(range(0, len(X_test_hog), 1000), desc="Predicting", ncols=80):
        y_pred_test.extend(clf.predict(X_test_hog[i:i+1000]))
    
    acc = accuracy_score(y_test, y_pred_test)
    
    # Hiển thị kết quả
    print("\n" + "=" * 60)
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print("=" * 60)
    
    # Per-digit accuracy
    cm = confusion_matrix(y_test, y_pred_test)
    print("\n  Per-digit accuracy:")
    for i in range(10):
        correct = cm[i, i]
        total = cm[i].sum()
        bar = "█" * int(correct / total * 20)
        print(f"    {i}: {bar:<20} {correct:>4}/{total} ({100*correct/total:.1f}%)")

    # Save model
    os.makedirs("models", exist_ok=True)
    dump(clf, "models/svm_hog.joblib")
    print(f"\n  Saved: models/svm_hog.joblib")


if __name__ == "__main__":
    main()
