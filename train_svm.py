import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
from skimage.feature import hog
from tqdm import tqdm
import time

from dataset_main import load_dataset_from_folder

# === THAM SỐ HOG (điều chỉnh dựa trên phân tích dataset) ===
# Dataset có nét mảnh (median ~0.6px), nên cần HOG parameter chi tiết
HOG_ORIENTATIONS = 12        # Tăng từ 9 để capture gradient chi tiết hơn
HOG_PIXELS_PER_CELL = (4, 4) # Giữ nguyên, phù hợp với 28x28
HOG_CELLS_PER_BLOCK = (2, 2) # Giữ nguyên
HOG_BLOCK_NORM = 'L2-Hys'    # Chuẩn hoá block

def hog_features(images, desc="Extracting HOG"):
    """Tạo HOG feature từ ảnh dataset (nền trắng, chữ đen, uint8 0-255).
    
    Tham số:
    - images: array (N, 28, 28) float32 [0,1] từ load_dataset_from_folder
    - desc: mô tả cho progress bar
    
    Lưu ý: Dataset đã chuẩn hoá về [0,1] trong load_dataset_from_folder,
    nên dùng * 255 để convert về uint8 là đúng.
    """
    feats = []
    for img in tqdm(images, desc=desc, unit="img", ncols=80):
        # img: (28,28) float32 [0,1] (nền=1.0=trắng, chữ~0=đen)
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
    print("       TRAIN SVM MODEL - Handwritten Digit Recognition")
    print("=" * 60)
    
    # === BƯỚC 1: Load dataset ===
    print("\n[1/4] Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset_from_folder("dataset-main")
    print(f"      Train: {X_train.shape[0]:,} images")
    print(f"      Test:  {X_test.shape[0]:,} images")

    # === BƯỚC 2: Extract HOG features ===
    print("\n[2/4] Extracting HOG features...")
    X_train_hog = hog_features(X_train, desc="Train HOG")
    X_test_hog  = hog_features(X_test, desc="Test HOG ")
    print(f"      Train HOG shape: {X_train_hog.shape}")
    print(f"      Test HOG shape:  {X_test_hog.shape}")

    # === BƯỚC 3: Train SVM ===
    print("\n[3/4] Training SVM (RBF kernel)...")
    print("      Parameters: C=10.0, gamma='scale', probability=True")
    print("      This may take a few minutes...\n")
    
    # Progress bar giả lập cho training (SVM không có callback)
    clf = svm.SVC(
        kernel='rbf',
        gamma='scale',
        C=10.0,
        class_weight='balanced',
        probability=True,
        verbose=False  # Tắt verbose của sklearn, dùng progress riêng
    )
    
    # Hiển thị spinner trong khi train
    start_time = time.time()
    clf.fit(X_train_hog, y_train)
    train_time = time.time() - start_time
    print(f"      Training completed in {train_time:.1f} seconds")

    # === BƯỚC 4: Evaluate ===
    print("\n[4/4] Evaluating model...")
    
    # Progress bar cho prediction
    print("      Predicting on train set...")
    y_pred_train = []
    for i in tqdm(range(0, len(X_train_hog), 1000), desc="Train pred", ncols=80):
        batch = X_train_hog[i:i+1000]
        y_pred_train.extend(clf.predict(batch))
    y_pred_train = np.array(y_pred_train)
    
    print("      Predicting on test set...")
    y_pred_test = []
    for i in tqdm(range(0, len(X_test_hog), 1000), desc="Test pred ", ncols=80):
        batch = X_test_hog[i:i+1000]
        y_pred_test.extend(clf.predict(batch))
    y_pred_test = np.array(y_pred_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print("\n" + "=" * 60)
    print("                        RESULTS")
    print("=" * 60)
    print(f"  Train Accuracy: {acc_train*100:.2f}%")
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
