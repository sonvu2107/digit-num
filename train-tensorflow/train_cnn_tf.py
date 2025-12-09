"""
=============================================================================
TRAIN CNN - Huấn luyện mạng Neural Network nhận diện chữ số viết tay
=============================================================================

Chạy lệnh:
    python train_cnn_tf.py              # Train với dataset-main
    python train_cnn_tf.py mnist_png    # Train với MNIST

Kết quả: File model lưu tại models/cnn_digit_tf.keras
"""

# ============= IMPORT THƯ VIỆN =============
import os
import sys

# Tắt warning của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np                          # Xử lý mảng số
import tensorflow as tf                     # Framework deep learning
from tensorflow import keras                # API cấp cao của TensorFlow
from keras import layers                    # Các layer để xây dựng model
from sklearn.metrics import classification_report, confusion_matrix  # Đánh giá model

# Import hàm load dataset từ file khác
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_main import load_dataset_from_folder


# ============= BƯỚC 1: XÂY DỰNG MODEL CNN =============
def build_cnn():
    """
    Xây dựng mạng CNN (Convolutional Neural Network).
    
    Cấu trúc:
        Input (28x28x1) - Ảnh grayscale 28x28 pixel
            ↓
        Conv2D (32 filters) - Tìm 32 đặc trưng cơ bản (cạnh, góc...)
        MaxPooling - Giảm kích thước 1/2 → 14x14
            ↓
        Conv2D (64 filters) - Tìm 64 đặc trưng phức tạp hơn
        MaxPooling - Giảm kích thước 1/2 → 7x7
            ↓
        Conv2D (64 filters) - Tìm đặc trưng cao cấp
        MaxPooling - Giảm kích thước 1/2 → 3x3
            ↓
        Flatten - Kéo phẳng thành vector 576 số
        Dense (128) - Học các pattern
        Dense (10) - Output: xác suất 10 chữ số (0-9)
    """
    model = keras.Sequential()
    
    # --- Layer đầu vào ---
    model.add(layers.Input(shape=(28, 28, 1)))  # Ảnh 28x28, 1 kênh màu
    
    # --- Block 1: Tìm đặc trưng cơ bản ---
    model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))  # 28x28 → 14x14
    
    # --- Block 2: Tìm đặc trưng phức tạp ---
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))  # 14x14 → 7x7
    
    # --- Block 3: Tìm đặc trưng cao cấp ---
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))  # 7x7 → 3x3
    model.add(layers.Dropout(0.25))  # Tắt ngẫu nhiên 25% neuron → chống overfitting
    
    # --- Phân loại ---
    model.add(layers.Flatten())              # 3x3x64 = 576 số
    model.add(layers.Dense(128, activation='relu'))  # 128 neuron ẩn
    model.add(layers.Dropout(0.5))           # Tắt 50% neuron
    model.add(layers.Dense(10, activation='softmax'))  # 10 output (0-9)
    
    return model


# ============= BƯỚC 2: TĂNG CƯỜNG DỮ LIỆU =============
def create_data_augmentation():
    """
    Tạo các biến đổi ngẫu nhiên để tăng đa dạng dữ liệu.
    Giúp model học tốt hơn với các kiểu chữ viết khác nhau.
    """
    return keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,       # Xoay ngẫu nhiên ±15 độ
        width_shift_range=0.1,   # Dịch ngang ±10%
        height_shift_range=0.1,  # Dịch dọc ±10%
        zoom_range=0.1           # Phóng to/thu nhỏ ±10%
    )


# ============= BƯỚC 3: ĐÁNH GIÁ MODEL =============
def evaluate_model(model, X_test, y_test):
    """
    Đánh giá model với các chỉ số:
        - Precision: Trong số dự đoán "đúng", bao nhiêu % thực sự đúng?
        - Recall: Trong số thực tế "đúng", model tìm được bao nhiêu %?
        - F1-Score: Trung bình hài hòa của Precision và Recall
        - Confusion Matrix: Ma trận nhầm lẫn giữa các chữ số
    """
    # Dự đoán
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.argmax(axis=1)  # Lấy class có xác suất cao nhất
    
    # In báo cáo
    print("\n" + "="*60)
    print("  BÁO CÁO ĐÁNH GIÁ (PRECISION / RECALL / F1-SCORE)")
    print("="*60)
    print(classification_report(y_test, y_pred, digits=4))
    
    # In confusion matrix
    print("  CONFUSION MATRIX (Ma trận nhầm lẫn)")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print("      " + "".join(f"{i:5d}" for i in range(10)))  # Header
    print("-"*60)
    for i in range(10):
        print(f"  {i}:  " + "".join(f"{cm[i][j]:5d}" for j in range(10)))


# ============= CHƯƠNG TRÌNH CHÍNH =============
def main():
    # --- Đọc tham số từ command line ---
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset-main"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("\n" + "="*60)
    print(f"  HUẤN LUYỆN CNN - Dataset: {dataset} - Epochs: {epochs}")
    print("="*60)
    
    # --- BƯỚC 1: Load dữ liệu ---
    print("\n[1/5] Đang load dữ liệu...")
    X_train, y_train, X_test, y_test = load_dataset_from_folder(dataset)
    
    # Thêm chiều channel: (N, 28, 28) → (N, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"      Train: {len(X_train):,} ảnh")
    print(f"      Test:  {len(X_test):,} ảnh")
    
    # --- BƯỚC 2: Tạo data augmentation ---
    print("\n[2/5] Cấu hình data augmentation...")
    datagen = create_data_augmentation()
    
    # --- BƯỚC 3: Xây dựng model ---
    print("\n[3/5] Xây dựng model CNN...")
    model = build_cnn()
    
    # Compile: cấu hình cách model học
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Thuật toán tối ưu
        loss='sparse_categorical_crossentropy',  # Hàm loss cho bài toán phân loại
        metrics=['accuracy']  # Theo dõi accuracy trong quá trình train
    )
    
    # In thông tin model
    model.summary()
    
    # --- BƯỚC 4: Huấn luyện ---
    print("\n[4/5] Bắt đầu huấn luyện...")
    
    # Callback: tự động giảm learning rate khi model không cải thiện
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        factor=0.5,     # Giảm LR còn 1/2
        patience=3,     # Chờ 3 epoch không cải thiện
        verbose=1
    )
    
    # Train!
    model.fit(
        datagen.flow(X_train, y_train, batch_size=64),  # Batch 64 ảnh
        epochs=epochs,
        validation_data=(X_test, y_test),  # Đánh giá trên test set
        callbacks=[lr_callback]
    )
    
    # --- BƯỚC 5: Đánh giá và lưu model ---
    print("\n[5/5] Đánh giá model...")
    
    # Tính accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n>>> ĐỘ CHÍNH XÁC TRÊN TẬP TEST: {accuracy*100:.2f}% <<<")
    
    # In báo cáo chi tiết
    evaluate_model(model, X_test, y_test)
    
    # Lưu model
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_digit_tf.keras")
    print(f"\n✓ Đã lưu model: models/cnn_digit_tf.keras")
    print("="*60)


# Chạy chương trình
if __name__ == "__main__":
    main()
