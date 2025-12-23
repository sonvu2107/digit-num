"""
=============================================================================
TRAIN CNN EXTENDED - Huấn luyện CNN nhận diện CHỮ SỐ + CHỮ CÁI (0-9 + A,B,C)
=============================================================================

Cải tiến v2:
  - Class weight: tập trung vào A, B, C (đặc biệt B)
  - Augmentation: rotation giảm, thêm shear
  - EarlyStopping: restore best weights

Chạy lệnh:
    python train_cnn_extended.py          # Train 13 classes với 20 epochs
    python train_cnn_extended.py 25       # Train với 25 epochs

Kết quả: File model lưu tại models/cnn_digit_letter_tf.keras
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Import hàm load dataset mở rộng
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_main_extended import load_all_datasets, LABEL_NAMES, NUM_CLASSES

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def build_cnn_extended():
    """
    Xây dựng mạng CNN cho 13 classes (0-9 + A, B, C).
    """
    model = keras.Sequential()
    
    model.add(layers.Input(shape=(28, 28, 1)))
    
    # Block 1
    model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    
    # Block 2
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    
    # Block 3
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(0.25))
    
    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    
    return model


def create_data_augmentation():
    """
    Data augmentation được tối ưu cho letters:
    - Rotation giảm xuống 8° (tránh B→6)
    - Thêm shear để tăng đa dạng
    """
    return keras.preprocessing.image.ImageDataGenerator(
        rotation_range=8,         # Giảm từ 15 → 8 (tránh B tròn giống 6)
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        shear_range=0.10          # Thêm shear
    )


def compute_class_weight():
    """
    Class weight để tập trung vào letters (đặc biệt B).
    B có confusion cao với 6 nên weight cao nhất.
    """
    cw = {i: 1.0 for i in range(NUM_CLASSES)}
    cw[10] = 2.0  # A - nhầm với 2/4/9
    cw[11] = 3.0  # B - nhầm mạnh với 6 (74 mẫu)
    cw[12] = 1.6  # C - tương đối ổn
    print(f"Class weights: A={cw[10]}, B={cw[11]}, C={cw[12]}")
    return cw


def evaluate_model(model, X_test, y_test):
    """Đánh giá model với 13 classes."""
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.argmax(axis=1)
    
    print("\n" + "="*60)
    print("  BÁO CÁO ĐÁNH GIÁ (13 CLASSES: 0-9 + A, B, C)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4))
    
    # Confusion matrix
    print("  CONFUSION MATRIX")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    
    # Header
    print("      " + "".join(f"{name:>5}" for name in LABEL_NAMES))
    print("-"*70)
    for i, name in enumerate(LABEL_NAMES):
        row = "".join(f"{cm[i][j]:5d}" for j in range(len(LABEL_NAMES)))
        print(f"  {name}:  {row}")
    
    # Highlight confusions for A/B/C
    print("\n[ANALYSIS] Key confusions:")
    for letter, idx in [("A", 10), ("B", 11), ("C", 12)]:
        total = cm[idx].sum()
        correct = cm[idx][idx]
        recall = correct / total * 100
        confusions = [(LABEL_NAMES[j], cm[idx][j]) for j in range(NUM_CLASSES) if j != idx and cm[idx][j] > 0]
        confusions.sort(key=lambda x: -x[1])
        top_conf = ", ".join([f"{name}:{cnt}" for name, cnt in confusions[:3]])
        print(f"  {letter}: recall={recall:.1f}%, confusions: {top_conf}")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    print("\n" + "="*60)
    print(f"  HUẤN LUYỆN CNN EXTENDED v2 - 13 Classes - Epochs: {epochs}")
    print("  [Improvements: class_weight, shear aug, early stopping]")
    print("="*60)
    
    # BƯỚC 1: Load dữ liệu
    print("\n[1/5] Đang load dữ liệu...")
    os.chdir(ROOT_DIR)
    X_train, y_train, X_test, y_test = load_all_datasets()
    
    # Thêm chiều channel
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"      Train: {len(X_train):,} ảnh")
    print(f"      Test:  {len(X_test):,} ảnh")
    print(f"      Classes: {NUM_CLASSES} ({', '.join(LABEL_NAMES)})")
    
    # BƯỚC 2: Data augmentation (cải tiến)
    print("\n[2/5] Cấu hình data augmentation (rotation=8°, shear=0.1)...")
    datagen = create_data_augmentation()
    
    # BƯỚC 3: Xây dựng model
    print("\n[3/5] Xây dựng model CNN (13 classes)...")
    model = build_cnn_extended()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # BƯỚC 4: Huấn luyện với improvements
    print("\n[4/5] Bắt đầu huấn luyện...")
    
    # Callbacks
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        verbose=1
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,  # Giữ weights tốt nhất!
        verbose=1
    )
    
    # Class weight
    class_weights = compute_class_weight()
    
    # Train!
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[lr_callback, early_stop],
        class_weight=class_weights  # Tập trung vào A/B/C
    )
    
    # BƯỚC 5: Đánh giá và lưu
    print("\n[5/5] Đánh giá model...")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n>>> ĐỘ CHÍNH XÁC: {accuracy*100:.2f}% <<<")
    
    evaluate_model(model, X_test, y_test)
    
    # Lưu model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "cnn_digit_letter_tf.keras")
    model.save(model_path)
    print(f"\n✓ Đã lưu model: {model_path}")
    print(f"  Best epoch restored by EarlyStopping")
    print("="*60)


if __name__ == "__main__":
    main()
