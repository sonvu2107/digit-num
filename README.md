# Digit Num - Nhận diện chữ số viết tay

Dự án nhận diện chữ số viết tay sử dụng mô hình CNN (Convolutional Neural Network) với TensorFlow, đạt độ chính xác **99.55%**.

## Tính năng
- **GUI vẽ chữ số**: Canvas 280x280 để vẽ và nhận diện chữ số.
- **Nhận diện thời gian thực**: Sử dụng mô hình CNN đã train.
- **Dataset**:
  - `mnist_png`: 60k ảnh train, 10k ảnh test (MNIST chuẩn).
  - `dataset_extra`: Mẫu tự vẽ từ GUI (tự động merge vào train).
- **Tiền xử lý thông minh**: Tự động căn giữa, chỉnh nghiêng (deskew), chuẩn hóa ảnh đầu vào.

## Kết quả Training

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 99.55% |
| **Train Samples** | 60,538 (60k MNIST + 538 extra) |
| **Test Samples** | 10,000 |
| **Epochs** | 20 |
| **Model Size** | 130,890 params (~511 KB) |

### Per-digit Accuracy
| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 99.69% | 99.80% | 99.75% |
| 1 | 99.82% | 99.56% | 99.69% |
| 2 | 99.61% | 99.52% | 99.56% |
| 3 | 99.02% | 99.90% | 99.46% |
| 4 | 99.29% | 99.80% | 99.54% |
| 5 | 99.89% | 99.33% | 99.61% |
| 6 | 99.89% | 99.16% | 99.53% |
| 7 | 99.22% | 99.61% | 99.42% |
| 8 | 99.39% | 99.69% | 99.54% |
| 9 | 99.70% | 99.11% | 99.40% |

## Cấu trúc dự án
```
digit-num/
├── models/
│   └── cnn_digit_tf.keras  # TensorFlow CNN model
├── tf/
│   ├── app_gui_tf.py       # GUI app
│   ├── infer_tf.py         # Inference module
│   └── train_cnn_tf.py     # Training script
├── preprocess.py           # Pipeline tiền xử lý ảnh
├── dataset_main.py         # Module load dataset
├── dataset_extra/          # Dataset bổ sung từ GUI
└── mnist_png/              # Dataset MNIST
```

## Yêu cầu cài đặt
- Python 3.10+

```powershell
pip install -r requirements.txt
```

Hoặc:
```powershell
pip install numpy opencv-python pillow scikit-learn tensorflow tqdm scikit-image
```

## Sử dụng

### 1. Chạy ứng dụng nhận diện
```powershell
cd tf
python app_gui_tf.py
```

- Vẽ chữ số lên canvas.
- Bấm "Nhận diện" để xem kết quả.
- Bấm "Lưu mẫu" để lưu ảnh vào `dataset_extra/`.

### 2. Huấn luyện mô hình
```powershell
cd tf
python train_cnn_tf.py          # Mặc định 10 epochs
python train_cnn_tf.py all 20   # 20 epochs
```

## Kiến trúc CNN
```
Input (28x28x1)
    ↓
Conv2D(32) + MaxPool → 14x14x32
    ↓
Conv2D(64) + MaxPool → 7x7x64
    ↓
Conv2D(64) + MaxPool → 3x3x64
    ↓
Dropout(0.25) + Flatten → 576
    ↓
Dense(128) + Dropout(0.5)
    ↓
Dense(10) + Softmax → Output
```
