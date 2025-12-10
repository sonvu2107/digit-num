# Digit Num - Nhận diện chữ số viết tay

Dự án nhận diện chữ số viết tay sử dụng mô hình CNN (Convolutional Neural Network) với độ chính xác cao (~99.6%).

## Tính năng
- **GUI vẽ chữ số**: Canvas 280x280 nền trắng, nét đen.
- **Nhận diện thời gian thực**: Sử dụng mô hình CNN đã train.
- **Hỗ trợ 2 dataset**:
  - `dataset-main`: 40k ảnh train, 10k ảnh test (nền trắng, chữ đen).
  - `mnist_png`: 60k ảnh train, 10k ảnh test (nền đen, chữ trắng).
- **Tiền xử lý thông minh**: Tự động căn giữa, chỉnh nghiêng (deskew), chuẩn hóa ảnh đầu vào.

## Cấu trúc dự án
```
app_gui.py          # GUI vẽ + gọi nhận diện
preprocess.py       # Pipeline tiền xử lý ảnh (Deskew, Resize, Center)
infer.py            # Wrapper gọi model CNN
train_cnn.py        # Script train CNN (PyTorch)
train_mlp.py        # Script train MLP (để tham khảo)
train_svm.py        # Script train SVM (để tham khảo)
dataset_main.py     # Module load dataset
models/             # Chứa model đã train (.pth, .joblib)
dataset-main/       # Dataset gốc
mnist_png/          # Dataset MNIST
(có thay đổi, sẽ update lại cấu trúc sau)
```

## Yêu cầu cài đặt
- Python 3.10+
- Các thư viện:
```powershell
pip install numpy opencv-python pillow scikit-learn joblib torch torchvision tqdm
```

## Sử dụng

### 1. Chạy ứng dụng nhận diện
```powershell
python app_gui.py
```
- Vẽ chữ số lên canvas đen.
- Bấm "Nhận diện" để xem kết quả dự đoán và độ tin cậy (confidence).

### 2. Huấn luyện mô hình (Training)

**Train CNN (Khuyến nghị):**
```powershell
# Train với dataset gốc (mặc định)
python train_cnn.py

# Train với MNIST
python train_cnn.py mnist_png

# Train với số epochs tùy chọn
python train_cnn.py mnist_png 30
```

**Train MLP / SVM (Tham khảo):**
```powershell
python train_mlp.py [dataset] [epochs]
python train_svm.py [dataset]
```

## Thuật toán & Cải tiến

### 1. Tiền xử lý (Preprocessing)
Để đảm bảo model hoạt động tốt với nét vẽ từ GUI, pipeline xử lý bao gồm:
- **Grayscale & Blur**: Giảm nhiễu.
- **Otsu Thresholding**: Tách nét vẽ khỏi nền.
- **ROI Cropping**: Cắt vùng chứa chữ số.
- **Deskewing**: Tự động xoay ảnh để chữ số thẳng đứng (dựa trên image moments).
- **Aspect Ratio Preserving Resize**: Resize về 20x20 giữ nguyên tỉ lệ, sau đó pad vào trung tâm ảnh 28x28.
- **Color Inversion**: Đảm bảo đầu vào luôn là **nền trắng, chữ đen** (hoặc ngược lại tùy config) để đồng bộ với tập train.

### 2. Mô hình CNN
Mô hình CNN được thiết kế với kiến trúc:
- **3 Convolutional Layers**: (32, 64, 64 filters) để trích xuất đặc trưng không gian.
- **MaxPooling**: Giảm chiều dữ liệu, giữ lại đặc trưng quan trọng.
- **Dropout (0.25, 0.5)**: Chống overfitting hiệu quả.
- **Fully Connected Layers**: Phân lớp cuối cùng.
- **Accuracy**: Đạt ~99.42% trên tập test.

### 3. Dataset Handling
- Hỗ trợ load linh hoạt giữa `dataset-main` và `mnist_png`.
- Tự động xử lý sự khác biệt về màu sắc (nền đen/trắng) giữa các dataset thông qua flag `invert` trong `dataset_main.py`.

---
*Dự án được phát triển với sự hỗ trợ của AI Assistant.*
