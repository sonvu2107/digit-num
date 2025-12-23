# Digit Num - Nhận diện chữ số và chữ cái viết tay

Dự án nhận diện chữ số viết tay (0-9) và chữ cái (A, B, C) sử dụng mô hình CNN (Convolutional Neural Network) với độ chính xác cao (~99.4% cho chữ số, ~95%+ cho extended model).

## Mục lục

- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [Thuật toán & Kiến trúc](#thuật-toán--kiến-trúc)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)

## Tính năng

- **GUI vẽ và nhận diện**: Canvas 280x280 nền đen, nét trắng với slider điều chỉnh độ dày nét
- **Nhận diện thời gian thực**: Sử dụng mô hình CNN đã train (TensorFlow)
- **Hỗ trợ 2 mode**:
  - **Chữ số (0-9)**: Model cơ bản với 10 classes
  - **Extended (0-9 + A, B, C)**: Model mở rộng với 13 classes
- **Xử lý ảnh từ file**: Mở và nhận diện ảnh từ máy tính với auto-invert cho ảnh giấy
- **Lưu mẫu training**: Lưu mẫu vẽ vào `dataset_extra/` để train tiếp
- **Tiền xử lý thông minh**: 
  - Tự động căn giữa
  - Adaptive thresholding cho ảnh giấy
  - Tự động phát hiện và đảo màu (nền sáng → nền tối)
  - ROI cropping và resize giữ tỉ lệ

## Cấu trúc dự án

```
digit-num/
├── preprocess.py              # Pipeline tiền xử lý ảnh (Deskew, Resize, Center)
├── dataset_main.py            # Module load dataset chữ số (0-9)
├── dataset_main_extended.py   # Module load dataset mở rộng (0-9 + A, B, C)
│
├── tf/                        # TensorFlow implementation (Khuyến nghị)
│   ├── app_gui_tf.py          # GUI vẽ + nhận diện (13 classes)
│   ├── infer_tf.py            # Inference chữ số (10 classes)
│   ├── infer_tf_extended.py   # Inference mở rộng (13 classes)
│   ├── train_cnn_tf.py        # Train CNN chữ số
│   └── train_cnn_extended.py  # Train CNN mở rộng
│
├── pytorch/                   # PyTorch implementation (Tham khảo)
│   ├── app_gui.py
│   ├── infer.py
│   ├── train_cnn.py
│   ├── train_mlp.py
│   └── train_svm.py
│
├── models/                    # Chứa model đã train
│   ├── cnn_digit_tf.keras     # Model chữ số (TensorFlow)
│   ├── cnn_digit_letter_tf.keras  # Model mở rộng (TensorFlow)
│   ├── cnn_digit.pth          # Model chữ số (PyTorch)
│   ├── mlp_digit.joblib       # Model MLP
│   └── svm_hog.joblib         # Model SVM
│
├── dataset_extra/             # Mẫu tự vẽ từ GUI (0-9 + A, B, C)
├── mnist_png/                 # Dataset MNIST (60k train, 10k test)
├── dataset_letters/           # Dataset chữ cái EMNIST (A, B, C)
│
├── requirements.txt           # Dependencies
└── README.md                  # File này
```

## Yêu cầu hệ thống

- **Python**: 3.10 hoặc cao hơn
- **Hệ điều hành**: Windows, Linux, macOS
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **GPU**: Không bắt buộc (có thể chạy trên CPU), nhưng GPU sẽ train nhanh hơn

## Cài đặt

### 1. Clone hoặc tải dự án

```bash
# Nếu có git
git clone <repository-url>
cd digit-num

# Hoặc giải nén file zip
```

### 2. Tạo môi trường ảo (Khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Hoặc cài đặt thủ công
pip install numpy==1.26.4 opencv-python==4.9.0.80 scikit-learn==1.3.2 scikit-image==0.22.0 joblib==1.3.2 Pillow==10.0.1 tqdm==4.66.1

# TensorFlow (CPU)
pip install tensorflow>=2.13.0

# TensorFlow (GPU - nếu có CUDA)
# Xem hướng dẫn tại: https://www.tensorflow.org/install/gpu

# PyTorch (tùy chọn, chỉ cần nếu dùng pytorch/)
# CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 4. Tải dataset (Tùy chọn)

- **MNIST**: Giải nén `mnist_png.zip` vào thư mục gốc
- **Dataset letters**: Giải nén `dataset_letters/` nếu có
- Nếu không có dataset, vẫn có thể chạy GUI với model đã train sẵn

## Hướng dẫn sử dụng

### 1. Chạy ứng dụng GUI (Khuyến nghị)

```bash
# Chạy GUI với model mở rộng (0-9 + A, B, C)
python tf/app_gui_tf.py
```

**Giao diện GUI:**
- **Canvas vẽ**: Vẽ chữ số/chữ cái bằng chuột (nền đen, nét trắng)
- **Slider "Độ dày nét"**: Điều chỉnh độ dày nét vẽ (8-28 pixels)
- **Nút "Nhận diện"**: Dự đoán chữ số/chữ cái đã vẽ
- **Nút "Xoá"**: Xóa canvas
- **Dropdown "Nhãn"**: Chọn nhãn (0-9, A, B, C) để lưu mẫu
- **Nút "Lưu mẫu"**: Lưu ảnh đã vẽ vào `dataset_extra/<label>/` để train tiếp
- **Nút "Mở ảnh"**: Mở ảnh từ máy tính và nhận diện (hỗ trợ auto-invert cho ảnh giấy)

**Cách sử dụng:**
1. Vẽ chữ số hoặc chữ cái lên canvas
2. Bấm "Nhận diện" để xem kết quả và độ tin cậy (%)
3. (Tùy chọn) Chọn nhãn và bấm "Lưu mẫu" để lưu vào dataset
4. (Tùy chọn) Bấm "Mở ảnh" để nhận diện ảnh từ file

### 2. Sử dụng từ code Python

```python
from tf.infer_tf_extended import predict_extended
import cv2

# Đọc ảnh
img_bgr = cv2.imread("path/to/image.png")

# Nhận diện (tự động invert nếu ảnh giấy)
label, probs = predict_extended(img_bgr, auto_invert=True)

if label:
    print(f"Kết quả: {label}")
    print(f"Độ tin cậy: {max(probs)*100:.2f}%")
```

### 3. Sử dụng với ảnh đã preprocess

```python
from tf.infer_tf_extended import predict_extended_from_array
from preprocess import preprocess_digit_from_bgr
import cv2

# Preprocess ảnh
img_bgr = cv2.imread("path/to/image.png")
digit_array = preprocess_digit_from_bgr(img_bgr, auto_invert=True)

# Predict
label, probs = predict_extended_from_array(digit_array)
```

## Huấn luyện mô hình

### Train model chữ số (0-9)

```bash
# Train với tất cả dataset có sẵn (mặc định)
python tf/train_cnn_tf.py

# Train với dataset cụ thể
python tf/train_cnn_tf.py mnist_png

# Train với số epochs tùy chọn
python tf/train_cnn_tf.py all 20
```

**Tham số:**
- `dataset`: `"all"` (mặc định), `"mnist_png"`, hoặc `"dataset-main"`
- `epochs`: Số epochs (mặc định: 10)

**Kết quả:**
- Model được lưu tại: `models/cnn_digit_tf.keras`
- Accuracy thường đạt ~99.4% trên test set

### Train model mở rộng (0-9 + A, B, C)

```bash
# Train với 20 epochs (mặc định)
python tf/train_cnn_extended.py

# Train với số epochs tùy chọn
python tf/train_cnn_extended.py 25
```

**Đặc điểm:**
- Sử dụng class weights để tập trung vào chữ cái (đặc biệt B dễ nhầm với 6)
- Data augmentation tối ưu: rotation giảm xuống 8° (tránh B→6), thêm shear
- Early stopping với restore best weights
- Kết hợp MNIST + EMNIST Letters + dataset_extra

**Kết quả:**
- Model được lưu tại: `models/cnn_digit_letter_tf.keras`
- Accuracy thường đạt ~95%+ trên test set (13 classes)

### Train với PyTorch (Tham khảo)

```bash
# Train CNN
python pytorch/train_cnn.py [dataset] [epochs]

# Train MLP
python pytorch/train_mlp.py [dataset] [epochs]

# Train SVM
python pytorch/train_svm.py [dataset]
```

## 🔬 Thuật toán & Kiến trúc

### 1. Tiền xử lý (Preprocessing)

Pipeline xử lý ảnh từ GUI hoặc file:

1. **Grayscale & Gaussian Blur**: Chuyển sang grayscale và làm mịn (kernel 3x3)
2. **Thresholding**:
   - **Otsu Threshold**: Cho ảnh vẽ từ GUI (ánh sáng đều)
   - **Adaptive Threshold**: Cho ảnh giấy (xử lý bóng, gradient)
3. **Contour Detection**: Tìm và lọc các contour hợp lệ
   - Loại bỏ contour quá nhỏ (< 8x8 pixels)
   - Loại bỏ contour chạm viền (thường là khung vẽ)
   - Gộp tất cả contour hợp lệ thành 1 bounding box
4. **ROI Cropping**: Cắt vùng chứa chữ số/chữ cái
5. **Aspect Ratio Presizing**: Resize về 22x22 giữ nguyên tỉ lệ
6. **Centering**: Căn giữa vào canvas 28x28 (nền đen)
7. **Normalization**: Chuẩn hóa về [0, 1] (float32)

**Auto-invert**: Tự động phát hiện và đảo màu nếu ảnh có nền sáng (mean brightness > 127)

### 2. Kiến trúc CNN

**Model chữ số (10 classes):**
```
Input: (28, 28, 1)
  ↓
Conv2D(32, 3x3) + ReLU
MaxPooling2D(2x2)  → (14, 14, 32)
  ↓
Conv2D(64, 3x3) + ReLU
MaxPooling2D(2x2)  → (7, 7, 64)
  ↓
Conv2D(64, 3x3) + ReLU
MaxPooling2D(2x2)  → (3, 3, 64)
Dropout(0.25)
  ↓
Flatten  → 576
  ↓
Dense(128) + ReLU
Dropout(0.5)
  ↓
Dense(10) + Softmax  → Output: 10 classes
```

**Model mở rộng (13 classes):**
- Tương tự nhưng output layer là Dense(13)
- Sử dụng class weights: A=2.0, B=3.0, C=1.6

**Hyperparameters:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Sparse Categorical Crossentropy
- Batch size: 64
- Data augmentation: rotation, shift, zoom, shear

### 3. Data Augmentation

- **Rotation**: ±8° (extended) hoặc ±15° (basic)
- **Width/Height Shift**: ±10%
- **Zoom**: ±10%
- **Shear**: ±10% (chỉ extended model)

## Dataset

### Dataset có sẵn

1. **MNIST** (`mnist_png/`):
   - 60,000 ảnh train, 10,000 ảnh test
   - Format: Nền đen, chữ trắng
   - Classes: 0-9

2. **EMNIST Letters** (`dataset_letters/`):
   - Chữ cái A, B, C
   - Format: Tự động chuẩn hóa

3. **Dataset Extra** (`dataset_extra/`):
   - Mẫu tự vẽ từ GUI
   - Cấu trúc: `dataset_extra/<label>/*.png`
   - Tự động merge vào train set

### Thêm mẫu training

1. Chạy GUI: `python tf/app_gui_tf.py`
2. Vẽ mẫu và chọn nhãn đúng
3. Bấm "Lưu mẫu"
4. Mẫu được lưu vào `dataset_extra/<label>/`
5. Train lại model để sử dụng mẫu mới

## Troubleshooting

### Lỗi: "Model not found"

**Nguyên nhân**: Model chưa được train hoặc đường dẫn sai.

**Giải pháp**:
```bash
# Train model trước
python tf/train_cnn_tf.py
# hoặc
python tf/train_cnn_extended.py
```

### Lỗi: "No module named 'tensorflow'"

**Giải pháp**:
```bash
pip install tensorflow>=2.13.0
```

### Lỗi: "Không nhận diện được"

**Nguyên nhân có thể**:
- Ảnh trống hoặc quá nhỏ
- Nét vẽ quá mỏng
- Ảnh có nền sáng (cần auto_invert=True)

**Giải pháp**:
- Tăng độ dày nét vẽ (slider)
- Vẽ rõ ràng hơn, không quá nhỏ
- Với ảnh giấy, dùng nút "Mở ảnh" (tự động invert)

### Lỗi khi train: "Out of memory"

**Giải pháp**:
- Giảm batch size trong code (mặc định 64 → 32 hoặc 16)
- Giảm số epochs
- Chỉ load một phần dataset (thêm `max_per_class` parameter)

### Model accuracy thấp

**Cải thiện**:
- Tăng số epochs
- Thêm nhiều mẫu vào `dataset_extra/`
- Điều chỉnh data augmentation
- Kiểm tra preprocessing có đúng không

### GUI không hiển thị

**Nguyên nhân**: Thiếu thư viện GUI.

**Giải pháp**:
```bash
# tkinter thường có sẵn với Python
# Nếu thiếu, cài đặt:
# Ubuntu/Debian: sudo apt-get install python3-tk
# macOS: tkinter thường có sẵn
# Windows: tkinter có sẵn
```

## Ghi chú

- Model TensorFlow được khuyến nghị sử dụng (ổn định, dễ deploy)
- Model PyTorch trong `pytorch/` chỉ để tham khảo
- File test (`test_*.py`) và convert (`convert_*.py`) không được cập nhật trong repo
- CSV files được ignore trong git (xem `.gitignore`)