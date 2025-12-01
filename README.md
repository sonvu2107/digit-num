# Digit Num - Nhận diện chữ số viết tay

Dự án nhỏ: GUI vẽ chữ số (canvas nền đen, nét trắng) và 2 bộ dự đoán đã train sẵn:
- SVM (HOG features) — `models/svm_hog.joblib`
- MLP (pixel 28x28) — `models/mlp_digit.joblib`

Mục tiêu
- Cho phép vẽ chữ số bằng chuột trong GUI và so sánh dự đoán giữa SVM và MLP.
- Có pipeline tiền xử lý (`preprocess.py`) để đưa ảnh GUI về định dạng 28×28 giống dataset train.

**Lưu ý quan trọng:** dataset train có ảnh *nền trắng, chữ đen*. GUI mặc định là *nền đen, chữ trắng*. `preprocess.py` đã đảo màu và chuẩn hóa để phù hợp với dataset.

**Khuyến nghị khi vẽ để test:**
- Đặt `LINE_WIDTH` trong `app_gui.py` khoảng `12`–`18` (mặc định `18`).
- Không đặt `LINE_WIDTH` quá nhỏ (ví dụ `1`) vì sau khi crop+resize xuống 28×28 nét sẽ bị mất, dẫn tới lỗi dự đoán.
- Vẽ chữ to, chiếm 60–80% chiều cao canvas.

Cấu trúc chính của repo
```
app_gui.py          # GUI vẽ + gọi infer
preprocess.py       # Tiền xử lý ảnh từ GUI -> 28x28 float32 [0,1]
infer.py            # wrapper gọi SVM và MLP
train_mlp.py        # script train MLP (nếu cần train lại)
train_svm.py        # script train SVM (nếu cần train lại)
models/             # chứa model đã train (.joblib)
dataset-main/       # chứa train/test dataset (28x28 images)
```

Yêu cầu môi trường
- Python 3.8+ (cũng chạy trên 3.10+)
- Các thư viện (cài bằng pip):

```powershell
pip install numpy opencv-python pillow scikit-image scikit-learn joblib
# Nếu chạy model CNN (TensorFlow), cài tensorflow tương ứng
# pip install "tensorflow<2.16"
```

Chạy GUI
```powershell
python app_gui.py
```
- Vẽ chữ số bằng chuột, nhấn `Nhận diện` để gọi 2 model.
- Kết quả hiển thị trên giao diện (SVM, MLP và confidence nếu có).

Tiền xử lý (`preprocess.py`) — tóm tắt
- Input: ảnh BGR (OpenCV) từ GUI, kích thước ví dụ `280x280`, nền đen, chữ trắng.
- Bước chính:
  1. Chuyển sang grayscale, blur nhẹ.
  2. Threshold nhị phân để tách chữ trắng trên nền đen.
  3. Tìm contour lớn nhất, crop vùng chữ.
  4. Deskew (dựng thẳng chữ nếu nghiêng).
  5. Resize (giữ tỉ lệ) sao cho phần lớn chữ có kích thước ~20px, pad ra `28x28` với nền trắng.
  6. Đảo màu (từ chữ trắng nền đen -> chữ đen nền trắng).
  7. Chuẩn hóa về float32 trong `[0,1]` và trả về.

Ghi chú: Trong code hiện tại đã thêm `canvas = 1.0 - canvas` để đảo màu phù hợp với dataset train.

Debug & lưu ảnh 28×28 để so sánh
- Nếu vẫn gặp nhiều lỗi dự đoán, bạn có thể lưu ảnh `28x28` sau bước preprocess ra file để xem model thực sự nhận gì.
- Ví dụ chỉnh `preprocess.py` (tạm thời) để lưu:

```python
cv2.imwrite(f"debug_{label_or_idx}.png", (canvas*255).astype('uint8'))
```

(hoặc thêm tham số `save_debug=True` cho hàm `preprocess_digit_from_bgr` rồi gọi từ `app_gui.py` khi cần).

Điều chỉnh cần cân nhắc nếu vẫn có lỗi
- Tăng `LINE_WIDTH` trong `app_gui.py` (12–18).
- Bỏ hoặc không dùng `erode` trong `preprocess.py` (erode làm mảnh nét hơn).
- Tinh chỉnh blur/deskew/resize nếu ảnh GUI khác nhiều so với tập train.

Chạy training (nếu muốn train lại)
- MLP:
```powershell
python train_mlp.py
```
- SVM (HOG):
```powershell
python train_svm.py
```
- CNN (nếu có script):
```powershell
python train_cnn.py
```

Cách báo lỗi / góp ý
- Nếu bạn thấy model vẫn đoán sai nhiều, gửi kèm các ảnh `debug_*.png` (ảnh 28×28 sau preprocess) + ảnh chụp GUI gốc để mình xem và đề xuất điều chỉnh (blur/deskew/resize hoặc augmentation).

License & nguồn
- Đây là một dự án demo/nội bộ, không có license cụ thể.

---
