"""
infer.py

Hàm wrapper để gọi 2 model đã train sẵn:
- SVM dùng HOG features (file: models/svm_hog.joblib)
- MLP dùng pixel 28x28 (file: models/mlp_digit.joblib)

Input cho cả 2 hàm dự đoán là ảnh BGR (OpenCV) từ GUI.
Tiền xử lý được thực hiện bởi `preprocess.preprocess_digit_from_bgr`
Hàm `preprocess` trả về ảnh 28x28 float32 trong [0,1], chữ MÀU ĐEN (gần 0), nền MÀU TRẮNG (gần 1).
"""

import numpy as np
import cv2
from joblib import load as joblib_load
from skimage.feature import hog

from preprocess import preprocess_digit_from_bgr

# Load 2 model đã train sẵn từ thư mục `models/`
svm_model = joblib_load("models/svm_hog.joblib")    # HOG + SVM
mlp_model = joblib_load("models/mlp_digit.joblib")  # MLP (pixel)

# === THAM SỐ HOG (PHẢI khớp với train_svm.py) ===
# Nếu thay đổi các tham số này, cần train lại model SVM
HOG_ORIENTATIONS = 12         # số hướng gradient (tăng từ 9 để capture nét mảnh)
HOG_PIXELS_PER_CELL = (4, 4)  # kích thước mỗi cell (pixel)
HOG_CELLS_PER_BLOCK = (2, 2)  # số cell trong mỗi block
HOG_BLOCK_NORM = 'L2-Hys'     # chuẩn hoá block


def hog_features_single(img_28):
    """Tạo HOG feature từ ảnh 28x28 chuẩn.

    Tham số:
    - img_28: ảnh 28x28 float32 trong [0,1], chữ đen nền trắng
    Trả về:
    - feature 1-d vector dạng (1, N) phù hợp input cho SVM.
    
    Lưu ý: Các tham số HOG phải khớp với tham số đã dùng khi train model.
    """
    img = (img_28 * 255).astype("uint8")
    feat = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM
    )
    return feat.reshape(1, -1)

def predict_with_svm(img_bgr):
    """Dự đoán nhãn bằng pipeline HOG -> SVM.

    Trả về tuple `(pred, probs)`:
    - pred: int label (0-9) hoặc None nếu ảnh trống
    - probs: array xác suất [p0, p1, ..., p9] nếu model hỗ trợ, hoặc None
    
    Lưu ý: SVM cần train với probability=True để có predict_proba.
    Nếu không có, probs sẽ là None.
    """
    digit_28 = preprocess_digit_from_bgr(img_bgr)
    if digit_28 is None:
        return None, None

    feat = hog_features_single(digit_28)
    pred = int(svm_model.predict(feat)[0])
    
    # Lấy probability nếu model hỗ trợ
    probs = None
    if hasattr(svm_model, "predict_proba"):
        try:
            probs = svm_model.predict_proba(feat)[0]
        except:
            pass  # Model không hỗ trợ probability
    
    return pred, probs

def predict_with_mlp(img_bgr):
    """Dự đoán nhãn bằng MLP trên pixel 28x28 flattened.

    Trả về tuple `(pred, probs)`:
    - pred: int label
    - probs: dạng vector xác suất nếu model hỗ trợ `predict_proba`, hoặc `None`.
    Nếu ảnh không có chữ sẽ trả về `(None, None)`.
    """
    digit_28 = preprocess_digit_from_bgr(img_bgr)
    if digit_28 is None:
        return None, None

    x_flat = digit_28.reshape(1, -1)

    probs = None
    if hasattr(mlp_model, "predict_proba"):
        probs = mlp_model.predict_proba(x_flat)[0]

    pred = int(mlp_model.predict(x_flat)[0])

    return pred, probs

if __name__ == "__main__":
    # test nhanh
    img = cv2.imread("test.png")  # đổi ảnh test
    svm_pred, svm_probs = predict_with_svm(img)
    print("SVM:", svm_pred, "probs:", svm_probs)
    mlp_pred, mlp_probs = predict_with_mlp(img)
    print("MLP:", mlp_pred, "probs:", mlp_probs)
