"""
preprocess.py

Hàm tiền xử lý để chuyển ảnh BGR (từ GUI) về ảnh 28x28 chuẩn dùng cho model.

Yêu cầu input:
- `img_bgr`: ảnh BGR từ GUI (kích thước ví dụ 280x280), nền ĐEN, chữ TRẮNG (canvas GUI mặc định).

Output:
- Trả về `canvas` dạng numpy float32 shape (28,28), giá trị trong [0,1].
- Format trả về: chữ MÀU ĐEN (gần 0), nền MÀU TRẮNG (gần 1). Đây là format giống với dataset `dataset-main/train/...`.

Ghi chú quan trọng:
- Sử dụng threshold để tìm contour chữ (chữ trắng trên nền đen), crop vùng chữ, deskew, resize về kích thước phù hợp rồi đảo màu và chuẩn hoá.
- Thêm bước thinning để làm mảnh nét vẽ, khớp với dataset có nét mảnh (0.3-1px)
"""

import cv2
import numpy as np


def deskew(img):
    """Giảm lệch (deskew) ảnh nhị phân 2D

    Tham số:
    - img: ảnh nhị phân (0 hoặc 255), chuẩn là chữ trắng trên nền đen
    Trả về ảnh đã áp deskew để chữ thẳng đứng hơn trước khi resize.
    """

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    skew = np.clip(skew, -1.0, 1.0)  # Giới hạn skew để tránh biến dạng quá mức
    h, w = img.shape
    M = np.float32([[1, skew, -0.5 * skew * h],
                    [0,   1,                0]])
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )


def thin_stroke(img, iterations=1):
    """Làm mảnh nét vẽ bằng morphological erosion.
    
    Tham số:
    - img: ảnh nhị phân (chữ trắng 255, nền đen 0)
    - iterations: số lần erode (1-2 thường đủ)
    
    Trả về ảnh đã làm mảnh nét.
    """
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=iterations)
    return eroded


# === THAM SỐ TIỀN XỬ LÝ (điều chỉnh dựa trên phân tích dataset) ===
# Dataset là ảnh NHỊ PHÂN (chỉ có 0 và 255), nét khá dày (~100-140 black pixels)
BLUR_KERNEL = (3, 3)    # Blur nhỏ để giữ chi tiết
RESIZE_TARGET = 22      # Kích thước digit sau resize

def preprocess_digit_from_bgr(img_bgr):
    # 1. BGR -> GRAY
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    # 2. Nhị phân: chữ TRẮNG, nền ĐEN (phù hợp canvas)
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. Lấy contour chữ
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Kiểm tra vùng chữ có đủ lớn không
    if w < 5 or h < 5:
        return None
        
    # chữ trắng, nền đen - crop sát vùng chữ
    digit = th[y:y+h, x:x+w]   

    # 4. Deskew - luôn áp dụng để giống dataset
    digit = deskew(digit)

    # 5. Resize giữ tỉ lệ
    h, w = digit.shape
    if h > w:
        new_h = RESIZE_TARGET
        new_w = max(1, int(w * (RESIZE_TARGET / h)))
    else:
        new_w = RESIZE_TARGET
        new_h = max(1, int(h * (RESIZE_TARGET / w)))

    # Dùng INTER_AREA để resize mượt, sau đó threshold lại
    digit_resized = cv2.resize(digit, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
    
    # Threshold sau resize để đảm bảo nhị phân
    _, digit_resized = cv2.threshold(digit_resized, 127, 255, cv2.THRESH_BINARY)

    # 6. Canvas nền trắng (255), chữ đen - căn giữa
    canvas = np.full((28, 28), 255, dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2

    # chữ trắng -> đen (đảo màu)
    digit_final = 255 - digit_resized
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_final

    # 7. Chuẩn hoá [0,1] - dataset là chữ đen (0) trên nền trắng (255->1.0)
    canvas = canvas.astype("float32") / 255.0
    return canvas
