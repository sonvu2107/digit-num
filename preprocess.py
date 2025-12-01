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
    h, w = img.shape
    M = np.float32([[1, skew, -0.5 * skew * h],
                    [0,   1,                0]])
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )


# === THAM SỐ TIỀN XỬ LÝ (điều chỉnh dựa trên phân tích dataset) ===
# Dataset có nét mảnh (median ~0.6px), digit chiếm gần hết 28x28
BLUR_KERNEL = (3, 3)    # Kernel nhỏ để giữ chi tiết nét mảnh (trước: 5x5)
RESIZE_TARGET = 22      # Kích thước digit sau resize, trước khi pad về 28x28 (trước: 20)

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
    # chữ trắng, nền đen
    digit = th[y:y+h, x:x+w]   

    # 4. Deskew
    digit = deskew(digit)

    # 5. Resize + pad về 28x28 (digit chiếm RESIZE_TARGET pixel)
    h, w = digit.shape
    if h > w:
        new_h = RESIZE_TARGET
        new_w = max(1, int(w * (RESIZE_TARGET / h)))
    else:
        new_w = RESIZE_TARGET
        new_h = max(1, int(h * (RESIZE_TARGET / w)))

    digit_resized = cv2.resize(digit, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)

    # 6. Canvas nền trắng (255), chữ đen
    canvas = np.full((28, 28), 255, dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2

    # chữ trắng -> đen
    digit_final = 255 - digit_resized
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_final

    # 7. Chuẩn hoá [0,1]
    canvas = canvas.astype("float32") / 255.0
    return canvas
