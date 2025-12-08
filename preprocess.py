"""
preprocess.py - Tiền xử lý ảnh từ GUI về format 28x28 chuẩn

Pipeline xử lý:
    1. BGR -> Grayscale -> Blur nhẹ
    2. Threshold Otsu: tách chữ trắng trên nền đen
    3. Tìm contour lớn nhất -> crop vùng chữ
    4. Deskew: chỉnh nghiêng dựa trên moment
    5. Resize giữ tỉ lệ về RESIZE_TARGET pixel
    6. Căn giữa vào canvas 28x28, đảo màu (chữ đen, nền trắng)
    7. Chuẩn hoá [0,1]

Input:  ảnh BGR từ GUI (280x280), nền ĐEN, chữ TRẮNG
Output: numpy (28,28) float32 [0,1], chữ ĐEN (~0), nền TRẮNG (~1)
"""

import cv2
import numpy as np

# === THAM SỐ TIỀN XỬ LÝ ===
BLUR_KERNEL = (3, 3)    # Gaussian blur kernel (giữ nhỏ để không mất chi tiết)
RESIZE_TARGET = 22      # Kích thước digit sau resize (để lại margin khi căn giữa)


def deskew(img):
    """Chỉnh nghiêng ảnh dựa trên image moments.
    
    Thuật toán:
        - Tính moment ảnh để xác định độ nghiêng (skew)
        - Skew = mu11 / mu02 (tỉ lệ moment bậc 2)
        - Áp dụng affine transform để chỉnh thẳng
    
    Args:
        img: ảnh nhị phân (0/255), chữ trắng nền đen
    
    Returns:
        ảnh đã chỉnh nghiêng
    """
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    
    # Tính độ nghiêng, giới hạn [-1, 1] tránh biến dạng quá mức
    skew = m['mu11'] / m['mu02']
    skew = np.clip(skew, -1.0, 1.0)
    
    h, w = img.shape
    # Ma trận affine: shear theo chiều ngang
    M = np.float32([[1, skew, -0.5 * skew * h],
                    [0, 1, 0]])
    
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def preprocess_digit_from_bgr(img_bgr):
    """Tiền xử lý ảnh BGR từ GUI về format 28x28 chuẩn.
    
    Args:
        img_bgr: ảnh BGR từ GUI (thường 280x280), nền đen, chữ trắng
    
    Returns:
        numpy (28,28) float32 [0,1], chữ đen nền trắng
        None nếu không tìm thấy chữ
    """
    # 1. BGR -> Grayscale + Blur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    # 2. Threshold Otsu: tự động chọn ngưỡng tối ưu
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Tìm contour chữ (chữ trắng trên nền đen)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Lấy contour lớn nhất (giả sử là chữ số)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    if w < 5 or h < 5:
        return None
    
    # Crop vùng chữ
    digit = th[y:y+h, x:x+w]

    # 4. Deskew
    digit = deskew(digit)

    # 5. Resize giữ tỉ lệ
    h, w = digit.shape
    if h > w:
        new_h = RESIZE_TARGET
        new_w = max(1, int(w * (RESIZE_TARGET / h)))
    else:
        new_w = RESIZE_TARGET
        new_h = max(1, int(h * (RESIZE_TARGET / w)))

    digit_resized = cv2.resize(digit, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
    
    # Threshold lại để đảm bảo nhị phân sau resize
    _, digit_resized = cv2.threshold(digit_resized, 127, 255, cv2.THRESH_BINARY)

    # 6. Căn giữa vào canvas 28x28, đảo màu
    canvas = np.full((28, 28), 255, dtype=np.uint8)  # Nền trắng
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    
    digit_final = 255 - digit_resized  # Đảo: trắng->đen
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_final

    # 7. Chuẩn hoá [0,1]
    return canvas.astype("float32") / 255.0
