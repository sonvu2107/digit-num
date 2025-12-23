"""
preprocess.py - Tiền xử lý ảnh từ GUI về format 28x28

Pipeline xử lý:
    1. BGR -> Grayscale -> Blur nhẹ
    2. Threshold Otsu: giữ chữ trắng trên nền đen (input từ GUI đã đúng MNIST)
    3. Tìm contour lớn nhất -> crop vùng chữ
    4. Deskew: chỉnh nghiêng dựa trên moment
    5. Resize giữ tỉ lệ về RESIZE_TARGET pixel
    6. Căn giữa vào canvas 28x28
    7. Chuẩn hoá [0,1]

Input:  ảnh BGR từ GUI (280x280), nền ĐEN, chữ TRẮNG (đúng MNIST format)
Output: numpy (28,28) float32 [0,1], nền ĐEN (~0), chữ TRẮNG (~1) - chuẩn MNIST
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


def preprocess_digit_from_bgr(img_bgr, auto_invert=False):
    """Tiền xử lý ảnh BGR từ GUI về format 28x28 chuẩn MNIST.
    
    Args:
        img_bgr: ảnh BGR từ GUI (thường 280x280), nền ĐEN, chữ TRẮNG
        auto_invert: Nếu True, tự động phát hiện và đảo màu nếu nền sáng (ảnh giấy)
    
    Returns:
        numpy (28,28) float32 [0,1], nền đen (~0), chữ trắng (~1) - chuẩn MNIST
        None nếu không tìm thấy chữ
    """
    # 1. BGR -> Grayscale + Blur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    # Auto-invert: phát hiện nền sáng (ảnh giấy trắng chữ đen)
    is_light_background = False
    if auto_invert:
        # Tính trung bình độ sáng, nếu > 127 thì nền sáng
        mean_brightness = np.mean(gray)
        is_light_background = mean_brightness > 127
        print(f"[preprocess] Mean brightness: {mean_brightness:.1f}, is_light_background: {is_light_background}")

    # 2. Threshold - chọn phương pháp phù hợp
    if auto_invert:
        # Adaptive Threshold: xử lý tốt ảnh có ánh sáng không đều (bóng, gradient)
        # ADAPTIVE_THRESH_GAUSSIAN_C: dùng weighted sum của vùng lân cận
        # Block size = 21: kích thước vùng lân cận (phải lẻ)
        # C = 10: hằng số trừ đi (điều chỉnh độ nhạy)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV if is_light_background else cv2.THRESH_BINARY,
                                   21, 10)
        
        # Làm dày nét vẽ (dilation) - nét trên giấy thường mảnh hơn MNIST
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.dilate(th, kernel, iterations=2)
    else:
        # Otsu threshold: cho ảnh vẽ từ GUI (ánh sáng đều)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Tìm contour chữ (chữ trắng trên nền đen)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_h, img_w = th.shape
    margin = 10  # pixels từ viền (tăng lên để filter tốt hơn)
    
    # Lọc contour: loại bỏ những contour chạm viền ảnh (thường là khung bao)
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Bỏ qua contour quá nhỏ
        if w < 8 or h < 8 or area < 50:
            continue
        
        # Bỏ qua contour chạm viền ảnh (thường là khung vẽ bao quanh)
        touches_edge = (x <= margin or y <= margin or 
                        x + w >= img_w - margin or y + h >= img_h - margin)
        
        # Bỏ qua contour chiếm gần hết ảnh (khung bao)
        area_ratio = (w * h) / (img_w * img_h)
        is_too_large = area_ratio > 0.7
        
        if not touches_edge and not is_too_large:
            valid_contours.append(cnt)
    
    # Nếu có valid contours, GỘP TẤT CẢ thành 1 bounding box
    if valid_contours:
        # Gộp tất cả points từ các contour hợp lệ
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
    else:
        # Fallback: lấy contour lớn nhất
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
    
    if w < 5 or h < 5:
        return None
    
    print(f"[preprocess] Combined bounding box: x={x}, y={y}, w={w}, h={h}, valid_contours={len(valid_contours)}")
    
    # Crop vùng chữ
    digit = th[y:y+h, x:x+w]

    # 4. Deskew - TẠM TẮT vì gây lỗi với một số chữ số
    # digit = deskew(digit)

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

    # 6. Căn giữa vào canvas 28x28 (nền đen, chữ trắng)
    canvas = np.zeros((28, 28), dtype=np.uint8)  # Nền đen
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    
    # Giữ nguyên: chữ trắng trên nền đen
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized

    # 7. Chuẩn hoá [0,1]
    return canvas.astype("float32") / 255.0
