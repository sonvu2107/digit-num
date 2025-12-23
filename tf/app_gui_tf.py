"""
app_gui_tf.py - GUI vẽ và nhận diện chữ số + chữ cái (TensorFlow)
Version:
 - Hỗ trợ 13 classes: 0-9 + A, B, C
 - Slider chỉnh độ dày nét
 - Lưu debug ảnh preprocess
 - Nút "Lưu mẫu" vào dataset_extra để train tiếp
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import sys
import time

# Thêm thư mục cha vào path để import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infer_tf_extended import predict_extended, predict_extended_from_array, LABEL_NAMES
from preprocess import preprocess_digit_from_bgr

CANVAS_SIZE = 280
DEFAULT_LINE_WIDTH = 12

# Đường dẫn thư mục (từ thư mục gốc)
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_SAVE_DIR = os.path.join(ROOT_DIR, "dataset_extra")
DEBUG_IMAGES_DIR = os.path.join(ROOT_DIR, "debug_images")

# Đảm bảo các thư mục tồn tại
os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
for label in LABEL_NAMES:  # 0-9 + A, B, C
    os.makedirs(os.path.join(DATASET_SAVE_DIR, label), exist_ok=True)


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ số & chữ cái - CNN (TensorFlow)")

        # Slider độ dày nét
        self.line_width = tk.IntVar(value=DEFAULT_LINE_WIDTH)

        # Canvas nền đen (giống MNIST)
        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

        # Ảnh PIL nền đen (giống MNIST)
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_pos)

        # Nút Nhận diện
        tk.Button(root, text="Nhận diện", command=self.on_predict)\
            .grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        # Nút Xoá
        tk.Button(root, text="Xoá", command=self.clear_canvas)\
            .grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        # Slider UI
        tk.Label(root, text="Độ dày nét:")\
            .grid(row=2, column=1, sticky="w", padx=10, pady=(10, 0))

        tk.Scale(
            root, from_=8, to=28, orient=tk.HORIZONTAL,
            variable=self.line_width, showvalue=True, length=160
        ).grid(row=3, column=1, sticky="ew", padx=10, pady=5)

        # Chọn nhãn để lưu (0-9 + A, B, C)
        tk.Label(root, text="Nhãn (0-9, A-C):")\
            .grid(row=4, column=1, sticky="w", padx=10, pady=(10, 0))

        self.label_var = tk.StringVar(value="0")
        self.label_combo = tk.OptionMenu(root, self.label_var, *LABEL_NAMES)
        self.label_combo.config(width=5)
        self.label_combo.grid(row=5, column=1, sticky="w", padx=10, pady=5)

        # Nút Lưu mẫu
        tk.Button(root, text="Lưu mẫu", command=self.save_sample)\
            .grid(row=6, column=1, sticky="ew", padx=10, pady=5)

        # Nút Mở ảnh từ máy
        tk.Button(root, text="Mở ảnh", command=self.open_image, bg="#4CAF50", fg="white")\
            .grid(row=7, column=1, sticky="ew", padx=10, pady=5)

        # Label kết quả
        self.label_result = tk.Label(
            root, text="Kết quả: ...",
            font=("Arial", 16, "bold"), fg="blue"
        )
        self.label_result.grid(row=6, column=0, sticky="w", padx=10, pady=5)

        self.last_x = self.last_y = None

    # ==== Vẽ ====
    def start_pos(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.last_x is not None and self.last_y is not None:
            w = self.line_width.get()
            # Vẽ lên canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=w, fill="white", capstyle=tk.ROUND, smooth=True
            )
            # Vẽ lên ảnh PIL
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill="white", width=w
            )
        self.last_x, self.last_y = event.x, event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Kết quả: ...")
        self.last_x = self.last_y = None

    # ==== Nhận diện ====
    def on_predict(self):
        self.label_result.config(text="Đang xử lý...")
        self.root.update()
        self.root.after(50, self._do_predict)

    def _do_predict(self):
        img_bgr = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)

        # Preprocess để debug + reuse
        digit = preprocess_digit_from_bgr(img_bgr)
        if digit is None:
            messagebox.showwarning("Cảnh báo", "Không nhận diện được (ảnh trống?).")
            self.label_result.config(text="Kết quả: ...")
            return

        # Lưu debug ảnh preprocess
        dbg = (digit * 255).astype("uint8")
        debug_path = os.path.join(DEBUG_IMAGES_DIR, "debug_preprocess.png")
        cv2.imwrite(debug_path, dbg)

        # Gọi model predict
        try:
            pred, probs = predict_extended(img_bgr)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi: {e}")
            return

        if pred is None or probs is None:
            self.label_result.config(text="Kết quả: ...")
            return

        conf = float(max(probs)) * 100.0
        self.label_result.config(text=f"Kết quả: {pred} ({conf:.1f}%)")

    # ==== Lưu mẫu vào dataset_extra/<label>/... ====
    def save_sample(self):
        img_bgr = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        digit = preprocess_digit_from_bgr(img_bgr)
        if digit is None:
            messagebox.showwarning("Cảnh báo", "Không có chữ để lưu (ảnh trống?).")
            return

        # Lấy nhãn từ dropdown
        label = self.label_var.get()
        if label not in LABEL_NAMES:
            messagebox.showerror("Lỗi", f"Nhãn không hợp lệ: {label}")
            return

        # Chuyển digit (28x28 float [0,1]) -> uint8 để lưu PNG
        img_28 = (digit * 255).astype("uint8")

        # Đường dẫn thư mục theo nhãn
        label_dir = os.path.join(DATASET_SAVE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        # Tạo tên file duy nhất theo timestamp
        filename = time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}.png"
        save_path = os.path.join(label_dir, filename)

        cv2.imwrite(save_path, img_28)
        messagebox.showinfo("Đã lưu", f"Đã lưu mẫu vào:\n{save_path}")

    # ==== Mở ảnh từ máy ====
    def open_image(self):
        """Mở ảnh từ máy tính, hiển thị lên canvas và nhận diện."""
        filepath = filedialog.askopenfilename(
            title="Chọn ảnh chữ số/chữ cái",
            filetypes=[
                ("Ảnh", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("Tất cả", "*.*")
            ]
        )
        if not filepath:
            return

        # Đọc ảnh từ file
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            messagebox.showerror("Lỗi", f"Không thể đọc ảnh:\n{filepath}")
            return

        # Resize về CANVAS_SIZE để hiển thị
        img_resized = cv2.resize(img_bgr, (CANVAS_SIZE, CANVAS_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Cập nhật canvas và ảnh PIL
        self.image = Image.fromarray(img_rgb)
        self.draw = ImageDraw.Draw(self.image)

        # Hiển thị lên canvas
        self.canvas.delete("all")
        from PIL import ImageTk
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Tự động nhận diện với auto_invert=True (cho ảnh giấy)
        self.label_result.config(text="Đang xử lý ảnh...")
        self.root.update()
        
        # Preprocess với auto_invert để xử lý ảnh giấy trắng chữ đen
        digit = preprocess_digit_from_bgr(img_resized, auto_invert=True)
        if digit is None:
            messagebox.showwarning("Cảnh báo", "Không nhận diện được trong ảnh.")
            self.label_result.config(text="Kết quả: ...")
            return

        # Lưu debug ảnh preprocess
        dbg = (digit * 255).astype("uint8")
        debug_path = os.path.join(DEBUG_IMAGES_DIR, "debug_preprocess_upload.png")
        cv2.imwrite(debug_path, dbg)

        # Predict
        try:
            pred, probs = predict_extended_from_array(digit)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi nhận diện: {e}")
            return

        if pred is None or probs is None:
            self.label_result.config(text="Kết quả: ...")
            return

        conf = float(max(probs)) * 100.0
        self.label_result.config(text=f"Kết quả: {pred} ({conf:.1f}%)")


if __name__ == "__main__":
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()
