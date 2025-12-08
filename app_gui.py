"""
app_gui.py - GUI vẽ và nhận diện chữ số viết tay

Giao diện:
    - Canvas 280x280 nền đen để vẽ chữ số bằng chuột
    - Nút "Nhận diện": gọi model CNN
    - Nút "Xoá": xoá canvas
    - Hiển thị kết quả với confidence %

Sử dụng:
    1. Vẽ chữ số (0-9) bằng chuột trái
    2. Bấm "Nhận diện" để xem kết quả
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import cv2

from infer import predict_with_cnn

# === Tham số GUI ===
CANVAS_SIZE = 280   # Kích thước canvas (10x kích thước ảnh 28x28)
LINE_WIDTH = 22     # Độ dày nét vẽ (sau resize sẽ ~2px, khớp với dataset)


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ số viết tay - CNN")

        # Canvas để vẽ
        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross"
        )
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)

        # Ảnh PIL để lưu nét vẽ (RGB, nền đen, nét trắng)
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Bind sự kiện chuột
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_pos)

        # Nút bấm
        btn_predict = tk.Button(root, text="Nhận diện", command=self.on_predict)
        btn_predict.grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        btn_clear = tk.Button(root, text="Xoá", command=self.clear_canvas)
        btn_clear.grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        # Label hiển thị kết quả
        self.label_result = tk.Label(root, text="Kết quả: ...", font=("Arial", 16, "bold"), fg="blue")
        self.label_result.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        # Để vẽ mượt
        self.last_x = None
        self.last_y = None

    def start_pos(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # vẽ lên canvas Tkinter
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=LINE_WIDTH,
                fill="white",
                capstyle=tk.ROUND,
                smooth=True
            )
            # vẽ lên ảnh PIL (để xử lý)
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill="white",
                width=LINE_WIDTH
            )
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Kết quả: ...")

    def on_predict(self):
        # Hiển thị "Thinking..." trong khi xử lý
        self.label_result.config(text="Đang xử lý...")
        self.root.update()  # Force update GUI
        
        # Delay nhẹ để user thấy hiệu ứng thinking
        self.root.after(100, self._do_predict)
    
    def _do_predict(self):
        """Thực hiện dự đoán sau khi hiển thị thinking"""
        # Chuyển ảnh PIL (RGB) -> numpy -> BGR cho OpenCV
        img_rgb = np.array(self.image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Gọi model CNN
        try:
            pred, probs = predict_with_cnn(img_bgr)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán:\n{e}")
            return

        if pred is None:
            messagebox.showwarning("Cảnh báo", "Không nhận diện được chữ số (ảnh trống?).")
            self.label_result.config(text="Kết quả: ...")
            return

        # Hiển thị kết quả
        if probs is not None:
            conf = probs[pred] * 100
            self.label_result.config(text=f"Kết quả: {pred} ({conf:.1f}%)")
        else:
            self.label_result.config(text=f"Kết quả: {pred}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
