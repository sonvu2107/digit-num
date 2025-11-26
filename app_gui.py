"""
app_gui.py

Giao diện GUI đơn giản để vẽ chữ số bằng chuột và gọi 2 bộ dự đoán:
- SVM (dùng HOG)
- MLP (dùng pixel 28x28)

Hướng dẫn nhanh:
- Canvas có nền đen, vẽ bằng màu trắng.
- `LINE_WIDTH` nên đặt trong khoảng 12-18 (không đặt quá nhỏ <8), vì sau khi crop+resize xuống 28x28 nét sẽ mảnh lại.
"""

import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import cv2

from infer import predict_with_svm, predict_with_mlp


CANVAS_SIZE = 280   # kích thước canvas hiển thị (pixel)
# Độ dày nét vẽ khi vẽ trên canvas lớn. Khi resize về 28x28 sẽ suy giảm xuống ~1-2px.
LINE_WIDTH = 18   # khuyến nghị: 12-18. KHÔNG đặt < 8.


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ số viết tay - SVM & MLP")

        # Canvas để vẽ
        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross"
        )
        self.canvas.grid(row=0, column=0, rowspan=4, padx=10, pady=10)

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
        self.label_svm = tk.Label(root, text="SVM: ...", font=("Arial", 14))
        self.label_svm.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        self.label_mlp = tk.Label(root, text="MLP: ...", font=("Arial", 14))
        self.label_mlp.grid(row=3, column=1, sticky="w", padx=10, pady=5)

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
        self.label_svm.config(text="SVM: ...")
        self.label_mlp.config(text="MLP: ...")

    def on_predict(self):
        # Chuyển ảnh PIL (RGB) -> numpy -> BGR cho OpenCV
        img_rgb = np.array(self.image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Gọi model SVM
        try:
            svm_pred = predict_with_svm(img_bgr)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán bằng SVM:\n{e}")
            return

        # Gọi model MLP
        try:
            mlp_pred, probs = predict_with_mlp(img_bgr)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán bằng MLP:\n{e}")
            return

        if svm_pred is None and mlp_pred is None:
            messagebox.showwarning("Cảnh báo", "Không nhận diện được chữ số (ảnh trống?).")
            return

        self.label_svm.config(text=f"SVM: {svm_pred if svm_pred is not None else 'N/A'}")
        if mlp_pred is not None:
            if probs is not None:
                conf = np.max(probs)
                self.label_mlp.config(text=f"MLP: {mlp_pred} (conf: {conf:.2f})")
            else:
                self.label_mlp.config(text=f"MLP: {mlp_pred}")
        else:
            self.label_mlp.config(text="MLP: N/A")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
