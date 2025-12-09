"""
app_gui_tf.py - GUI vẽ và nhận diện chữ số (TensorFlow)
Nền TRẮNG, chữ ĐEN (giống dataset)

Sử dụng: python app_gui_tf.py
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import cv2

from infer_tf import predict_with_cnn_tf

CANVAS_SIZE = 280
LINE_WIDTH = 22


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ số - CNN (TensorFlow)")

        # Canvas nền TRẮNG
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, 
                                bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)

        # Ảnh PIL nền TRẮNG
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_pos)

        tk.Button(root, text="Nhận diện", command=self.on_predict).grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        tk.Button(root, text="Xoá", command=self.clear_canvas).grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        self.label_result = tk.Label(root, text="Kết quả: ...", font=("Arial", 16, "bold"), fg="blue")
        self.label_result.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        self.last_x = self.last_y = None

    def start_pos(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.last_x and self.last_y:
            # Vẽ nét ĐEN
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=LINE_WIDTH, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill="black", width=LINE_WIDTH)
        self.last_x, self.last_y = event.x, event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Kết quả: ...")

    def on_predict(self):
        self.label_result.config(text="Đang xử lý...")
        self.root.update()
        self.root.after(100, self._do_predict)

    def _do_predict(self):
        img_bgr = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        
        try:
            pred, probs = predict_with_cnn_tf(img_bgr)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi: {e}")
            return

        if pred is None:
            messagebox.showwarning("Cảnh báo", "Không nhận diện được (ảnh trống?).")
            self.label_result.config(text="Kết quả: ...")
            return

        conf = probs[pred] * 100 if probs is not None else 0
        self.label_result.config(text=f"Kết quả: {pred} ({conf:.1f}%)")


if __name__ == "__main__":
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()
