"""
app_gui.py - GUI vẽ và nhận diện chữ số viết tay

Giao diện:
    - Canvas 280x280 nền đen để vẽ chữ số bằng chuột
    - Nút "Nhận diện": gọi 3 model (CNN, SVM, MLP)
    - Nút "Xoá": xoá canvas
    - Hiển thị kết quả với confidence %

Sử dụng:
    1. Vẽ chữ số (0-9) bằng chuột trái
    2. Bấm "Nhận diện" để xem kết quả
    3. CNN là model chính xác nhất (hiển thị màu xanh)
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import cv2

from infer import predict_with_svm, predict_with_mlp, predict_with_cnn

# === Tham số GUI ===
CANVAS_SIZE = 280   # Kích thước canvas (10x kích thước ảnh 28x28)
LINE_WIDTH = 22     # Độ dày nét vẽ (sau resize sẽ ~2px, khớp với dataset)


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ số viết tay - CNN & SVM & MLP")

        # Canvas để vẽ
        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross"
        )
        self.canvas.grid(row=0, column=0, rowspan=5, padx=10, pady=10)

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
        self.label_cnn = tk.Label(root, text="CNN: ...", font=("Arial", 14, "bold"), fg="blue")
        self.label_cnn.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        self.label_svm = tk.Label(root, text="SVM: ...", font=("Arial", 14))
        self.label_svm.grid(row=3, column=1, sticky="w", padx=10, pady=5)

        self.label_mlp = tk.Label(root, text="MLP: ...", font=("Arial", 14))
        self.label_mlp.grid(row=4, column=1, sticky="w", padx=10, pady=5)

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
        self.label_cnn.config(text="CNN: ...")
        self.label_svm.config(text="SVM: ...")
        self.label_mlp.config(text="MLP: ...")

    def on_predict(self):
        # Hiển thị "Thinking..." trong khi xử lý
        self.label_cnn.config(text="CNN: Thinking...")
        self.label_svm.config(text="SVM: Thinking...")
        self.label_mlp.config(text="MLP: Thinking...")
        self.root.update()  # Force update GUI
        
        # Delay nhẹ để user thấy hiệu ứng thinking
        self.root.after(300, self._do_predict)
    
    def _do_predict(self):
        """Thực hiện dự đoán sau khi hiển thị thinking"""
        # Chuyển ảnh PIL (RGB) -> numpy -> BGR cho OpenCV
        img_rgb = np.array(self.image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Gọi model CNN (ưu tiên)
        cnn_pred, cnn_probs = None, None
        try:
            cnn_pred, cnn_probs = predict_with_cnn(img_bgr)
        except Exception as e:
            print(f"CNN error: {e}")

        # Gọi model SVM
        try:
            svm_pred, svm_probs = predict_with_svm(img_bgr)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán bằng SVM:\n{e}")
            return

        # Gọi model MLP
        try:
            mlp_pred, mlp_probs = predict_with_mlp(img_bgr)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán bằng MLP:\n{e}")
            return

        if cnn_pred is None and svm_pred is None and mlp_pred is None:
            messagebox.showwarning("Cảnh báo", "Không nhận diện được chữ số (ảnh trống?).")
            self.label_cnn.config(text="CNN: ...")
            self.label_svm.config(text="SVM: ...")
            self.label_mlp.config(text="MLP: ...")
            return

        # Hiển thị kết quả CNN
        if cnn_pred is not None:
            if cnn_probs is not None:
                cnn_conf = cnn_probs[cnn_pred] * 100
                self.label_cnn.config(text=f"CNN: {cnn_pred} ({cnn_conf:.1f}%)")
            else:
                self.label_cnn.config(text=f"CNN: {cnn_pred}")
        else:
            self.label_cnn.config(text="CNN: N/A")

        # Hiển thị kết quả SVM với confidence
        if svm_pred is not None:
            if svm_probs is not None:
                svm_conf = svm_probs[svm_pred] * 100
                self.label_svm.config(text=f"SVM: {svm_pred} ({svm_conf:.1f}%)")
            else:
                self.label_svm.config(text=f"SVM: {svm_pred}")
        else:
            self.label_svm.config(text="SVM: N/A")
        
        # Hiển thị kết quả MLP với confidence
        if mlp_pred is not None:
            if mlp_probs is not None:
                mlp_conf = mlp_probs[mlp_pred] * 100
                self.label_mlp.config(text=f"MLP: {mlp_pred} ({mlp_conf:.1f}%)")
            else:
                self.label_mlp.config(text=f"MLP: {mlp_pred}")
        else:
            self.label_mlp.config(text="MLP: N/A")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
