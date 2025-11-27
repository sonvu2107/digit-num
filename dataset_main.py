import numpy as np
import cv2
import glob
import os

def load_dataset_from_folder(root="dataset-main"):
    """
    root/
      train/
        0/*.png
        ...
        9/*.png
      test/
        0/*.png
        ...
        9/*.png
    """
    def load_split(split):
        imgs = []
        labels = []
        split_dir = os.path.join(root, split)
        for digit in range(10):
            class_dir = os.path.join(split_dir, str(digit))
            pattern = os.path.join(class_dir, "*.png")
            for f in glob.glob(pattern):
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (28, 28))
                imgs.append(img)
                labels.append(digit)
        X = np.array(imgs, dtype="float32") / 255.0
        y = np.array(labels, dtype="int64")
        return X, y

    X_train, y_train = load_split("train")
    X_test, y_test   = load_split("test")

    print("Loaded dataset:")
    print("  Train:", X_train.shape, y_train.shape)
    print("  Test :", X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test
