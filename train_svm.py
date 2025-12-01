import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from joblib import dump
from skimage.feature import hog

from dataset_main import load_dataset_from_folder

def hog_features(images):
    feats = []
    for img in images:
        # img: (28,28) float32 [0,1]
        img = (img * 255).astype("uint8")
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        feats.append(feat)
    return np.array(feats, dtype="float32")

def main():
    print("=== LOAD DATASET ===")
    X_train, y_train, X_test, y_test = load_dataset_from_folder("dataset-main")

    print("=== EXTRACT HOG FEATURES ===")
    X_train_hog = hog_features(X_train)
    X_test_hog  = hog_features(X_test)
    print("Train HOG shape:", X_train_hog.shape)

    print("=== TRAIN SVM (RBF) ===")
    clf = svm.SVC(kernel='rbf', gamma='scale')
    
    clf.fit(X_train_hog, y_train)

    print("=== EVALUATE ===")
    y_pred = clf.predict(X_test_hog)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy (HOG+SVM):", acc)

    os.makedirs("models", exist_ok=True)
    dump(clf, "models/svm_hog.joblib")
    print("Saved model to models/svm_hog.joblib")

if __name__ == "__main__":
    main()
