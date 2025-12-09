"""
train_mlp.py - Train MLP (Multi-Layer Perceptron) classifier

Hỗ trợ 2 dataset:
    - dataset-main: 40k train, 10k test
    - mnist_png: 60k train, 10k test

Sử dụng:
    python train_mlp.py                    # Train với dataset-main
    python train_mlp.py mnist_png          # Train với MNIST
    python train_mlp.py mnist_png 30       # Train với MNIST, 30 epochs

MLP Architecture:
    Input (784) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Output(10, Softmax)
"""

import os
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump

from dataset_main import load_dataset_from_folder


def main():
    # === Parse arguments ===
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset-main"
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print("=" * 60)
    print("       TRAIN MLP MODEL - Pixel Features")
    print("=" * 60)
    print(f"\nDataset: {dataset}")
    print(f"Epochs:  {max_iter}")
    
    # === Load dataset ===
    print("\n[1/3] Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset_from_folder(dataset)
    
    # Flatten 28x28 -> 784
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    print(f"      Input shape: {X_train_flat.shape}")

    # === Train MLP ===
    print("\n[2/3] Training MLP...")
    print("      Architecture: 784 -> 256 -> 128 -> 10")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=128,
        max_iter=max_iter,
        verbose=True,
        random_state=42
    )
    
    mlp.fit(X_train_flat, y_train)

    # === Evaluate ===
    print("\n[3/3] Evaluating...")
    y_pred = mlp.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print("=" * 60)
    
    # Per-digit accuracy
    cm = confusion_matrix(y_test, y_pred)
    print("\n  Per-digit accuracy:")
    for i in range(10):
        correct = cm[i, i]
        total = cm[i].sum()
        bar = "█" * int(correct / total * 20)
        print(f"    {i}: {bar:<20} {correct:>4}/{total} ({100*correct/total:.1f}%)")

    # Save model
    os.makedirs("models", exist_ok=True)
    dump(mlp, "models/mlp_digit.joblib")
    print(f"\n  Saved: models/mlp_digit.joblib")


if __name__ == "__main__":
    main()
