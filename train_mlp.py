import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

from dataset_main import load_dataset_from_folder

def main():
    print("=== LOAD DATASET ===")
    X_train, y_train, X_test, y_test = load_dataset_from_folder("dataset-main")
    # X_*: (N, 28, 28), float32 [0,1]

    # Flatten từ 28x28 -> 784
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)

    print("Train shape:", X_train_flat.shape)

    # MLP: 2 hidden layers: 256, 128 neuron, relu, adam
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=128,
        max_iter=20,          # có thể tăng lên 30–50 nếu muốn
        verbose=True,
        random_state=42
    )

    print("=== TRAIN MLP ===")
    mlp.fit(X_train_flat, y_train)

    print("=== EVALUATE MLP ===")
    y_pred = mlp.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy (MLP):", acc)

    os.makedirs("models", exist_ok=True)
    dump(mlp, "models/mlp_digit.joblib")
    print("Saved model to models/mlp_digit.joblib")

if __name__ == "__main__":
    main()
