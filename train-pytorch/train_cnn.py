"""
train_cnn.py - Train CNN (Convolutional Neural Network) với PyTorch

Hỗ trợ 2 dataset:
    - dataset-main: 40k train, 10k test (dataset gốc)
    - mnist_png: 60k train, 10k test (MNIST chuẩn)

Sử dụng:
    python train_cnn.py                    # Train với dataset-main
    python train_cnn.py mnist_png          # Train với MNIST
    python train_cnn.py mnist_png 20       # Train với MNIST, 20 epochs

Architecture:
    Input (1, 28, 28)
    -> Conv2d(32, 3x3) + ReLU + MaxPool(2x2)  -> (32, 14, 14)
    -> Conv2d(64, 3x3) + ReLU + MaxPool(2x2)  -> (64, 7, 7)  
    -> Conv2d(64, 3x3) + ReLU + MaxPool(2x2)  -> (64, 3, 3)
    -> Flatten -> Dense(128) + ReLU -> Dense(10)

Data Augmentation:
    - Random rotation ±15°
    - Random shift ±2 pixels
    - Random scale 0.9-1.1x
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

from dataset_main import load_dataset_from_folder


class DigitCNN(nn.Module):
    """CNN 3 lớp convolution cho nhận diện chữ số 28x28.
    
    Tổng parameters: ~130k (rất nhẹ)
    """
    
    def __init__(self):
        super(DigitCNN, self).__init__()
        
        # === Convolutional Layers ===
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)  # 7x7 -> 7x7
        
        # MaxPool giảm kích thước 1/2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout: tắt ngẫu nhiên neurons để tránh overfitting
        self.dropout1 = nn.Dropout(0.25)  # Sau conv
        self.dropout2 = nn.Dropout(0.5)   # Trước output
        
        # === Fully Connected Layers ===
        # 64 channels x 3x3 = 576 -> 128 -> 10
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass: x shape (batch, 1, 28, 28)"""
        # Block 1: Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        
        # Block 2
        x = self.pool(self.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        
        # Block 3
        x = self.pool(self.relu(self.conv3(x)))  # -> (batch, 64, 3, 3)
        x = self.dropout1(x)
        
        # Flatten và FC
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout2(self.relu(self.fc1(x)))
        x = self.fc2(x)  # Logits (chưa softmax)
        
        return x


def augment_batch(images):
    """Data augmentation on-the-fly cho một batch.
    
    Các biến đổi:
        - Rotation: xoay ±15° (50% chance)
        - Translation: dịch ±2px (50% chance)  
        - Scale: zoom 0.9-1.1x (30% chance)
    
    Args:
        images: tensor (batch, 1, 28, 28)
    
    Returns:
        augmented tensor cùng shape
    """
    augmented = images.clone()
    
    for i in range(images.size(0)):
        img = augmented[i, 0].numpy()
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, M, (28, 28), borderValue=1.0)
        
        # Random shift
        if np.random.random() > 0.5:
            tx, ty = np.random.randint(-2, 3, 2)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (28, 28), borderValue=1.0)
        
        # Random scale
        if np.random.random() > 0.7:
            scale = np.random.uniform(0.9, 1.1)
            new_size = int(28 * scale)
            img_resized = cv2.resize((img * 255).astype(np.uint8),
                                     (new_size, new_size))
            
            # Pad hoặc crop về 28x28
            result = np.full((28, 28), 255, dtype=np.uint8)
            if new_size > 28:
                start = (new_size - 28) // 2
                result = img_resized[start:start+28, start:start+28]
            else:
                start = (28 - new_size) // 2
                result[start:start+new_size, start:start+new_size] = img_resized
            img = result.astype(np.float32) / 255.0
        
        augmented[i, 0] = torch.from_numpy(img)
    
    return augmented


def main():
    # === Parse arguments ===
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset-main"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("=" * 60)
    print("       TRAIN CNN MODEL - PyTorch")
    print("=" * 60)
    print(f"\nDataset: {dataset}")
    print(f"Epochs:  {epochs}")
    
    # Device: GPU nếu có, không thì CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    
    # === Load dataset ===
    print("\n[1/4] Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset_from_folder(dataset)
    
    # Reshape: (N, 28, 28) -> (N, 1, 28, 28) cho CNN
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    # Convert sang PyTorch tensors
    train_dataset = TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test),
                                 torch.from_numpy(y_test))
    
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # === Create model ===
    print("\n[2/4] Creating model...")
    model = DigitCNN().to(device)
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss: CrossEntropy = Softmax + NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam với learning rate decay
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # === Training ===
    print("\n[3/4] Training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=80)
        for images, labels in pbar:
            # Augmentation
            images = augment_batch(images)
            images, labels = images.to(device), labels.to(device)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                           acc=f"{100*correct/total:.1f}%")
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = sum(
            (model(img.to(device)).argmax(1) == lab.to(device)).sum().item()
            for img, lab in test_loader
        )
        print(f"      Val acc: {100*val_correct/len(X_test):.2f}%")
    
    print(f"\n      Training time: {time.time() - start_time:.1f}s")
    
    # === Final evaluation ===
    print("\n[4/4] Final evaluation...")
    model.eval()
    
    per_class = [[0, 0] for _ in range(10)]  # [correct, total]
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(1).cpu()
            
            for pred, label in zip(preds, labels):
                per_class[label.item()][1] += 1
                if pred == label:
                    per_class[label.item()][0] += 1
    
    total_correct = sum(c[0] for c in per_class)
    total_samples = sum(c[1] for c in per_class)
    
    print("\n" + "=" * 60)
    print(f"  Test Accuracy: {100*total_correct/total_samples:.2f}%")
    print("=" * 60)
    
    print("\n  Per-digit:")
    for i, (correct, total) in enumerate(per_class):
        bar = "█" * int(correct / total * 20)
        print(f"    {i}: {bar:<20} {correct:>4}/{total} ({100*correct/total:.1f}%)")
    
    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_digit.pth")
    print(f"\n  Saved: models/cnn_digit.pth")


if __name__ == "__main__":
    main()
