
import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Configuration

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("q2_results", exist_ok=True)

# 2. Load MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

CLASS_NAMES = [str(i) for i in range(10)]

# 3. LeNet-5

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),    # 28 -> 24
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),       # 24 -> 12

            nn.Conv2d(6, 16, kernel_size=5),   # 12 -> 8
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),       # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
# 4. Modern three-layer CNN

class ModernCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14 -> 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 5. Evaluation

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return (
        total_loss / total,
        100.0 * correct / total,
        all_labels,
        all_predictions
    )

# 6. Training

def train_model(model, model_name):
    print(f"\n{'=' * 55}")
    print(f"Training {model_name}")
    print(f"{'=' * 55}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = 100.0 * correct / total

        test_loss, test_accuracy, _, _ = evaluate(
            model, test_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

    # Final test predictions and confusion matrix data
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, labels, predictions = evaluate(
        model, test_loader, criterion
    )

    # Save trained model weights
    torch.save(
        model.state_dict(),
        f"q2_results/{model_name.lower()}_weights.pth"
    )

    # Save history and final metrics
    with open(
        f"q2_results/{model_name.lower()}_history.json", "w"
    ) as f:
        json.dump(history, f, indent=4)

    return model, history, labels, predictions, test_accuracy


# 7. Plot metrics and confusion matrix

def save_results(model_name, history, labels, predictions):
    folder = "q2_results"
    epochs = range(1, EPOCHS + 1)
    name = model_name.lower()

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o",
             label="Training")
    plt.plot(epochs, history["test_accuracy"], marker="o",
             label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name}: Training and Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/{name}_accuracy.png", dpi=200)
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o",
             label="Training")
    plt.plot(epochs, history["test_loss"], marker="o",
             label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{model_name}: Training and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/{name}_loss.png", dpi=200)
    plt.close()

    # Confusion matrix: rows are true classes, columns predicted classes
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label, predicted_label] += 1

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix.numpy(), cmap="Blues")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.title(f"{model_name}: Test Confusion Matrix")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(matrix[i, j].item()),
                     ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{folder}/{name}_confusion_matrix.png", dpi=200)
    plt.close()

# 8. Run both experiments

lenet = LeNet5()
lenet, lenet_history, labels, predictions, lenet_accuracy = train_model(
    lenet, "LeNet5"
)
save_results("LeNet5", lenet_history, labels, predictions)

modern = ModernCNN()
modern, modern_history, labels, predictions, modern_accuracy = train_model(
    modern, "ModernCNN"
)
save_results("ModernCNN", modern_history, labels, predictions)

print("\n" + "=" * 55)
print("FINAL TEST ACCURACY COMPARISON")
print("=" * 55)
print(f"LeNet-5:   {lenet_accuracy:.2f}%")
print(f"Modern CNN: {modern_accuracy:.2f}%")
print("\nAll results saved in q2_results/")
