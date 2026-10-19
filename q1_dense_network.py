
import time
import torch
from torchvision import datasets, transforms


# 1. Configuration

BATCH_SIZE = 50
EPOCHS = 10
LEARNING_RATE = 0.01
LR_DECAY = 0.9
HIDDEN_SIZE = 100
NUM_CLASSES = 10

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Added GPU support
print(f"Using device: {device}")


# 2. Load CIFAR-10

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Convert images to flattened vectors: 32 * 32 * 3 = 3072
X_train = train_dataset.data
X_train = torch.tensor(X_train, dtype=torch.float32)
X_train = X_train.reshape(-1, 3072) / 255.0
y_train = torch.tensor(train_dataset.targets, dtype=torch.long)

X_test = torch.tensor(test_dataset.data, dtype=torch.float32)
X_test = X_test.reshape(-1, 3072) / 255.0
y_test = torch.tensor(test_dataset.targets, dtype=torch.long)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# 3. Initialize parameters

input_size = 3072

# Xavier-style initialization
W1 = torch.randn(input_size, HIDDEN_SIZE, device=device) * (
    (2.0 / (input_size + HIDDEN_SIZE)) ** 0.5
)
b1 = torch.zeros(HIDDEN_SIZE, device=device)

W2 = torch.randn(HIDDEN_SIZE, NUM_CLASSES, device=device) * (
    (2.0 / (HIDDEN_SIZE + NUM_CLASSES)) ** 0.5
)
b2 = torch.zeros(NUM_CLASSES, device=device)


# 4. Forward pass and loss

def forward(X):
    Z1 = X @ W1 + b1
    H = torch.sigmoid(Z1)
    logits = H @ W2 + b2
    return H, logits


def cross_entropy_loss(logits, labels):
    # Numerically stable softmax cross-entropy
    log_probs = logits - torch.logsumexp(
        logits, dim=1, keepdim=True
    )
    return -log_probs[
        torch.arange(labels.shape[0], device=device), labels
    ].mean()


# 5. Evaluate accuracy and loss

@torch.no_grad()
def evaluate(X, y):
    total_loss = 0.0
    correct = 0
    total = X.shape[0]

    for start in range(0, total, 500):
        xb = X[start:start + 500]
        yb = y[start:start + 500]

        _, logits = forward(xb)
        loss = cross_entropy_loss(logits, yb)

        total_loss += loss.item() * yb.shape[0]
        correct += (logits.argmax(dim=1) == yb).sum().item()

    return total_loss / total, 100.0 * correct / total

# 6. Training with manual backpropagation

history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

learning_rate = LEARNING_RATE

for epoch in range(EPOCHS):
    start_time = time.time()

    # Shuffle training data
    indices = torch.randperm(X_train.shape[0], device=device)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0.0
    num_batches = 0

    for start in range(0, X_train.shape[0], BATCH_SIZE):
        Xb = X_shuffled[start:start + BATCH_SIZE]
        yb = y_shuffled[start:start + BATCH_SIZE]
        batch_n = Xb.shape[0]

        # Forward pass
        H, logits = forward(Xb)
        loss = cross_entropy_loss(logits, yb)

        # Gradient of softmax + cross-entropy
        probs = torch.softmax(logits, dim=1)
        probs[
            torch.arange(batch_n, device=device), yb
        ] -= 1.0
        dZ2 = probs / batch_n

        # Gradients for output layer
        dW2 = H.T @ dZ2
        db2 = dZ2.sum(dim=0)

        # Backpropagate through sigmoid hidden layer
        dH = dZ2 @ W2.T
        dZ1 = dH * H * (1.0 - H)

        # Gradients for hidden layer
        dW1 = Xb.T @ dZ1
        db1_grad = dZ1.sum(dim=0)

        # Manual gradient-descent updates
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1_grad

        epoch_loss += loss.item()
        num_batches += 1

    # Evaluate after each epoch
    train_loss, train_acc = evaluate(X_train, y_train)
    test_loss, test_acc = evaluate(X_test, y_test)

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_accuracy"].append(test_acc)

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.2f}% | "
        f"Test accuracy: {test_acc:.2f}% | "
        f"Time: {elapsed:.1f}s"
    )

    learning_rate *= LR_DECAY

# 7. Final results

print("\nFinal results")
print(f"Training accuracy: {history['train_accuracy'][-1]:.2f}%")
print(f"Test accuracy:     {history['test_accuracy'][-1]:.2f}%")

# 8. Save training history and generate plots

import json
import os
import matplotlib.pyplot as plt

os.makedirs("q1_results", exist_ok=True)

# Save per-epoch metrics
with open("q1_results/history.json", "w") as f:
    json.dump(history, f, indent=4)

epochs = range(1, EPOCHS + 1)

# Plot accuracy
plt.figure()
plt.plot(epochs, history["train_accuracy"], marker="o", label="Training")
plt.plot(epochs, history["test_accuracy"], marker="o", label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Q1: Training and Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("q1_results/accuracy.png", dpi=200)
plt.close()

# Plot loss
plt.figure()
plt.plot(epochs, history["train_loss"], marker="o", label="Training")
plt.plot(epochs, history["test_loss"], marker="o", label="Test")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Q1: Training and Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("q1_results/loss.png", dpi=200)
plt.close()

# Generate test-set confusion matrix
@torch.no_grad()
def get_predictions(X):
    predictions = []
    for start in range(0, len(X), 500):
        _, logits = forward(X[start:start + 500])
        predictions.append(logits.argmax(dim=1))
    return torch.cat(predictions)

predictions = get_predictions(X_test)
confusion = torch.bincount(
    y_test * NUM_CLASSES + predictions,
    minlength=NUM_CLASSES * NUM_CLASSES
).reshape(NUM_CLASSES, NUM_CLASSES).cpu().numpy()

plt.figure(figsize=(9, 8))
plt.imshow(confusion, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Q1: CIFAR-10 Test Confusion Matrix")
plt.xticks(range(NUM_CLASSES))
plt.yticks(range(NUM_CLASSES))

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, str(confusion[i, j]),
                 ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig("q1_results/confusion_matrix.png", dpi=200)
plt.close()

print("\nSaved results to the q1_results/ folder.")
