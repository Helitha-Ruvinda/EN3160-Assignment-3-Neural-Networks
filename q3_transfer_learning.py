
import copy
import json
import os
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("q3_results", exist_ok=True)
os.makedirs("data", exist_ok=True)


# --------------------------------------------------
# 2. Download Hymenoptera dataset if necessary
# --------------------------------------------------
data_dir = os.path.join("data", "hymenoptera_data")
zip_path = os.path.join("data", "hymenoptera_data.zip")

if not os.path.isdir(os.path.join(data_dir, "train")):
    if not os.path.exists(zip_path):
        print("Downloading Hymenoptera dataset...")
        url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall("data")

print("Dataset ready.")


# --------------------------------------------------
# 3. Image transformations and data loaders
# --------------------------------------------------
weights = models.ResNet18_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

image_datasets = {
    split: datasets.ImageFolder(
        os.path.join(data_dir, split),
        data_transforms[split]
    )
    for split in ["train", "val"]
}

dataloaders = {
    split: DataLoader(
        image_datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    for split in ["train", "val"]
}

class_names = image_datasets["train"].classes
print("Classes:", class_names)
print("Training images:", len(image_datasets["train"]))
print("Validation images:", len(image_datasets["val"]))


# --------------------------------------------------
# 4. Build pretrained ResNet-18
# --------------------------------------------------
def create_model(freeze_features=False):
    model = models.resnet18(weights=weights)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace ImageNet's 1000-class output with 2 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))

    return model.to(device)


# --------------------------------------------------
# 5. Evaluate a model
# --------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return (
        total_loss / total,
        100.0 * correct / total,
        all_labels,
        all_predictions
    )


# --------------------------------------------------
# 6. Train either fine-tuned or frozen model
# --------------------------------------------------
def run_experiment(experiment_name, freeze_features):
    print("\n" + "=" * 55)
    print(experiment_name)
    print("=" * 55)

    model = create_model(freeze_features)
    criterion = nn.CrossEntropyLoss()

    # Train only the classifier when features are frozen
    trainable_params = filter(
        lambda p: p.requires_grad, model.parameters()
    )

    optimizer = optim.SGD(
        trainable_params,
        lr=LEARNING_RATE,
        momentum=0.9
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = 100.0 * correct / total

        val_loss, val_accuracy, _, _ = evaluate(
            model, dataloaders["val"], criterion
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2f}% | "
            f"Time: {time.time() - start_time:.1f}s"
        )

    # Evaluate the best validation checkpoint
    model.load_state_dict(best_weights)
    val_loss, val_accuracy, labels, predictions = evaluate(
        model, dataloaders["val"], criterion
    )

    name = experiment_name.lower().replace(" ", "_")

    torch.save(model.state_dict(), f"q3_results/{name}_weights.pth")

    with open(f"q3_results/{name}_history.json", "w") as f:
        json.dump(history, f, indent=4)

    return history, labels, predictions, val_accuracy


# --------------------------------------------------
# 7. Save plots and confusion matrix
# --------------------------------------------------
def save_results(name, history, labels, predictions):
    folder = "q3_results"
    epochs = range(1, EPOCHS + 1)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o",
             label="Training")
    plt.plot(epochs, history["val_accuracy"], marker="o",
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{name}: Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/{name}_accuracy.png", dpi=200)
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o",
             label="Training")
    plt.plot(epochs, history["val_loss"], marker="o",
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{name}: Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/{name}_loss.png", dpi=200)
    plt.close()

    # Confusion matrix
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label, predicted_label] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix.numpy(), cmap="Blues")
    plt.colorbar()
    plt.xticks(range(2), class_names)
    plt.yticks(range(2), class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"{name}: Validation Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j].item()),
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{folder}/{name}_confusion_matrix.png", dpi=200)
    plt.close()


# --------------------------------------------------
# 8. Run both experiments
# --------------------------------------------------
results = {}

for name, freeze in [
    ("Fine_Tuning", False),
    ("Frozen_Features", True)
]:
    history, labels, predictions, accuracy = run_experiment(
        name, freeze
    )

    save_results(name, history, labels, predictions)
    results[name] = accuracy

print("\nFINAL VALIDATION ACCURACY")
print(f"Fine-tuning:       {results['Fine_Tuning']:.2f}%")
print(f"Frozen features:   {results['Frozen_Features']:.2f}%")
print("\nAll Q3 results saved in q3_results/")
