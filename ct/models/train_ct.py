import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


# -------------------------
# Settings
# -------------------------

DATA_PATH = "ct/data"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3


# Use GPU if available, otherwise CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# -------------------------
# Image preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Load dataset
# -------------------------

dataset = datasets.ImageFolder(
    DATA_PATH,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# -------------------------
# Train / validation split
# -------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -------------------------
# Load ResNet18
# -------------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# Freeze the pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace final layer
number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2
)

model = model.to(DEVICE)


# -------------------------
# Loss and optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# -------------------------
# Training
# -------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = 100 * correct / total


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# -------------------------
# Save model
# -------------------------

MODEL_PATH = "ct/models/ct_resnet18.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print("CT model saved successfully!")
print("Saved at:", MODEL_PATH)