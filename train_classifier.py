import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---- Settings ----
DATA_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # will use CPU if no GPU — that's fine

# ---- Image preprocessing for training ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # standard for pretrained models
])

train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
val_data = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

num_classes = len(train_data.classes)
print("Classes found:", train_data.classes)

# ---- Load lightweight pretrained model ----
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Freeze the "vision" part — we only train the final classifier layer.
# This is why training is fast and doesn't need a huge dataset.
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final layer to output OUR classes instead of the original 1000
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# ---- Training loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_data)

    # ---- Validation ----
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(val_data)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

# ---- Save the trained model ----
torch.save({
    "model_state": model.state_dict(),
    "classes": train_data.classes
}, "models/damage_classifier.pth")

print("Saved model to models/damage_classifier.pth")