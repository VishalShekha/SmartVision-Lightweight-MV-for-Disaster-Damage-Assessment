import torch
from torchvision import models
import torch.nn as nn

checkpoint_path = "best_model.pth.zip"

state = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 4)
model.load_state_dict(state)
model.eval()

print("Checkpoint loaded successfully")
print("Number of output classes:", model.classifier[1].out_features)