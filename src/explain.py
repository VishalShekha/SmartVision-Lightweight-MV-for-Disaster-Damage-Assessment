import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms
import torch.nn as nn

def load_model(model_path="models/damage_classifier.pth"):
    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes

def explain_prediction(model, classes, restored_image_bgr):
    """
    Takes the ALREADY-RESTORED image (numpy array, BGR from OpenCV)
    Returns: predicted label, confidence, and a heatmap overlay image
    """
    rgb_img = cv2.cvtColor(restored_image_bgr, cv2.COLOR_BGR2RGB)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))
    rgb_float = np.float32(rgb_img_resized) / 255.0

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(rgb_img_resized).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        label = classes[pred_idx]
        confidence = probs[pred_idx].item()

    # Grad-CAM heatmap — target the last convolutional block of MobileNetV2
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    heatmap_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return label, confidence, heatmap_overlay