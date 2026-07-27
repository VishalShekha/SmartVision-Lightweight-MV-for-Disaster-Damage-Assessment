# Building a Disaster Damage Assessment AI — Complete Beginner's Guide

This guide walks you through building the full project from zero, assuming you're new to coding AI projects. We'll build it in stages, and each stage will actually run and produce a visible result, so you always know it's working before moving to the next step.

---

## What You're Actually Building

A pipeline with 3 stages that chain together:

```
Blurry/low-res drone photo
        ↓
[Stage 1] Image Restoration — cleans and sharpens the photo
        ↓
[Stage 2] Lightweight Classifier — labels damage level (No damage / Minor / Major / Destroyed)
        ↓
[Stage 3] Explainability (Grad-CAM) — draws a heatmap showing WHY it chose that label
        ↓
Result shown in a simple web app (runs on your laptop, no internet needed after setup)
```

Each of the "3 novelties" in your project brief maps directly to one stage above. That's the whole project — nothing more mysterious than that.

---

## Part 0 — Tools You Need Installed

You only need to do this once.

1. **Python 3.10 or 3.11** — Download from [python.org](https://www.python.org/downloads/). During install on Windows, check the box "Add Python to PATH".
2. **VS Code** (or any code editor) — [code.visualstudio.com](https://code.visualstudio.com/). Install the "Python" extension inside it.
3. **Git** (optional but useful) — [git-scm.com](https://git-scm.com/)

Check installation worked by opening a terminal (Command Prompt / Terminal / VS Code's terminal) and typing:

```bash
python --version
pip --version
```

You should see version numbers, not an error.

---

## Part 1 — Project Setup

Create a folder and a virtual environment (a clean, isolated Python workspace so packages don't conflict with other projects).

```bash
mkdir disaster-damage-ai
cd disaster-damage-ai
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll know it worked because your terminal line now starts with `(venv)`.

Now install the core libraries:

```bash
pip install opencv-python numpy matplotlib pillow scikit-image gradio grad-cam
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

What each one is for:
| Library | Purpose |
|---|---|
| `torch`, `torchvision` | The AI/deep learning engine and pretrained models |
| `opencv-python` | Classic image processing (denoising, sharpening) |
| `numpy` | Number crunching, arrays |
| `matplotlib` | Plotting/visualizing images |
| `pillow` | Loading/saving images |
| `scikit-image` | More image quality tools |
| `gradio` | Turns your pipeline into a simple web app UI |
| `grad-cam` | Ready-made explainability (heatmap) tool |

Create this folder structure:

```
disaster-damage-ai/
│
├── data/              ← training images go here
├── models/            ← saved AI models go here
├── src/
│   ├── restore.py     ← Stage 1: image cleanup
│   ├── classify.py    ← Stage 2: damage classifier
│   ├── explain.py      ← Stage 3: heatmap explainability
│   └── pipeline.py    ← glues all 3 stages together
├── app.py             ← the web app
└── train_classifier.py
```

You can create these empty files now (`type nul > src\restore.py` on Windows or `touch src/restore.py` on Mac/Linux) — we'll fill them in below.

---

## Part 2 — Get a Dataset

You need labeled examples of "damaged" vs "not damaged" buildings to train the classifier.

**Best option for beginners:** Use the **xBD dataset** (built exactly for this — pre/post-disaster satellite imagery with damage labels across 4 classes: no-damage, minor-damage, major-damage, destroyed). It's the standard academic dataset for this exact task.

- Search "xBD dataset xView2" and register at their site (free, for research use) to download.
- It's large, so for a first working prototype, just download **one disaster region's subset** (a few hundred MB), not the whole thing.

**Simpler alternative if xBD feels heavy for a first pass:** Use a smaller Kaggle dataset like "Satellite Images of Hurricane Damage" (binary: damage / no damage). This lets you get the whole pipeline working end-to-end in one afternoon, and you can upgrade to xBD's 4-class version afterward.

Put your images into folders like this (this format is called "ImageFolder" style — PyTorch reads it automatically):

```
data/
├── train/
│   ├── no_damage/
│   ├── minor_damage/
│   ├── major_damage/
│   └── destroyed/
└── val/
    ├── no_damage/
    ├── minor_damage/
    ├── major_damage/
    └── destroyed/
```

Aim for at least ~100–200 images per folder to start (more is better, but this is enough to prove the pipeline works).

---

## Part 3 — Novelty #1: Smart Image Rescue (Restoration)

This step takes a blurry/noisy/pixelated photo and cleans it up **before** the AI ever sees it. We'll use classical, fast techniques (no heavy AI needed here — this keeps it lightweight, which also helps Novelty #2).

Create `src/restore.py`:

```python
import cv2
import numpy as np

def restore_image(image_path, save_path=None):
    """
    Cleans up a degraded image:
    1. Denoise (removes graininess/compression artifacts)
    2. Sharpen (recovers edge detail lost to blur)
    3. Upscale if resolution is very low (helps the AI 'see' small damage details)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1. Denoise — removes speckle/compression noise from bad transmission
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                                templateWindowSize=7, searchWindowSize=21)

    # 2. Sharpen — unsharp mask technique to recover edges
    gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # 3. Upscale small images so the classifier has enough detail to work with
    h, w = sharpened.shape[:2]
    if max(h, w) < 512:
        scale = 512 / max(h, w)
        sharpened = cv2.resize(sharpened, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)

    if save_path:
        cv2.imwrite(save_path, sharpened)

    return sharpened


if __name__ == "__main__":
    # quick manual test
    result = restore_image("data/train/destroyed/example1.jpg", "test_restored.jpg")
    print("Restored image saved to test_restored.jpg — open it and compare to the original.")
```

**Test it:** Run `python src/restore.py` (adjust the file path to a real image you have) and open `test_restored.jpg` next to the original. You should visibly see less graininess and crisper edges.

> **Later upgrade (optional, once the basic version works):** Replace this with a small learned model called a lightweight super-resolution network (e.g., FSRCNN or a tiny autoencoder trained on pairs of "clean image" / "artificially blurred version of it"). This is what makes it a genuine "AI-based restoration" novelty instead of just classic filters, and is a great addition if you want extra project credit. Classical filtering above is the right starting point though — get the full pipeline working first.

---

## Part 4 — Novelty #2: Featherlight AI for Local Devices

This is the classifier that labels damage severity. We use **transfer learning**: start from a small pretrained model (already knows how to "see" general image features) and just retrain its final layer on your building-damage data. This is fast, needs little data, and stays small enough to run on a laptop CPU — which is the whole point of "edge-optimized."

We'll use **MobileNetV2** — a model literally designed by Google to run on phones. That's your "lightweight" model.

Create `train_classifier.py`:

```python
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
```

**Run it:**

```bash
python train_classifier.py
```

On a laptop CPU with ~100–200 images per class, this should take somewhere between a few minutes and half an hour, depending on your machine. You'll see accuracy printed each epoch — you want validation accuracy climbing, ideally into the 70–90%+ range depending on how clean your dataset is.

> Why this is "edge-optimized": MobileNetV2 has ~3.5 million parameters vs. ~25 million+ for something like ResNet50. It was specifically designed to run in real time on mobile chips, so a laptop CPU handles it comfortably — no GPU or cloud server required.

---

## Part 5 — Novelty #3: Visual Transparency (Explainability)

This produces the heatmap showing *where* the model looked (e.g., a collapsed roofline) to reach its decision. We'll use **Grad-CAM**, the standard technique for this.

Create `src/explain.py`:

```python
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
```

This gives you three things back: the predicted damage class, how confident the model is, and an image with a red/yellow heatmap glow over the exact pixels that drove the decision.

---

## Part 6 — Combine Everything Into One Pipeline

Create `src/pipeline.py`:

```python
from src.restore import restore_image
from src.explain import load_model, explain_prediction
import cv2

# Load model once (reused across images)
_model, _classes = load_model()

def run_pipeline(image_path):
    # Stage 1: restore
    restored = restore_image(image_path)

    # Stage 2 + 3: classify + explain
    label, confidence, heatmap = explain_prediction(_model, _classes, restored)

    return {
        "restored_image": restored,
        "label": label,
        "confidence": confidence,
        "heatmap": heatmap
    }

if __name__ == "__main__":
    result = run_pipeline("data/val/destroyed/example.jpg")
    print(f"Prediction: {result['label']} ({result['confidence']:.1%} confidence)")
    cv2.imwrite("output_heatmap.jpg", cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR))
    print("Saved heatmap to output_heatmap.jpg")
```

**Test it:** `python -m src.pipeline` (run from the project root folder so the imports work). Check the printed prediction and open `output_heatmap.jpg`.

---

## Part 7 — Build the Web App (So It's Actually Usable)

This is what you'll demo — a person uploads a photo, and the app shows the cleaned image, the label, and the heatmap.

Create `app.py` in your project root:

```python
import gradio as gr
import cv2
import numpy as np
from src.restore import restore_image
from src.explain import load_model, explain_prediction

model, classes = load_model()

def process(image):
    # Gradio gives us a numpy RGB image; convert to BGR for OpenCV functions
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("temp_input.jpg", bgr)

    restored = restore_image("temp_input.jpg")
    label, confidence, heatmap = explain_prediction(model, classes, restored)

    restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    result_text = f"Damage Level: {label.upper()}  |  Confidence: {confidence:.1%}"

    return restored_rgb, heatmap, result_text

demo = gr.Interface(
    fn=process,
    inputs=gr.Image(label="Upload drone/satellite photo (can be blurry/low quality)"),
    outputs=[
        gr.Image(label="Restored Image (Stage 1)"),
        gr.Image(label="AI Explanation Heatmap (Stage 3)"),
        gr.Text(label="Damage Assessment (Stage 2)")
    ],
    title="Disaster Damage Assessment AI",
    description="Upload a degraded building photo. The system restores it, classifies damage severity, and shows what it based its decision on — all running locally."
)

if __name__ == "__main__":
    demo.launch()
```

**Run it:**

```bash
python app.py
```

A local link (like `http://127.0.0.1:7860`) will print in your terminal — open it in your browser. Upload a photo and you'll see all three stages' outputs side by side. This works fully offline once the model is trained (Gradio just uses your browser as the display — no internet needed to actually run the AI).

---

## Part 8 — Test It Properly

1. Deliberately degrade a few clean test images (blur and shrink them) to prove the restoration step matters:

```python
import cv2
img = cv2.imread("clean_test_image.jpg")
blurred = cv2.GaussianBlur(img, (15, 15), 0)
small = cv2.resize(blurred, (64, 64))  # simulate very low-res drone capture
cv2.imwrite("degraded_test.jpg", small)
```

2. Run `degraded_test.jpg` through your app both **with** and **without** the restoration step (temporarily bypass `restore_image` and feed the raw image straight to the classifier) and compare accuracy/confidence. This comparison is great evidence for your report — it directly proves Novelty #1 is doing real work.

3. Record how long each stage takes (`import time`, wrap each function call) to prove Novelty #2's "runs fast on a laptop" claim — e.g., "full pipeline: 0.8 seconds on a standard laptop CPU, no GPU."

---

## Part 9 — Package It for "Edge/Offline" Use (Optional Polish)

To strengthen your "runs on ordinary laptop, no internet" claim:

- **Export to ONNX** (a portable, optimized model format) so it can run without PyTorch installed at all, and runs faster on CPU:
```python
import torch
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "models/damage_classifier.onnx")
```
- Mention in your report that the whole app was tested with Wi-Fi disabled, to concretely demonstrate offline capability.

---

## Part 10 — What to Put in Your Report/Presentation

For each novelty, structure it as: **Problem → Your Solution → Evidence**

1. **Smart Image Rescue:** Show side-by-side: degraded photo → restored photo → classification results for both (accuracy/confidence difference).
2. **Featherlight AI:** State model size (MobileNetV2 ≈ 14MB), and measured inference time on your laptop with no GPU.
3. **Explainability:** Show 3–4 heatmap examples across different damage levels, and briefly describe what visual feature each one highlights (e.g., "the model focused on the collapsed roofline" for a destroyed building).

---

## Quick Troubleshooting

| Problem | Likely Fix |
|---|---|
| `ModuleNotFoundError` | You forgot to activate `venv`, or forgot `pip install` for that package |
| Training accuracy stuck near random guessing | Too few images per class, or images mislabeled/misplaced in wrong folders |
| `CUDA out of memory` | You don't need a GPU for this — set `DEVICE = "cpu"` manually in the training script |
| Gradio app image looks color-shifted (blue-ish) | Missing an RGB↔BGR conversion — OpenCV uses BGR, everything else (PIL, Gradio) uses RGB |
| Heatmap looks blank/uniform | Wrong `target_layers` in Grad-CAM, or model wasn't actually trained (check `models/damage_classifier.pth` exists and isn't from a failed run) |

---

## Suggested Order to Actually Do This (Time Estimate)

1. Part 0–1 (setup): 30 min
2. Part 2 (dataset): 1–2 hours (mostly download time)
3. Part 3 (restoration): 30 min
4. Part 4 (train classifier): 1–3 hours including training time
5. Part 5–6 (explainability + pipeline): 1 hour
6. Part 7 (web app): 30 min
7. Part 8–10 (testing + polish + report): 2–3 hours

Total: realistically a solid weekend project, or 2–3 evenings.

You now have a real, working, demoable AI system — not just a concept. Good luck!
