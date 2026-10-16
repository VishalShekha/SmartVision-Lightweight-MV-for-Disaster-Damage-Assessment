from pathlib import Path
import io
import zipfile
import warnings

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "best_model.pth.zip"

IMAGE_SIZE = 224
CLASS_NAMES = [
    "destroyed",
    "major_damage",
    "minor_damage",
    "no_damage"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGMENTER_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

st.set_page_config(
    page_title="Disaster Building Damage Assessment",
    page_icon="🛰️",
    layout="wide"
)

st.title("Satellite / Drone Disaster Damage Assessment")

st.caption(
    "Automatic building segmentation, paired PRE/POST classification "
    "and Grad-CAM visualization"
)

st.info(
    "Upload matching PRE-disaster and POST-disaster images. "
    "The app automatically identifies candidate building regions, "
    "classifies their damage and displays the results. "
    "No manual polygon input is required."
)


@st.cache_resource(show_spinner="Loading damage classification model...")
def load_damage_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Place best_model.pth.zip beside app_final.py."
        )

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASS_NAMES)
    )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False
        )

    except Exception:
        with zipfile.ZipFile(CHECKPOINT_PATH, "r") as archive:

            candidates = [
                name for name in archive.namelist()
                if name.lower().endswith((".pth", ".pt", ".bin"))
            ]

            if not candidates:
                raise RuntimeError(
                    "The ZIP does not contain a PyTorch checkpoint."
                )

            with archive.open(candidates[0]) as file:
                checkpoint = torch.load(
                    io.BytesIO(file.read()),
                    map_location="cpu",
                    weights_only=False
                )

    if isinstance(checkpoint, dict):

        for key in (
            "state_dict",
            "model_state_dict",
            "model"
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format.")

    checkpoint = {
        key.replace("module.", "", 1)
        if key.startswith("module.")
        else key: value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


@st.cache_resource(show_spinner="Loading building segmentation model...")
def load_segmenter():

    from transformers import (
        SegformerImageProcessor,
        SegformerForSemanticSegmentation
    )

    processor = SegformerImageProcessor.from_pretrained(
        SEGMENTER_ID
    )

    segmenter = SegformerForSemanticSegmentation.from_pretrained(
        SEGMENTER_ID
    )

    segmenter.to(DEVICE)
    segmenter.eval()

    id2label = segmenter.config.id2label

    building_ids = [
        int(index)
        for index, label in id2label.items()
        if "building" in str(label).lower()
    ]

    if not building_ids:
        raise RuntimeError(
            "The segmentation model does not contain a building class."
        )

    return processor, segmenter, building_ids


def read_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")
    return np.asarray(image)


def restore_image(image_rgb):

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    denoised = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        5,
        5,
        7,
        21
    )

    blurred = cv2.GaussianBlur(
        denoised,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.35,
        blurred,
        -0.35,
        0
    )

    return cv2.cvtColor(
        sharpened,
        cv2.COLOR_BGR2RGB
    )


def detect_buildings(
    image_rgb,
    processor,
    segmenter,
    building_ids,
    min_area=250,
    max_buildings=100
):

    image = Image.fromarray(image_rgb)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = segmenter(**inputs)

        logits = F.interpolate(
            outputs.logits,
            size=image_rgb.shape[:2],
            mode="bilinear",
            align_corners=False
        )

        labels = logits.argmax(dim=1)[0].cpu().numpy()

    building_mask = np.isin(
        labels,
        building_ids
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    building_mask = cv2.morphologyEx(
        building_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    building_mask = cv2.morphologyEx(
        building_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    count, component_labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            building_mask,
            connectivity=8
        )
    )

    height, width = building_mask.shape
    regions = []

    for component_id in range(1, count):

        x, y, box_width, box_height, area = stats[component_id]

        if area < min_area:
            continue

        if area > 0.35 * height * width:
            continue

        padding = max(
            2,
            int(0.06 * max(box_width, box_height))
        )

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + box_width + padding)
        y2 = min(height, y + box_height + padding)

        regions.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "area": int(area)
        })

    regions.sort(
        key=lambda region: region["area"],
        reverse=True
    )

    return regions[:max_buildings], building_mask


def get_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def prepare_pair(pre_crop, post_crop):

    pre_image = Image.fromarray(pre_crop).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    post_image = Image.fromarray(post_crop).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    pair = Image.new(
        "RGB",
        (IMAGE_SIZE * 2, IMAGE_SIZE)
    )

    pair.paste(pre_image, (0, 0))
    pair.paste(post_image, (IMAGE_SIZE, 0))

    # MobileNetV2 accepts 224x224 input.
    # This preprocessing must match the training pipeline.
    pair = pair.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    tensor = get_transform()(pair).unsqueeze(0)

    return tensor, pair


def predict_damage(model, pre_crop, post_crop):

    input_tensor, pair_image = prepare_pair(
        pre_crop,
        post_crop
    )

    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))

    return (
        predicted_index,
        probabilities,
        input_tensor,
        pair_image
    )


def generate_gradcam(model, input_tensor, class_index):

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import (
        ClassifierOutputTarget
    )

    target_layer = model.features[-1]

    with GradCAM(
        model=model,
        target_layers=[target_layer]
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[
                ClassifierOutputTarget(class_index)
            ]
        )[0]

    image_tensor = (
        input_tensor[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    display_image = np.clip(
        image_tensor * std + mean,
        0,
        1
    )

    visualization = show_cam_on_image(
        display_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


def draw_buildings(image_rgb, regions, predictions):

    output = image_rgb.copy()

    for index, region in enumerate(regions):

        x1 = region["x1"]
        y1 = region["y1"]
        x2 = region["x2"]
        y2 = region["y2"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 80),
            2
        )

        label = f"Building {index + 1}"

        if index < len(predictions):
            label += ": " + predictions[index]["class"].replace(
                "_", " "
            )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 80),
            1,
            cv2.LINE_AA
        )

    return output


with st.sidebar:

    st.header("Settings")

    min_area = st.slider(
        "Minimum building region area",
        min_value=50,
        max_value=3000,
        value=250,
        step=50
    )

    max_buildings = st.slider(
        "Maximum buildings to classify",
        min_value=1,
        max_value=100,
        value=30
    )

    st.caption(
        "The first run downloads the segmentation model. "
        "A GPU is used when available."
    )


col_pre, col_post = st.columns(2)

with col_pre:

    pre_upload = st.file_uploader(
        "PRE-disaster image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="pre_image"
    )

with col_post:

    post_upload = st.file_uploader(
        "POST-disaster image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="post_image"
    )


if pre_upload is not None and post_upload is not None:

    try:

        pre_original = read_image(pre_upload)
        post_original = read_image(post_upload)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                pre_original,
                caption="PRE-disaster image",
                use_container_width=True
            )

        with col2:
            st.image(
                post_original,
                caption="POST-disaster image",
                use_container_width=True
            )

        if pre_original.shape[:2] != post_original.shape[:2]:

            st.warning(
                "Image dimensions differ. PRE will be resized to the "
                "POST dimensions. Resizing does not guarantee geographic "
                "alignment, so use images covering the same area."
            )

            pre_original = cv2.resize(
                pre_original,
                (
                    post_original.shape[1],
                    post_original.shape[0]
                ),
                interpolation=cv2.INTER_LINEAR
            )

        if st.button(
            "Detect buildings and predict damage",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Preparing images and loading models..."):

                pre_image = restore_image(pre_original)
                post_image = restore_image(post_original)

                damage_model = load_damage_model()

                processor, segmenter, building_ids = load_segmenter()

            with st.spinner("Detecting candidate buildings..."):

                regions, building_mask = detect_buildings(
                    post_image,
                    processor,
                    segmenter,
                    building_ids,
                    min_area,
                    max_buildings
                )

            if not regions:

                st.error(
                    "No candidate buildings were detected. Try lowering "
                    "the minimum region area or using a clearer image. "
                    "The general-purpose segmentation model may not work "
                    "well on aerial imagery."
                )

            else:

                st.success(
                    f"Detected {len(regions)} candidate building regions."
                )

                st.warning(
                    "This segmentation model is pretrained on general "
                    "scene images, not specifically on disaster aerial "
                    "imagery. Some regions may not be buildings. Review "
                    "the results before relying on them."
                )

                predictions = []

                progress = st.progress(0)

                with st.spinner("Classifying PRE/POST building pairs..."):

                    for index, region in enumerate(regions):

                        x1 = region["x1"]
                        y1 = region["y1"]
                        x2 = region["x2"]
                        y2 = region["y2"]

                        pre_crop = pre_image[y1:y2, x1:x2]
                        post_crop = post_image[y1:y2, x1:x2]

                        if pre_crop.size == 0 or post_crop.size == 0:
                            continue

                        class_index, probabilities, input_tensor, pair_image = (
                            predict_damage(
                                damage_model,
                                pre_crop,
                                post_crop
                            )
                        )

                        predictions.append({
                            "building": index + 1,
                            "class": CLASS_NAMES[class_index],
                            "confidence": float(
                                probabilities[class_index]
                            ),
                            "probabilities": probabilities,
                            "region": region,
                            "pre_crop": pre_crop,
                            "post_crop": post_crop,
                            "input_tensor": input_tensor,
                            "pair_image": pair_image,
                            "class_index": class_index
                        })

                        progress.progress(
                            (index + 1) / len(regions)
                        )

                if not predictions:

                    st.error("No building crops could be classified.")

                else:

                    annotated_image = draw_buildings(
                        post_image,
                        [p["region"] for p in predictions],
                        predictions
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            annotated_image,
                            caption="Detected regions and damage predictions",
                            use_container_width=True
                        )

                    with col2:
                        st.image(
                            building_mask * 255,
                            caption="Building segmentation mask",
                            clamp=True,
                            use_container_width=True
                        )

                    st.subheader("Damage assessment results")

                    rows = []

                    for prediction in predictions:

                        region = prediction["region"]

                        rows.append({
                            "Building": prediction["building"],
                            "Predicted damage": prediction["class"].replace(
                                "_", " "
                            ).title(),
                            "Confidence": round(
                                prediction["confidence"] * 100,
                                2
                            ),
                            "Region area": region["area"],
                            "Bounding box": (
                                f"{region['x1']},{region['y1']},"
                                f"{region['x2']},{region['y2']}"
                            )
                        })

                    results_df = pd.DataFrame(rows)

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("Damage class counts")

                    counts = pd.Series(
                        [
                            p["class"].replace("_", " ").title()
                            for p in predictions
                        ]
                    ).value_counts()

                    st.bar_chart(counts)

                    selected_index = st.selectbox(
                        "Inspect a building",
                        options=list(range(len(predictions))),
                        format_func=lambda i: (
                            f"Building {predictions[i]['building']} — "
                            f"{predictions[i]['class'].replace('_', ' ').title()}"
                        )
                    )

                    selected = predictions[selected_index]

                    st.subheader(
                        f"Building {selected['building']} details"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            selected["pre_crop"],
                            caption="PRE building crop",
                            use_container_width=True
                        )

                    with col2:
                        st.image(
                            selected["post_crop"],
                            caption="POST building crop",
                            use_container_width=True
                        )

                    st.write("Class probabilities")

                    probability_df = pd.DataFrame({
                        "Class": [
                            name.replace("_", " ").title()
                            for name in CLASS_NAMES
                        ],
                        "Probability": selected["probabilities"]
                    }).set_index("Class")

                    st.bar_chart(probability_df)

                    st.subheader("Grad-CAM explanation")

                    try:

                        with st.spinner("Generating Grad-CAM..."):

                            cam_image = generate_gradcam(
                                damage_model,
                                selected["input_tensor"].to(DEVICE),
                                selected["class_index"]
                            )

                        st.image(
                            cam_image,
                            caption=(
                                "Highlighted areas influenced the classifier's "
                                "prediction. They are not proof of physical damage."
                            ),
                            use_container_width=True
                        )

                    except Exception as exc:

                        st.warning(
                            f"Grad-CAM could not be generated: {exc}"
                        )

                    csv_data = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download results as CSV",
                        data=csv_data,
                        file_name="building_damage_results.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "Confidence scores are model outputs, not necessarily "
                        "calibrated probabilities. Automatic building regions "
                        "may contain false detections."
                    )

    except Exception as exc:

        st.error(f"{type(exc).__name__}: {exc}")
        st.exception(exc)

else:

    st.caption(
        "Upload both matching PRE-disaster and POST-disaster images to begin."
    )