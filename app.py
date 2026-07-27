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