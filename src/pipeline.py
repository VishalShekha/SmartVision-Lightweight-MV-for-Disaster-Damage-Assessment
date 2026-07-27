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
    result = run_pipeline("data/train/damage/-93.6141_30.754263.jpeg")
    print(f"Prediction: {result['label']} ({result['confidence']:.1%} confidence)")
    cv2.imwrite("output_heatmap.jpeg", cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR))
    print("Saved heatmap to output_heatmap.jpeg")