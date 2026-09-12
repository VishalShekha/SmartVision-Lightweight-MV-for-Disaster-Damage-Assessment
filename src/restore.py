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
    result = restore_image("D:\\MV Project\\data\\train\\destroyed\\guatemala-volcano_00000019_7f9688c2.png", "test_restored.jpeg")
    print("Restored image saved to test_restored.jpeg — open it and compare to the original.")