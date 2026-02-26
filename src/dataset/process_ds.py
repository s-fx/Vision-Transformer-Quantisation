import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageChops


INPUT_ROOT = Path("/home/s-fx/fun/datasets/retinopathy-full-ds")
OUTPUT_ROOT = Path("/home/s-fx/fun/datasets/retinopathy-full-ds-cleaned")

SPLITS = ["train", "val", "test", "eval", "single_example"]
CLASSES = ["0", "1", "2", "3", "4"]

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def apply_clahe(img_path, img_size=256):
    # Read and resize the image
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
    #image = cv2.resize(image, (img_size, img_size))

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the LAB channels and convert back to RGB
    merged_lab = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # Save the preprocessed image
    #save_path = os.path.join(save_dir, os.path.basename(image_path))
    #cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    return final_image


def clean_fundus_image(img, black_thresh=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold (black background ≈ 0)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Largest contour = retina
    largest_contour = max(contours, key=cv2.contourArea)

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop
    cropped = img[y:y+h, x:x+w]
    cropped = cv2.resize(cropped, (512,512))
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return cropped


def process_dataset():
    for split in SPLITS:
        for cls in CLASSES:
            in_dir = INPUT_ROOT / split / cls
            out_dir = OUTPUT_ROOT / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            images = [
                p for p in in_dir.iterdir()
                if p.suffix.lower() in IMG_EXTS
            ]

            for img_path in tqdm(images, desc=f"{split}/{cls}", leave=False):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # apply clahe
                img = apply_clahe(img_path)

                cleaned = clean_fundus_image(img)
                if cleaned is None:
                    continue

                out_path = out_dir / img_path.name
                cv2.imwrite(str(out_path), cleaned)


if __name__ == "__main__":
    process_dataset()

