"""
Generate Mock Synthetic Dataset for Neonatal Lung Ultrasound Triage.

Creates a standard folder structure matching PyTorch ImageFolder conventions:
data/raw/
  ├── normal/
  ├── moderate_risk/
  └── high_risk/
"""

import os
from PIL import Image
import numpy as np

# Define paths and classes
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BASE_DIR = os.path.abspath(BASE_DIR)

CLASSES = ["normal", "moderate_risk", "high_risk"]
IMAGES_PER_CLASS = 10  # Lightweight set for pipeline testing
IMAGE_SIZE = (224, 224)


def generate_mock_dataset():
    print("[INFO] Generating synthetic dataset directory structure...")

    for class_name in CLASSES:
        class_dir = os.path.join(BASE_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(1, IMAGES_PER_CLASS + 1):
            file_path = os.path.join(class_dir, f"sample_{i:02d}.png")

            # Generate random grayscale synthetic image
            random_pixels = np.random.randint(
                0,
                256,
                IMAGE_SIZE,
                dtype=np.uint8
            )

            img = Image.fromarray(random_pixels, mode="L")
            img.save(file_path)

    print(
        f"[SUCCESS] Created {IMAGES_PER_CLASS * len(CLASSES)} synthetic images "
        f"across {len(CLASSES)} classes in '{BASE_DIR}'."
    )


if __name__ == "__main__":
    generate_mock_dataset()