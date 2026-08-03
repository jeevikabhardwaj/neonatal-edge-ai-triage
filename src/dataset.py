"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 2B: Dataset Loading, Preprocessing Pipeline, & DataLoader Verification

This script loads lung ultrasound image data from disk (`data/raw/`),
applies standard preprocessing and normalization transforms, wraps the dataset
into a PyTorch DataLoader, and verifies inference compatibility via a forward pass
using the baseline MobileNetV3-Small model architecture from `src/baseline_model.py`.
"""

from __future__ import annotations


import sys
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Ensure project root and src/ are in Python path for flexible execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.baseline_model import build_baseline_model, get_device

# Default ImageNet normalization parameters (standard for MobileNetV3 backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = (224, 224)


def get_data_dir(relative_path: str = "data/raw") -> Path:
    """
    Resolves the dataset directory path relative to the project root
    and verifies that the directory exists and contains data.

    Args:
        relative_path: Path to dataset relative to project root.

    Returns:
        Path: Resolved absolute path to the dataset directory.

    Raises:
        FileNotFoundError: If the directory does not exist or is empty,
                          instructing the user to run generate_mock_data.py.
    """
    data_dir = (PROJECT_ROOT / relative_path).resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset directory not found at: '{data_dir}'\n"
            "Please generate the synthetic mock dataset first by running:\n"
            "    python src/generate_mock_data.py"
        )

    # Check if directory contains class subfolders
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"\n[ERROR] Dataset directory '{data_dir}' contains no class folders.\n"
            "Please generate the synthetic mock dataset first by running:\n"
            "    python src/generate_mock_data.py"
        )

    return data_dir


def get_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Constructs the image preprocessing and normalization pipeline.

    The pipeline performs:
    1. Resize: Rescales input images to the target dimensions (224, 224).
    2. ToTensor: Converts PIL Images (or NumPy arrays) to PyTorch Tensors
       scaled to [0.0, 1.0] and handles channel dimension (C x H x W).
    3. Normalize: Normalizes RGB channels using ImageNet standard mean and std.

    Args:
        image_size: Target spatial resolution as (height, width).
        mean: Normalization mean for (R, G, B) channels.
        std: Normalization standard deviation for (R, G, B) channels.

    Returns:
        transforms.Compose: Composed torchvision transform pipeline.
    """
    return transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std,
    ),
])


def get_dataset(
    data_dir: Optional[Path | str] = None,
    transform: Optional[transforms.Compose] = None,
) -> datasets.ImageFolder:
    """
    Loads lung ultrasound images structured by class folders using PyTorch ImageFolder.

    Expected directory structure:
        data_dir/
            ├── normal/
            ├── moderate_risk/
            └── high_risk/

    Args:
        data_dir: Path to the root directory containing class folders.
                  If None, resolves automatically via get_data_dir().
        transform: Image transform pipeline. If None, uses get_transforms().

    Returns:
        datasets.ImageFolder: PyTorch ImageFolder dataset instance.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    else:
        data_dir = Path(data_dir).resolve()
        if not data_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Specified dataset path does not exist: '{data_dir}'\n"
                "Please check the path or run 'python src/generate_mock_data.py'."
            )

    if transform is None:
        transform = get_transforms()

    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Wraps a PyTorch Dataset in a DataLoader with configurable batching and shuffling.

    Args:
        dataset: The PyTorch Dataset to wrap.
        batch_size: Number of samples per batch.
        shuffle: Whether to reshuffle data at every epoch.
        num_workers: Number of subprocesses for data loading (0 for single process).
        pin_memory: If True, copies tensors to CUDA pinned memory before returning.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def verify_pipeline(batch_size: int = 4) -> None:
    """
    Executes an end-to-end verification of the data pipeline:
    1. Resolves data path and creates dataset & transforms.
    2. Builds DataLoader and extracts a sample batch.
    3. Instantiates the baseline MobileNetV3 model.
    4. Executes a forward pass and verifies tensor dimensions.
    """
    print("=" * 70)
    print("Milestone 2B: Dataset Loading & Model Inference Verification")
    print("=" * 70)

    # 1. Device Selection
    device = get_device()
    print(f"[1] Compute Device: {device}")

    # 2. Path Resolution & Dataset Loading
    data_dir = get_data_dir()
    print(f"[2] Resolved Dataset Directory: {data_dir}")

    transform_pipeline = get_transforms()
    dataset = get_dataset(data_dir=data_dir, transform=transform_pipeline)

    print(f"[3] Total Samples Found: {len(dataset)}")
    print(f"[4] Detected Classes ({len(dataset.classes)}): {dataset.classes}")
    print(f"    Class-to-Index Mapping: {dataset.class_to_idx}")

    # 3. DataLoader Construction
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"[5] DataLoader Initialized: batch_size={batch_size}, batches={len(dataloader)}")

    # 4. Batch Inspection
    images, labels = next(iter(dataloader))
    print(f"[6] Sample Batch Loaded:")
    print(f"    - Image Tensor Shape: {images.shape} (Batch, Channels, Height, Width)")
    print(f"    - Image Tensor Dtype: {images.dtype}")
    print(f"    - Image Value Range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"    - Label Tensor Shape: {labels.shape}")
    print(f"    - Ground Truth Labels in Batch: {labels.tolist()}")

    # 5. Baseline Model Forward Pass
    num_classes = len(dataset.classes)
    model = build_baseline_model(num_classes=num_classes)
    model.to(device)
    model.eval()

    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    print(f"[7] Forward Pass Output:")
    print(f"    - Logits Tensor Shape: {logits.shape}")
    print(f"    - Predicted Classes: {predicted_classes.cpu().tolist()}")
    print(f"    - Output Device: {logits.device}")

    # Shape Assertion
    expected_shape = (batch_size, num_classes)
    assert logits.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
    )

    print("-" * 70)
    print(
        "[SUCCESS] Data loading, preprocessing pipeline, DataLoader, "
        "and baseline forward pass verified successfully!"
    )
    print("=" * 70)


def main() -> None:
    """Entry point for standalone dataset verification."""
    verify_pipeline()


if __name__ == "__main__":
    main()
