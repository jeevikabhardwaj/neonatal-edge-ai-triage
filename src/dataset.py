"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 2: Medical Imaging Dataset Pipeline & Preprocessing Module

Pipeline Architecture & Methodological Rationale:
------------------------------------------------
1. Purpose of Dataset Pipeline:
   Provides a standardized, robust, and reproducible data ingestion and preprocessing
   framework for neonatal lung ultrasound (LUS) imagery. The pipeline handles raw image
   loading, domain-specific spatial and photometric transformations, tensor formatting,
   and batch generation for edge AI triage models.

2. Directory Structure & ImageFolder Utilization:
   PyTorch's `datasets.ImageFolder` is utilized as the standard dataset structure
   where subdirectories directly correspond to diagnostic risk tiers:
   `high_risk/`, `moderate_risk/`, and `normal/`. This convention ensures clean
   separation of clinical classes, deterministic label mapping, and modular extensibility
   for future multi-center clinical cohorts.

3. ImageNet Normalization Rationale:
   Input tensors are normalized using standard ImageNet channel statistics
   (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]). Because our triage models
   utilize pretrained vision backbones (e.g., MobileNetV3-Small), aligning input pixel
   distributions with the pretraining domain preserves learned convolutional filter activations
   and accelerates transfer learning convergence.

4. Separate Train/Evaluation Transform Pipelines:
   - Training Transforms: Integrate gentle, clinically valid augmentations (mild rotation
     up to ±10°, moderate color jitter) to improve model invariance to ultrasound probe
     tilt, gain settings, and acoustic attenuation without introducing unrealistic artifacts.
     (Note: Horizontal flipping is omitted to maintain anatomical orientation consistency).
   - Evaluation/Inference Transforms: Strictly deterministic resizing, grayscale-to-RGB
     replication, and normalization to guarantee unbiased, reproducible diagnostic metrics.

5. Reproducibility & Class Imbalance in Medical AI:
   Neonatal clinical datasets exhibit natural prevalence skew (e.g., high-risk respiratory
   distress cases are rarer than normal presentations). Deterministic random seeding ensures
   identical train/validation partitions across experimental runs. Inverse-frequency class
   weighting and weighted random sampling utilities prevent majority-class bias and ensure
   high diagnostic sensitivity on critical minority classes without discarding valuable samples.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import datasets, transforms

# Ensure project root and src/ are in Python path for flexible execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.baseline_model import build_baseline_model, get_device
except ImportError:
    from baseline_model import build_baseline_model, get_device

# =====================================================================
# Global Constants
# =====================================================================
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_SEED: int = 42


# =====================================================================
# Reproducibility Utilities
# =====================================================================
def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Sets deterministic random seeds across Python, NumPy, and PyTorch
    to ensure full experiment reproducibility in medical imaging benchmarks.

    Args:
        seed: Integer seed value (default: DEFAULT_SEED).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# =====================================================================
# Dataset Path Resolution
# =====================================================================
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


# =====================================================================
# Image Transform Pipeline
# =====================================================================
def get_transforms(
    is_train: bool = True,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Constructs the image preprocessing and augmentation transform pipeline.

    Training transforms:
    - Resize(DEFAULT_IMAGE_SIZE)
    - Convert grayscale images to RGB
    - RandomRotation(degrees=10)
    - ColorJitter(brightness=0.2, contrast=0.2)
    - ToTensor()
    - Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Validation / inference transforms:
    - Resize(DEFAULT_IMAGE_SIZE)
    - Convert grayscale images to RGB
    - ToTensor()
    - Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Note: RandomHorizontalFlip is intentionally excluded because lung ultrasound
    anatomic orientation (pleural line, probe marker) possesses diagnostic polarity.

    Args:
        is_train: Whether to include data augmentations for training (default: True).
        image_size: Target spatial dimensions as (height, width).
        mean: Normalization channel means.
        std: Normalization channel standard deviations.

    Returns:
        transforms.Compose: Composed torchvision transform pipeline.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# =====================================================================
# Dataset Loader
# =====================================================================
def get_dataset(
    data_dir: Optional[Union[Path, str]] = None,
    transform: Optional[transforms.Compose] = None,
    is_train: bool = False,
) -> datasets.ImageFolder:
    """
    Loads lung ultrasound images structured by class folders using PyTorch ImageFolder.

    Expected directory structure:
        data_dir/
            ├── high_risk/
            ├── moderate_risk/
            └── normal/

    Args:
        data_dir: Path to the root directory containing class folders.
            If None, resolves automatically via get_data_dir().
        transform: Image transform pipeline. If None, uses get_transforms(is_train=is_train).
        is_train: If transform is None, indicates whether to apply training augmentations.

    Returns:
        datasets.ImageFolder: PyTorch ImageFolder dataset instance.
    """
    if data_dir is None:
        resolved_data_dir = get_data_dir()
    else:
        resolved_data_dir = Path(data_dir).resolve()
        if not resolved_data_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Specified dataset path does not exist: '{resolved_data_dir}'\n"
                "Please check the path or run 'python src/generate_mock_data.py'."
            )

    if transform is None:
        transform = get_transforms(is_train=is_train)

    dataset = datasets.ImageFolder(root=str(resolved_data_dir), transform=transform)
    return dataset


# =====================================================================
# Dataset Statistics
# =====================================================================
def get_dataset_statistics(dataset: datasets.ImageFolder) -> Dict[str, Any]:
    """
    Computes summary statistics for a dataset instance.

    Args:
        dataset: PyTorch ImageFolder dataset instance.

    Returns:
        Dict[str, Any]: Dictionary containing dataset metrics:
            - num_samples: Total number of samples in the dataset.
            - num_classes: Number of distinct target classes.
            - class_names: List of class category names.
            - class_counts: Dictionary mapping class names to sample counts.
            - image_size: Standard input image resolution (H, W).
    """
    targets = (
        [target for _, target in dataset.samples]
        if hasattr(dataset, "samples")
        else list(dataset.targets)
    )
    class_counts = {
        cls_name: targets.count(idx)
        for cls_name, idx in dataset.class_to_idx.items()
    }
    return {
        "num_samples": len(dataset),
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "class_counts": class_counts,
        "image_size": DEFAULT_IMAGE_SIZE,
    }


# =====================================================================
# Class Imbalance Utilities
# =====================================================================
def compute_class_weights(dataset: datasets.ImageFolder) -> torch.FloatTensor:
    """
    Computes inverse-frequency class weights suitable for loss weighting
    (e.g., nn.CrossEntropyLoss(weight=...)) to counteract class imbalance.

    Weight formula for class c: w_c = N / (C * N_c)

    Args:
        dataset: PyTorch ImageFolder dataset instance.

    Returns:
        torch.FloatTensor: 1D Tensor of inverse-frequency class weights.
    """
    targets = (
        [target for _, target in dataset.samples]
        if hasattr(dataset, "samples")
        else list(dataset.targets)
    )
    num_samples = len(targets)
    num_classes = len(dataset.classes)

    class_counts = [targets.count(idx) for idx in range(num_classes)]
    weights = [
        num_samples / (num_classes * max(count, 1))
        for count in class_counts
    ]
    return torch.tensor(weights, dtype=torch.float32)


# =====================================================================
# Weighted Random Sampler
# =====================================================================
def get_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Constructs a PyTorch WeightedRandomSampler using inverse-frequency
    per-sample weights for balanced batch sampling.

    Args:
        dataset: PyTorch ImageFolder dataset instance.

    Returns:
        WeightedRandomSampler: Configured sampler with replacement.
    """
    targets = (
        [target for _, target in dataset.samples]
        if hasattr(dataset, "samples")
        else list(dataset.targets)
    )
    num_classes = len(dataset.classes)
    class_counts = [max(targets.count(idx), 1) for idx in range(num_classes)]
    class_weights = [1.0 / count for count in class_counts]

    sample_weights = [class_weights[target] for target in targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# =====================================================================
# DataLoader
# =====================================================================
def get_dataloader(
    dataset: Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """
    Wraps a PyTorch Dataset in a DataLoader with configurable batching, shuffling,
    and optional weighted sampling.

    Args:
        dataset: The PyTorch Dataset to wrap.
        batch_size: Number of samples per batch (default: DEFAULT_BATCH_SIZE).
        shuffle: Whether to reshuffle data at every epoch (ignored if sampler is provided).
        num_workers: Number of subprocesses for data loading (0 for single process).
        pin_memory: If True, copies tensors to CUDA pinned memory before returning.
        sampler: Optional PyTorch Sampler instance (e.g. WeightedRandomSampler).

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# =====================================================================
# Pipeline Verification
# =====================================================================
def verify_pipeline(batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """
    Executes an end-to-end verification of the data pipeline:
    1. Sets deterministic random seed for reproducibility.
    2. Resolves data path and creates dataset with training augmentations.
    3. Computes and displays dataset statistics, active transforms, and class imbalance weights.
    4. Builds DataLoader and extracts a sample batch.
    5. Instantiates the baseline MobileNetV3 model on optimal device.
    6. Executes a forward pass and asserts tensor dimensions.
    """
    print("=" * 70)
    print("Milestone 2: Dataset Pipeline & Preprocessing Verification")
    print("=" * 70)

    # 1. Reproducibility & Seed Configuration
    set_seed(DEFAULT_SEED)

    # 2. Path Resolution & Dataset Loading
    data_dir = get_data_dir()
    print(f"[1] Resolved Dataset Directory: {data_dir}")

    dataset = get_dataset(data_dir=data_dir, is_train=True)
    stats = get_dataset_statistics(dataset)

    print(f"[2] Dataset Statistics:")
    print(f"    - Total Samples    : {stats['num_samples']}")
    print(f"    - Classes ({stats['num_classes']})      : {stats['class_names']}")
    print(f"    - Class Counts     : {stats['class_counts']}")
    print(f"    - Input Resolution : {stats['image_size']}")
    print(f"    - Class-to-Index   : {dataset.class_to_idx}")

    # 3. Active Transforms Summary
    print(f"[3] Applied Training Transforms:")
    print("    - Resize(224,224)")
    print("    - Grayscale → RGB")
    print("    - RandomRotation(10°)")
    print("    - ColorJitter(brightness=0.2, contrast=0.2)")
    print("    - ToTensor()")
    print("    - Normalize(ImageNet)")

    # 4. Class Imbalance Inspection
    class_weights = compute_class_weights(dataset)
    print(f"[4] Computed Class Weights (Inverse Frequency):")
    for cls_name, idx in dataset.class_to_idx.items():
        print(f"    - {cls_name:<14} (idx {idx}): {class_weights[idx].item():.4f}")

    # 5. DataLoader Construction
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"[5] DataLoader Initialized: batch_size={batch_size}, batches={len(dataloader)}")

    # 6. Batch Inspection
    images, labels = next(iter(dataloader))
    print(f"[6] Sample Batch Loaded:")
    print(f"    - Input Batch Shape : {images.shape} (Batch, Channels, Height, Width)")
    print(f"    - Label Shape       : {labels.shape}")
    print(f"    - Image Dtype       : {images.dtype}")
    print(f"    - Value Range       : [{images.min():.3f}, {images.max():.3f}]")
    print(f"    - Batch Labels      : {labels.tolist()}")

    # 7. Baseline Model Forward Pass Verification
    device = get_device()
    num_classes = len(dataset.classes)
    model = build_baseline_model(num_classes=num_classes)
    model.to(device)
    model.eval()

    images = images.to(device)

    with torch.no_grad():
        output = model(images)
        probabilities = torch.softmax(output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    # 8. Output Verification & Assertions
    expected_shape = (batch_size, num_classes)
    assert output.shape == expected_shape, (
        f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"
    )

    print(f"[7] Forward Pass Output:")
    print(f"    - Output Tensor Shape : {output.shape}")
    print(f"    - Output Device       : {output.device}")
    print(f"    - Output Dtype        : {output.dtype}")
    print(f"    - Predicted Classes   : {predicted_classes.cpu().tolist()}")

    print("-" * 70)
    print("[SUCCESS] Dataset pipeline verified successfully.")
    print("=" * 70)


def main() -> None:
    """Entry point for standalone dataset verification."""
    verify_pipeline()


if __name__ == "__main__":
    main()
