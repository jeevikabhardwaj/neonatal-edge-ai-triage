"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 3: Baseline Image Classifier Training & Validation Pipeline

This module implements a modular, reproducible training and evaluation pipeline
for the baseline MobileNetV3-Small image classification model. It trains on
synthetic lung ultrasound images using transfer learning (frozen feature extractor),
evaluates on an 80/20 train/validation split, and saves the best model checkpoint.
"""

from __future__ import annotations


import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure project root and src/ are in sys.path for flexible execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.baseline_model import build_baseline_model, get_device
    from src.dataset import get_dataloader, get_dataset
except ImportError:
    from baseline_model import build_baseline_model, get_device
    from dataset import get_dataloader, get_dataset


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python random, NumPy, and PyTorch to ensure
    deterministic behavior and reproducible splits/initializations.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Splits a PyTorch Dataset into training and validation subsets using
    a deterministic random generator.

    Args:
        dataset: The full PyTorch Dataset to split.
        train_ratio: Proportion of samples allocated to training (default: 0.8).
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, val_dataset)
    """
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_dataset, val_dataset


def get_train_val_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wraps train and validation datasets in PyTorch DataLoaders.

    Args:
        train_dataset: Training subset.
        val_dataset: Validation subset.
        batch_size: Number of samples per batch.
        num_workers: Number of worker subprocesses for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Executes a single training epoch over the training dataset.

    Args:
        model: PyTorch neural network model.
        dataloader: Training DataLoader.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: PyTorch optimizer (e.g. AdamW).
        device: Target compute device (CPU, CUDA, or MPS).

    Returns:
        Tuple[float, float]: (average_training_loss, training_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Track statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_accuracy = running_corrects / total_samples if total_samples > 0 else 0.0

    return epoch_loss, epoch_accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation dataset without updating gradients.

    Args:
        model: PyTorch neural network model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Target compute device.

    Returns:
        Tuple[float, float]: (average_validation_loss, validation_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    val_accuracy = running_corrects / total_samples if total_samples > 0 else 0.0

    return val_loss, val_accuracy


def save_checkpoint(
    checkpoint: Dict[str, Any],
    filepath: Union[Path, str],
) -> None:
    """
    Saves the model checkpoint dictionary to the specified file path.

    Args:
        checkpoint: Dictionary containing epoch, model_state_dict,
                    optimizer_state_dict, validation_loss, etc.
        filepath: Destination file path.
    """
    dest_path = Path(filepath).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(dest_path))


def run_training(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    seed: int = 42,
    checkpoint_dir: Union[Path, str] = "models",
    checkpoint_name: str = "best_baseline_model.pth",
    num_workers: int = 0,
) -> Dict[str, List[float]]:
    """
    Coordinates the full training and validation lifecycle:
    1. Sets random seed for reproducibility.
    2. Loads dataset and performs 80/20 train/val split.
    3. Builds baseline model and freezes feature extraction layers.
    4. Trains classifier using AdamW and CrossEntropyLoss.
    5. Saves the checkpoint whenever validation loss improves.

    Args:
        epochs: Number of training epochs (default: 5).
        batch_size: Mini-batch size for DataLoaders (default: 4).
        learning_rate: Learning rate for AdamW optimizer (default: 1e-4).
        seed: Random seed for reproducibility.
        checkpoint_dir: Directory to save model checkpoints.
        checkpoint_name: Filename of the best checkpoint.
        num_workers: DataLoader worker subprocess count.

    Returns:
        Dict[str, List[float]]: Training history with loss and accuracy metrics.
    """
    # 1. Device and Seed Setup
    set_seed(seed)
    device = get_device()
    print("=" * 60)
    print("Multimodal Edge AI System - Milestone 3 Training Pipeline")
    print("=" * 60)
    print(f"[INFO] Compute Device: {device}")

    # 2. Dataset Preparation & 80/20 Split
    full_dataset = get_dataset()
    classes = full_dataset.classes
    num_classes = len(classes)

    train_dataset, val_dataset = split_dataset(
        dataset=full_dataset,
        train_ratio=0.8,
        seed=seed,
    )

    print(f"[INFO] Dataset Loaded: {len(full_dataset)} total samples across {num_classes} classes.")
    print(f"[INFO] Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")
    print(f"[INFO] Classes: {classes} (Mapping: {full_dataset.class_to_idx})")

    # 3. DataLoaders
    train_loader, val_loader = get_train_val_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 4. Model Initialization & Transfer Learning Freezing
    model = build_baseline_model(num_classes=num_classes)

    # Freeze ALL pretrained feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Ensure classifier head parameters are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Feature extractor frozen. Trainable parameters: {trainable_params:,} / {total_params:,}")

    # 5. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=learning_rate)

    # 6. Checkpoint Setup
    checkpoint_path = (PROJECT_ROOT / checkpoint_dir / checkpoint_name).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 7. Training History Tracking
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_val_loss = float("inf")

    # 8. Training and Validation Loop
    for epoch in range(1, epochs + 1):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Validate one epoch
        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_acc)

        # Formatted epoch logging
        print("-" * 48)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss         : {train_loss:.4f}")
        print(f"Train Accuracy     : {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"Validation Loss   : {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
"train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "classes": classes,
            }
            save_checkpoint(checkpoint, checkpoint_path)
            print(f"[CHECKPOINT] Validation loss improved to {val_loss:.4f}. Checkpoint saved.")

    print("-" * 48)
    return history


def main() -> None:
    """Main execution function for Milestone 3 training pipeline."""
    set_seed(seed=42)
    run_training(epochs=5)

    checkpoint_rel_path = "models/best_baseline_model.pth"
    print("\nTraining completed successfully.")
    print(f"Checkpoint saved at:\n{checkpoint_rel_path}")


if __name__ == "__main__":
    main()
