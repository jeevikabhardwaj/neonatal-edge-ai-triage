"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 3: Baseline Image Classifier Training & Validation Pipeline

This module implements a configurable, reproducible, research-grade training
and validation pipeline for the baseline MobileNetV3-Small triage model.
It supports YAML configuration, transfer learning with frozen backbones,
class imbalance handling via loss weighting and weighted sampling,
learning rate scheduling, early stopping, checkpointing, and CSV logging.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from tqdm import tqdm

# Ensure project root and src/ are in sys.path for flexible execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.baseline_model import (
        build_baseline_model,
        get_device,
        get_model_metadata,
    )
    from src.dataset import (
        compute_class_weights,
        get_dataloader,
        get_dataset,
        get_weighted_sampler,
        set_seed,
    )
except ImportError:
    from baseline_model import (
        build_baseline_model,
        get_device,
        get_model_metadata,
    )
    from dataset import (
        compute_class_weights,
        get_dataloader,
        get_dataset,
        get_weighted_sampler,
        set_seed,
    )


# =====================================================================
# Configuration Loader
# =====================================================================
def load_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Loads training and model configuration from a YAML file.

    Args:
        config_path: Optional path to YAML configuration file.
            If None, loads config/train_config.yaml relative to project root.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "train_config.yaml"
    else:
        config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Configuration file not found at: '{config_path}'"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config


# =====================================================================
# Dataset Splitting & DataLoader Builders
# =====================================================================
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
    pin_memory: bool = False,
    use_weighted_sampler: bool = False,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wraps train and validation datasets in PyTorch DataLoaders with optional
    weighted sampling for class imbalance.

    Args:
        train_dataset: Training subset.
        val_dataset: Validation subset.
        batch_size: Number of samples per batch.
        num_workers: Number of worker subprocesses for data loading.
        pin_memory: If True, copies tensors to pinned memory.
        use_weighted_sampler: Whether to use WeightedRandomSampler for training.
        shuffle: Whether to shuffle training data if not using weighted sampler.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    sampler = None
    if use_weighted_sampler:
        # Extract target labels for train_dataset (handling Subset or ImageFolder)
        if isinstance(train_dataset, torch.utils.data.Subset):
            full_dataset = train_dataset.dataset
            targets = [full_dataset.targets[i] for i in train_dataset.indices]
            num_classes = (
                len(full_dataset.classes)
                if hasattr(full_dataset, "classes")
                else len(set(targets))
            )
        else:
            targets = (
                [s[1] for s in train_dataset.samples]
                if hasattr(train_dataset, "samples")
                else list(train_dataset.targets)
            )
            num_classes = (
                len(train_dataset.classes)
                if hasattr(train_dataset, "classes")
                else len(set(targets))
            )

        class_counts = [max(targets.count(idx), 1) for idx in range(num_classes)]
        class_weights = [1.0 / count for count in class_counts]
        sample_weights = [class_weights[t] for t in targets]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# =====================================================================
# Training & Validation Epoch Loops
# =====================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Executes a single training epoch over the training dataset.

    Args:
        model: PyTorch neural network model.
        dataloader: Training DataLoader.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: PyTorch optimizer (e.g. AdamW).
        device: Target compute device (CPU, CUDA, or MPS).
        epoch: Current epoch index for progress bar display.
        epochs: Total epoch count for progress bar display.

    Returns:
        Tuple[float, float]: (average_training_loss, training_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    desc = f"Epoch {epoch}/{epochs} [Train]" if epoch and epochs else "Training"
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    for images, labels in progress_bar:
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

        batch_acc = torch.sum(preds == labels.data).item() / batch_size
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2%}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_accuracy = running_corrects / total_samples if total_samples > 0 else 0.0

    return epoch_loss, epoch_accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation dataset without updating gradients.

    Args:
        model: PyTorch neural network model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Target compute device.
        epoch: Current epoch index for progress bar display.
        epochs: Total epoch count for progress bar display.

    Returns:
        Tuple[float, float]: (average_validation_loss, validation_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    desc = f"Epoch {epoch}/{epochs} [Val]" if epoch and epochs else "Validation"
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size

            batch_acc = torch.sum(preds == labels.data).item() / batch_size
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2%}")

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    val_accuracy = running_corrects / total_samples if total_samples > 0 else 0.0

    return val_loss, val_accuracy


# =====================================================================
# Persistence & Logging Utilities
# =====================================================================
def save_checkpoint(
    checkpoint: Dict[str, Any],
    filepath: Union[Path, str],
) -> None:
    """
    Saves the model checkpoint dictionary to the specified file path.

    Args:
        checkpoint: Dictionary containing model_state_dict, optimizer_state_dict,
            scheduler_state_dict, epoch, validation metrics, history, and metadata.
        filepath: Destination file path.
    """
    dest_path = Path(filepath).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(dest_path))


def save_history_csv(
    history: Dict[str, List[Any]],
    filepath: Union[Path, str],
) -> None:
    """
    Exports the training history metrics to a CSV file.

    Args:
        history: Dictionary containing lists for epoch, train_loss, train_accuracy,
            validation_loss, validation_accuracy, and learning_rate.
        filepath: Destination CSV filepath.
    """
    dest_path = Path(filepath).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "learning_rate",
    ]

    with open(dest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        num_epochs = len(history.get("epoch", []))
        for i in range(num_epochs):
            row = {
                field: history[field][i]
                for field in fieldnames
                if field in history and i < len(history[field])
            }
            writer.writerow(row)


# =====================================================================
# Orchestrated Training Pipeline
# =====================================================================
def run_training(
    config: Optional[Dict[str, Any]] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = None,
    checkpoint_dir: Optional[Union[Path, str]] = None,
    checkpoint_name: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """
    Coordinates the full training, validation, early stopping, and logging lifecycle:
    1. Loads configuration from YAML (with optional CLI overrides).
    2. Seeds all random generators for full reproducibility.
    3. Loads raw ultrasound dataset and executes train/validation split.
    4. Configures optional class weighting and weighted random sampling.
    5. Initializes MobileNetV3-Small with specified backbone freeze state.
    6. Executes training loop with ReduceLROnPlateau and early stopping.
    7. Saves best model checkpoint in models/checkpoints/ and exports history CSV in models/history/.

    Args:
        config: Optional configuration dictionary.
        epochs: Optional epoch count override.
        batch_size: Optional batch size override.
        learning_rate: Optional learning rate override.
        seed: Optional random seed override.
        checkpoint_dir: Optional checkpoint directory override.
        checkpoint_name: Optional checkpoint filename override.
        num_workers: Optional DataLoader worker count override.

    Returns:
        Dict[str, List[Any]]: Comprehensive training history metrics dictionary.
    """
    start_time = time.time()

    # 1. Load Configuration
    if config is None:
        config = load_config()

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    ckpt_cfg = config.get("checkpoint", {})
    history_cfg = config.get("history", {})

    effective_seed = seed if seed is not None else config.get("seed", 42)
    effective_epochs = epochs if epochs is not None else train_cfg.get("epochs", 5)
    effective_batch_size = (
        batch_size if batch_size is not None else data_cfg.get("batch_size", 4)
    )
    effective_lr = (
        learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 1e-4)
    )
    effective_num_workers = (
        num_workers if num_workers is not None else data_cfg.get("num_workers", 0)
    )
    effective_ckpt_dir = (
        checkpoint_dir
        if checkpoint_dir is not None
        else ckpt_cfg.get("checkpoint_dir", ckpt_cfg.get("directory", "models/checkpoints"))
    )
    effective_ckpt_name = (
        checkpoint_name
        if checkpoint_name is not None
        else ckpt_cfg.get("checkpoint_name", ckpt_cfg.get("filename", "best_baseline_model.pth"))
    )
    history_csv_cfg = history_cfg.get("csv_path", ckpt_cfg.get("history_csv", "models/history/training_history.csv"))

    weight_decay = train_cfg.get("weight_decay", 0.01)
    scheduler_patience = train_cfg.get("scheduler_patience", 2)
    scheduler_factor = train_cfg.get("scheduler_factor", 0.5)
    early_stopping_enabled = train_cfg.get("early_stopping", True)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 5)
    use_class_weights = train_cfg.get("use_class_weights", True)
    use_weighted_sampler = data_cfg.get("use_weighted_sampler", False)
    val_split = data_cfg.get("validation_split", 0.2)
    shuffle = data_cfg.get("shuffle", True)
    pin_memory = data_cfg.get("pin_memory", False)
    freeze_backbone = model_cfg.get("freeze_backbone", True)

    # 2. Reproducibility & Device Configuration
    set_seed(effective_seed)
    device = get_device()

    # 3. Dataset Ingestion & Split
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    full_dataset = get_dataset(data_dir=PROJECT_ROOT / raw_dir)
    classes = full_dataset.classes
    num_classes = model_cfg.get("num_classes", len(classes))

    train_dataset, val_dataset = split_dataset(
        dataset=full_dataset,
        train_ratio=1.0 - val_split,
        seed=effective_seed,
    )

    # 4. DataLoaders Construction
    train_loader, val_loader = get_train_val_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=effective_batch_size,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        use_weighted_sampler=use_weighted_sampler,
        shuffle=shuffle,
    )

    # 5. Model Initialization & Transfer Learning
    model = build_baseline_model(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )
    model.to(device)

    model_meta = get_model_metadata(model=model, freeze_backbone=freeze_backbone)

    # 6. Loss Function Setup
    if use_class_weights:
        class_weights = compute_class_weights(full_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 7. Optimizer & Learning Rate Scheduler
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=effective_lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    # 8. Checkpoint & History Paths
    checkpoint_path = (PROJECT_ROOT / effective_ckpt_dir / effective_ckpt_name).resolve()
    history_csv_path = (PROJECT_ROOT / history_csv_cfg).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 9. Header Summary Output
    print("=" * 60)
    print("Multimodal Edge AI System - Milestone 3 Training Pipeline")
    print("=" * 60)
    print(f"Device               : {device}")
    print(f"Dataset Size         : {len(full_dataset)} samples")
    print(f"Training Samples     : {len(train_dataset)}")
    print(f"Validation Samples   : {len(val_dataset)}")
    print(f"Classes ({num_classes})          : {classes} (Mapping: {full_dataset.class_to_idx})")
    print(f"Optimizer            : {train_cfg.get('optimizer', 'AdamW')} (lr={effective_lr}, weight_decay={weight_decay})")
    print(f"Scheduler            : {train_cfg.get('scheduler', 'ReduceLROnPlateau')} (patience={scheduler_patience}, factor={scheduler_factor})")
    print(f"Early Stopping       : {'Enabled (patience=' + str(early_stopping_patience) + ')' if early_stopping_enabled else 'Disabled'}")
    print(f"Class Weights        : {'Enabled' if use_class_weights else 'Disabled'}")
    print(f"Weighted Sampler     : {'Enabled' if use_weighted_sampler else 'Disabled'}")
    print(f"Frozen Backbone      : {'Yes' if freeze_backbone else 'No'}")
    print(f"Trainable Parameters : {model_meta['trainable_parameters']:,} / {model_meta['total_parameters']:,}")
    print("=" * 60)

    # 10. Training Execution Loop
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, effective_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=effective_epochs,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=effective_epochs,
        )

        # Scheduler step with validation loss
        scheduler.step(val_loss)

        # Record metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_acc)
        history["learning_rate"].append(current_lr)

        # Log epoch results
        print("-" * 50)
        print(f"Epoch {epoch}/{effective_epochs}")
        print(f"Train Loss         : {train_loss:.4f}")
        print(f"Train Accuracy     : {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"Validation Loss   : {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
        print(f"Learning Rate      : {current_lr:.6f}")

        # Checkpoint evaluation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_loss": best_val_loss,
                "validation_loss": val_loss,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "history": history,
                "classes": classes,
                "model_version": "baseline_mobilenetv3_v1",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_metadata": get_model_metadata(model=model, freeze_backbone=freeze_backbone),
                "configuration": config,
            }
            save_checkpoint(checkpoint, checkpoint_path)
            rel_ckpt_str = str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.is_relative_to(PROJECT_ROOT) else str(checkpoint_path)
            print(f"[CHECKPOINT] Validation loss improved to {val_loss:.4f}. Checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(f"[INFO] No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        # Early stopping condition
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            print("\n[EARLY STOPPING]")
            print(
                f"Validation loss did not improve for {early_stopping_patience} consecutive epochs. Stopping training."
            )
            early_stopped = True
            break

    # 11. Export Training History CSV
    save_history_csv(history, history_csv_path)

    # 12. Final Console Summary
    duration = time.time() - start_time
    rel_ckpt_path = str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.is_relative_to(PROJECT_ROOT) else str(checkpoint_path)
    rel_csv_path = str(history_csv_path.relative_to(PROJECT_ROOT)) if history_csv_path.is_relative_to(PROJECT_ROOT) else str(history_csv_path)

    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Device                 : {device}")
    print(f"Dataset Size           : {len(full_dataset)} total samples")
    print(f"Training Samples       : {len(train_dataset)}")
    print(f"Validation Samples     : {len(val_dataset)}")
    print(f"Optimizer              : {train_cfg.get('optimizer', 'AdamW')}")
    print(f"Scheduler              : {train_cfg.get('scheduler', 'ReduceLROnPlateau')}")
    print(f"Initial Learning Rate  : {effective_lr}")
    print(f"Epochs Completed       : {len(history['epoch'])} / {effective_epochs}")
    print(f"Early Stopping Status  : {'Triggered' if early_stopped else 'Did not trigger'}")
    print(f"Training Duration      : {duration:.2f}s ({duration / 60:.2f} min)")
    print(f"Final Best Val Loss    : {best_val_loss:.4f}")
    print(f"Checkpoint saved:\n{rel_ckpt_path}")
    print(f"Training history:\n{rel_csv_path}")
    print("=" * 60)

    return history


def main() -> None:
    """Main execution function for Milestone 3 training pipeline."""
    config = load_config()
    set_seed(seed=config.get("seed", 42))
    run_training(config=config)
    print("\n[SUCCESS] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
