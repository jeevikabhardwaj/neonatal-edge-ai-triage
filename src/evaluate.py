"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 4: Model Evaluation, Performance Metrics, and Artifact Generation

This module implements a standalone, reproducible evaluation pipeline for the
trained MobileNetV3-Small lung ultrasound triage model. It loads the best checkpoint,
reconstructs the exact validation split, computes comprehensive classification metrics
(accuracy, precision, recall, F1, confusion matrix), and generates visualization artifacts
(training curves, confusion matrix plot, evaluation text report).
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless artifact generation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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


def load_checkpoint(
    checkpoint_path: Union[Path, str] = "models/best_baseline_model.pth",
) -> Dict[str, Any]:
    """
    Loads the saved model checkpoint from disk and returns its contents.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.

    Returns:
        Dict[str, Any]: Structured checkpoint dictionary containing model weights,
                        optimizer state, training metadata, metrics, and history.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at the specified path.
    """
    resolved_path = (PROJECT_ROOT / checkpoint_path).resolve() if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)

    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint file not found at: '{resolved_path}'\n"
            "Please train the baseline model first by running:\n"
            "    python src/train.py"
        )

    checkpoint: Dict[str, Any] = torch.load(str(resolved_path), map_location="cpu")
    return checkpoint


def get_validation_loader(
    dataset: Optional[Dataset] = None,
    train_ratio: float = 0.8,
    batch_size: int = 4,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, List[str]]:
    """
    Reconstructs the exact validation dataset and DataLoader used during training
    by applying the same deterministic random generator seed.

    Args:
        dataset: Full dataset instance (if None, loaded via get_dataset()).
        train_ratio: Proportion allocated to training during split (default: 0.8).
        batch_size: Mini-batch size for DataLoader (default: 4).
        seed: Random seed for reproducible splitting (default: 42).
        num_workers: Number of worker subprocesses for data loading.

    Returns:
        Tuple[DataLoader, List[str]]: Validation DataLoader and list of class names.
    """
    if dataset is None:
        dataset = get_dataset()

    classes = dataset.classes if hasattr(dataset, "classes") else []
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size

    generator = torch.Generator().manual_seed(seed)
    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return val_loader, classes


def load_trained_model(
    checkpoint: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Reconstructs the MobileNetV3-Small architecture, loads trained weights
    from the checkpoint dictionary, and prepares the model for evaluation.

    Args:
        checkpoint: Loaded checkpoint dictionary containing model_state_dict and classes.
        device: Target compute device (if None, auto-detected via get_device()).

    Returns:
        nn.Module: Reconstructed model in evaluation mode on target device.
    """
    if device is None:
        device = get_device()

    classes = checkpoint.get("classes", ["class_0", "class_1", "class_2"])
    num_classes = len(classes)

    model = build_baseline_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs model inference over the validation DataLoader and collects
    ground truth labels, predicted class labels, and output probabilities.

    Args:
        model: Trained PyTorch neural network model in eval mode.
        dataloader: Validation DataLoader.
        device: Target compute device.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - y_true: Ground truth labels (N,)
            - y_pred: Predicted class indices (N,)
            - y_probs: Softmax class probabilities (N, C)
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted_classes.cpu().numpy().tolist())
            all_probs.append(probabilities.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs) if all_probs else np.empty((0,))

    return y_true, y_pred, y_probs


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification performance metrics via scikit-learn.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        target_names: List of class label strings.

    Returns:
        Dict[str, Any]: Dictionary containing accuracy, precision, recall,
                        f1 score, classification report, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "classification_report": clf_report,
        "confusion_matrix": cm,
    }


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Union[Path, str] = "models/training_curves.png",
) -> Path:
    """
    Generates a two-panel visualization of training & validation loss and accuracy curves.

    Args:
        history: Training history dictionary with train/validation loss and accuracy lists.
        output_path: Destination path for the saved image artifact.

    Returns:
        Path: Resolved absolute path to the generated image file.
    """
    dest_path = (PROJECT_ROOT / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("validation_loss", [])
    train_acc = history.get("train_accuracy", [])
    val_acc = history.get("validation_accuracy", [])

    epochs = list(range(1, max(len(train_loss), len(val_loss), 1) + 1))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    # Top Plot: Training & Validation Loss
    ax1.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", marker="o", linewidth=2)
    ax1.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", marker="s", linewidth=2)
    ax1.set_title("Training & Validation Loss per Epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Bottom Plot: Training & Validation Accuracy
    ax2.plot(epochs, train_acc, label="Training Accuracy", color="#2ca02c", marker="o", linewidth=2)
    ax2.plot(epochs, val_acc, label="Validation Accuracy", color="#d62728", marker="s", linewidth=2)
    ax2.set_title("Training & Validation Accuracy per Epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(str(dest_path), dpi=300)
    plt.close(fig)

    return dest_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[Path, str] = "models/confusion_matrix.png",
) -> Path:
    """
    Plots and saves a high-contrast confusion matrix using pure matplotlib.

    Args:
        cm: 2D numpy array representing the confusion matrix.
        class_names: Names corresponding to class indices.
        output_path: Destination path for the saved image artifact.

    Returns:
        Path: Resolved absolute path to the generated image file.
    """
    dest_path = (PROJECT_ROOT / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels and ticks
    num_classes = len(class_names)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Validation Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate cell values
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    plt.tight_layout()
    plt.savefig(str(dest_path), dpi=300)
    plt.close(fig)

    return dest_path


def save_evaluation_report(
    checkpoint: Dict[str, Any],
    metrics: Dict[str, Any],
    output_path: Union[Path, str] = "models/evaluation_report.txt",
) -> Path:
    """
    Generates a structured, human-readable evaluation summary report text file.

    Args:
        checkpoint: Loaded model checkpoint dictionary.
        metrics: Computed metrics dictionary.
        output_path: Destination file path.

    Returns:
        Path: Resolved absolute path to the written report.
    """
    dest_path = (PROJECT_ROOT / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch = checkpoint.get("epoch", "N/A")
    ckpt_val_loss = checkpoint.get("validation_loss", "N/A")
    ckpt_val_acc = checkpoint.get("validation_accuracy", "N/A")

    if isinstance(ckpt_val_loss, float):
        ckpt_val_loss_str = f"{ckpt_val_loss:.4f}"
    else:
        ckpt_val_loss_str = str(ckpt_val_loss)

    if isinstance(ckpt_val_acc, float):
        ckpt_val_acc_str = f"{ckpt_val_acc:.4f} ({ckpt_val_acc * 100:.2f}%)"
    else:
        ckpt_val_acc_str = str(ckpt_val_acc)

    overall_acc = metrics["accuracy"]
    prec = metrics["precision"]
    rec = metrics["recall"]
    f1 = metrics["f1_score"]
    clf_report = metrics["classification_report"]

    report_content = f"""================================================================================
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 4: Baseline Image Classifier Evaluation Report
================================================================================

Timestamp               : {timestamp}
Evaluated Checkpoint    : models/best_baseline_model.pth
Checkpoint Saved Epoch  : {epoch}

--------------------------------------------------------------------------------
Checkpoint Recorded Metrics
--------------------------------------------------------------------------------
Validation Loss         : {ckpt_val_loss_str}
Validation Accuracy     : {ckpt_val_acc_str}

--------------------------------------------------------------------------------
Validation Set Evaluation Metrics
--------------------------------------------------------------------------------
Overall Accuracy        : {overall_acc:.4f} ({overall_acc * 100:.2f}%)
Weighted Precision      : {prec:.4f}
Weighted Recall         : {rec:.4f}
Weighted F1 Score       : {f1:.4f}

--------------------------------------------------------------------------------
Classification Report
--------------------------------------------------------------------------------
{clf_report}

--------------------------------------------------------------------------------
Confusion Matrix
--------------------------------------------------------------------------------
{metrics["confusion_matrix"]}

================================================================================
"""
    dest_path.write_text(report_content.strip() + "\n", encoding="utf-8")
    return dest_path


def run_evaluation(
    checkpoint_path: Union[Path, str] = "models/best_baseline_model.pth",
    report_path: Union[Path, str] = "models/evaluation_report.txt",
    curves_path: Union[Path, str] = "models/training_curves.png",
    cm_path: Union[Path, str] = "models/confusion_matrix.png",
) -> Dict[str, Any]:
    """
    Coordinates the full evaluation lifecycle:
    1. Loads checkpoint and reconstructs model and validation loader.
    2. Runs inference and computes evaluation metrics.
    3. Prints classification report to stdout.
    4. Generates and saves training curves, confusion matrix, and text report.

    Returns:
        Dict[str, Any]: Evaluation metrics dictionary.
    """
    print("=" * 65)
    print("Multimodal Edge AI System - Milestone 4 Model Evaluation")
    print("=" * 65)

    # 1. Device and Checkpoint Loading
    device = get_device()
    print(f"[INFO] Compute Device: {device}")

    checkpoint = load_checkpoint(checkpoint_path)
    print(f"[INFO] Loaded Checkpoint from: {checkpoint_path}")
    print(f"[INFO] Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"[INFO] Checkpoint Recorded Val Loss: {checkpoint.get('validation_loss', 'N/A'):.4f}")

    # 2. Reconstruct Model and Validation DataLoader
    model = load_trained_model(checkpoint=checkpoint, device=device)
    val_loader, classes = get_validation_loader(seed=42)
    print(f"[INFO] Validation Samples: {len(val_loader.dataset)} | Classes: {classes}")

    # 3. Predict on Validation Set
    y_true, y_pred, _ = evaluate_model(model=model, dataloader=val_loader, device=device)

    # 4. Compute Metrics
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, target_names=classes)

    print("\n" + "-" * 65)
    print("CLASSIFICATION REPORT")
    print("-" * 65)
    print(metrics["classification_report"])
    print("-" * 65)

    # 5. Generate Visualizations and Artifacts
    history = checkpoint.get("history", {})
    if history:
        plot_training_curves(history=history, output_path=curves_path)
    else:
        print("[WARNING] No training history found in checkpoint; skipping training curve generation.")

    plot_confusion_matrix(cm=metrics["confusion_matrix"], class_names=classes, output_path=cm_path)
    save_evaluation_report(checkpoint=checkpoint, metrics=metrics, output_path=report_path)

    return metrics


def main() -> None:
    """Main execution function."""
    run_evaluation()

    print("\nEvaluation completed successfully.\n")
    print("Saved:")
    print("training_curves.png")
    print("confusion_matrix.png")
    print("evaluation_report.txt")


if __name__ == "__main__":
    main()
