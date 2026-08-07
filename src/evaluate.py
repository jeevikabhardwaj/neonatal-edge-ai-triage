"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 4: Comprehensive Medical AI Model Evaluation & Artifact Pipeline

This module implements a research-grade, reproducible evaluation pipeline for the
trained MobileNetV3-Small neonatal lung ultrasound triage model. It loads the best
model checkpoint, reconstructs the deterministic validation split, computes core and
clinical performance metrics (Accuracy, Balanced Accuracy, Sensitivity/Recall, Specificity,
F1-score, Confusion Matrix, and ROC-AUC), performs misclassification analysis, exports
structured JSON metrics, and generates publication-quality visualization figures.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset, random_split

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


# =====================================================================
# Checkpoint & Model Loading
# =====================================================================
def load_checkpoint(
    checkpoint_path: Union[Path, str] = "models/checkpoints/best_baseline_model.pth",
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

    # Fallback check to legacy path if default path does not exist
    if not resolved_path.exists() or not resolved_path.is_file():
        legacy_path = (PROJECT_ROOT / "models" / "best_baseline_model.pth").resolve()
        if legacy_path.exists() and legacy_path.is_file():
            resolved_path = legacy_path

    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint file not found at: '{resolved_path}'\n"
            "Please train the baseline model first by running:\n"
            "    python src/train.py"
        )

    checkpoint: Dict[str, Any] = torch.load(str(resolved_path), map_location="cpu")
    return checkpoint


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

    classes = checkpoint.get("classes", ["high_risk", "moderate_risk", "normal"])
    num_classes = len(classes)

    model = build_baseline_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# =====================================================================
# Dataset & Validation Loader Rebuilding
# =====================================================================
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


def get_validation_sample_paths(val_dataset: Dataset) -> List[str]:
    """
    Extracts underlying image file paths for all samples in the validation dataset.

    Args:
        val_dataset: PyTorch Dataset or Subset instance.

    Returns:
        List[str]: List of image filesystem paths.
    """
    if isinstance(val_dataset, Subset):
        base_ds = val_dataset.dataset
        if hasattr(base_ds, "samples"):
            return [str(base_ds.samples[i][0]) for i in val_dataset.indices]
        elif hasattr(base_ds, "imgs"):
            return [str(base_ds.imgs[i][0]) for i in val_dataset.indices]
    elif hasattr(val_dataset, "samples"):
        return [str(s[0]) for s in val_dataset.samples]
    return []


# =====================================================================
# Model Inference
# =====================================================================
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
    y_probs = np.vstack(all_probs) if all_probs else np.empty((0, 0))

    return y_true, y_pred, y_probs


# =====================================================================
# Metrics Computation (Core, Clinical & ROC-AUC)
# =====================================================================
def compute_clinical_metrics(
    cm: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Computes per-class clinical metrics (Sensitivity / Recall, Specificity, Support)
    and highlights the high-risk triage class.

    Args:
        cm: 2D numpy confusion matrix of shape (C, C).
        class_names: List of class label strings.

    Returns:
        Dict[str, Any]: Dictionary containing per_class_sensitivity,
                        per_class_specificity, per_class_support, and high_risk_sensitivity.
    """
    num_classes = len(class_names)
    total_samples = int(np.sum(cm))

    per_class_sensitivity: Dict[str, float] = {}
    per_class_specificity: Dict[str, float] = {}
    per_class_support: Dict[str, int] = {}

    for i in range(num_classes):
        class_name = class_names[i]
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - (tp + fp + fn))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        support = int(tp + fn)

        per_class_sensitivity[class_name] = round(sensitivity, 4)
        per_class_specificity[class_name] = round(specificity, 4)
        per_class_support[class_name] = support

    # High Risk Sensitivity
    high_risk_key = "high_risk"
    if high_risk_key in per_class_sensitivity:
        high_risk_sensitivity = per_class_sensitivity[high_risk_key]
    elif len(class_names) > 0:
        high_risk_sensitivity = per_class_sensitivity[class_names[0]]
    else:
        high_risk_sensitivity = 0.0

    return {
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "per_class_support": per_class_support,
        "high_risk_sensitivity": high_risk_sensitivity,
    }


def compute_roc_auc_safe(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
) -> Tuple[Optional[float], str]:
    """
    Safely computes multiclass One-vs-Rest ROC-AUC score with graceful error handling
    for small datasets or missing classes.

    Args:
        y_true: Ground truth labels (N,).
        y_probs: Predicted class probabilities (N, C).
        class_names: List of class strings.

    Returns:
        Tuple[Optional[float], str]: (roc_auc_score or None, status_message)
    """
    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        return None, "ROC-AUC not computed (less than 2 distinct classes present in validation set)."

    if y_probs.ndim != 2 or y_probs.shape[1] < 2:
        return None, "ROC-AUC not computed (invalid probability matrix dimensions)."

    try:
        if len(unique_classes) == len(class_names):
            score = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted"))
            return round(score, 4), f"Weighted OvR ROC-AUC: {score:.4f}"
        else:
            # Subset of classes present in validation split
            score = float(
                roc_auc_score(
                    y_true,
                    y_probs[:, unique_classes],
                    multi_class="ovr",
                    average="weighted",
                    labels=unique_classes,
                )
            )
            return round(score, 4), f"Weighted OvR ROC-AUC (present classes): {score:.4f}"
    except Exception as exc:
        return None, f"ROC-AUC not computed (insufficient validation samples: {exc})."


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification, clinical, and statistical performance metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_probs: Softmax probability distribution matrix.
        target_names: List of class label strings.

    Returns:
        Dict[str, Any]: Complete metrics dictionary including accuracy, balanced accuracy,
                        precision, recall, f1 score, support, clinical metrics, ROC-AUC,
                        and classification reports.
    """
    if target_names is None:
        target_names = [f"class_{i}" for i in range(max(len(np.unique(y_true)), 1))]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    support = int(len(y_true))

    clf_report_str = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    clf_report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    # Clinical Metrics (Sensitivity / Recall & Specificity)
    clinical_metrics = compute_clinical_metrics(cm=cm, class_names=target_names)

    # ROC-AUC calculation (safe)
    if y_probs is not None and y_probs.size > 0:
        roc_auc, roc_auc_msg = compute_roc_auc_safe(
            y_true=y_true,
            y_probs=y_probs,
            class_names=target_names,
        )
    else:
        roc_auc, roc_auc_msg = None, "ROC-AUC not computed (no probability distributions provided)."

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "support": support,
        "classification_report": clf_report_str,
        "classification_report_dict": clf_report_dict,
        "confusion_matrix": cm,
        "per_class_sensitivity": clinical_metrics["per_class_sensitivity"],
        "per_class_specificity": clinical_metrics["per_class_specificity"],
        "per_class_support": clinical_metrics["per_class_support"],
        "high_risk_sensitivity": clinical_metrics["high_risk_sensitivity"],
        "roc_auc": roc_auc,
        "roc_auc_message": roc_auc_msg,
    }


# =====================================================================
# Misclassification Analysis Log
# =====================================================================
def save_misclassification_log(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    sample_paths: List[str],
    class_names: List[str],
    checkpoint_path: Union[Path, str],
    output_path: Union[Path, str] = "models/reports/misclassification_log.txt",
) -> Path:
    """
    Identifies all incorrectly predicted validation samples and generates a detailed
    audit log documenting sample path, true class, predicted class, and probability distribution.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        y_probs: Predicted class probability matrix.
        sample_paths: List of sample image file paths.
        class_names: List of class strings.
        checkpoint_path: Evaluated model checkpoint path.
        output_path: Destination filepath.

    Returns:
        Path: Resolved absolute path to the generated log.
    """
    dest_path = (PROJECT_ROOT / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_samples = len(y_true)

    misclassified_indices = [i for i in range(total_samples) if y_true[i] != y_pred[i]]
    total_errors = len(misclassified_indices)

    lines: List[str] = [
        "=" * 80,
        "Multimodal Edge AI System for Neonatal Respiratory Triage",
        "Milestone 4: Misclassification Analysis Log",
        "=" * 80,
        f"Timestamp                : {timestamp}",
        f"Evaluated Checkpoint     : {checkpoint_path}",
        f"Validation Samples Total : {total_samples}",
        f"Misclassified Count      : {total_errors} ({total_errors / total_samples * 100:.2f}% error rate)" if total_samples > 0 else f"Misclassified Count      : 0",
        "-" * 80,
    ]

    if total_errors == 0:
        lines.append("\nNo misclassified samples. Perfect classification on validation set.\n")
    else:
        lines.append("\nMISCLASSIFIED SAMPLES AUDIT TRAIL:\n")
        for count, idx in enumerate(misclassified_indices, start=1):
            true_cls = class_names[y_true[idx]] if y_true[idx] < len(class_names) else str(y_true[idx])
            pred_cls = class_names[y_pred[idx]] if y_pred[idx] < len(class_names) else str(y_pred[idx])
            pred_conf = float(y_probs[idx, y_pred[idx]]) if idx < len(y_probs) else 0.0

            raw_path = sample_paths[idx] if idx < len(sample_paths) else f"Sample_{idx}"
            try:
                rel_path = str(Path(raw_path).relative_to(PROJECT_ROOT))
            except (ValueError, Exception):
                rel_path = str(raw_path)

            lines.append(f"Sample #{count}:")
            lines.append(f"  Path                 : {rel_path}")
            lines.append(f"  True Class           : {true_cls}")
            lines.append(f"  Predicted Class      : {pred_cls}")
            lines.append(f"  Predicted Confidence : {pred_conf:.4f} ({pred_conf * 100:.2f}%)")
            lines.append("  Probability Distribution:")

            if idx < len(y_probs):
                for c_idx, c_name in enumerate(class_names):
                    prob_val = float(y_probs[idx, c_idx])
                    lines.append(f"    - {c_name:<18} : {prob_val:.4f} ({prob_val * 100:.2f}%)")
            lines.append("-" * 80)

    lines.append("=" * 80 + "\n")
    dest_path.write_text("\n".join(lines), encoding="utf-8")
    return dest_path


# =====================================================================
# Structured JSON Metrics Export
# =====================================================================
def save_metrics_json(
    metrics: Dict[str, Any],
    checkpoint_path: Union[Path, str],
    model_version: str = "baseline_mobilenetv3_v1",
    evaluation_runtime: float = 0.0,
    output_path: Union[Path, str] = "models/reports/evaluation_metrics.json",
) -> Path:
    """
    Exports comprehensive evaluation metrics to a standardized JSON artifact
    for downstream dashboard integration and automated reporting.

    Args:
        metrics: Computed metrics dictionary.
        checkpoint_path: Evaluated model checkpoint path.
        model_version: Architecture model version identifier.
        evaluation_runtime: Total evaluation duration in seconds.
        output_path: Destination JSON filepath.

    Returns:
        Path: Resolved absolute path to the generated JSON file.
    """
    dest_path = (PROJECT_ROOT / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cm = metrics["confusion_matrix"]
    cm_list = cm.tolist() if isinstance(cm, np.ndarray) else cm

    json_payload: Dict[str, Any] = {
        "evaluation_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": model_version,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_runtime_seconds": round(evaluation_runtime, 4),
        "support": metrics["support"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1_score"],
        "high_risk_sensitivity": metrics["high_risk_sensitivity"],
        "per_class_sensitivity": metrics["per_class_sensitivity"],
        "per_class_specificity": metrics["per_class_specificity"],
        "per_class_support": metrics["per_class_support"],
        "roc_auc": metrics["roc_auc"],
        "roc_auc_status": metrics["roc_auc_message"],
        "confusion_matrix": cm_list,
        "classification_report": metrics.get("classification_report_dict", {}),
    }

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    return dest_path


# =====================================================================
# Visualizations & Report Writers
# =====================================================================
def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Union[Path, str] = "models/figures/training_curves.png",
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
    output_path: Union[Path, str] = "models/figures/confusion_matrix.png",
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
    checkpoint_path: Union[Path, str] = "models/checkpoints/best_baseline_model.pth",
    output_path: Union[Path, str] = "models/reports/evaluation_report.txt",
) -> Path:
    """
    Generates a structured, human-readable evaluation summary report text file.

    Args:
        checkpoint: Loaded model checkpoint dictionary.
        metrics: Computed metrics dictionary.
        checkpoint_path: Path string to the evaluated checkpoint.
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

    ckpt_val_loss_str = f"{ckpt_val_loss:.4f}" if isinstance(ckpt_val_loss, float) else str(ckpt_val_loss)
    ckpt_val_acc_str = f"{ckpt_val_acc:.4f} ({ckpt_val_acc * 100:.2f}%)" if isinstance(ckpt_val_acc, float) else str(ckpt_val_acc)

    overall_acc = metrics["accuracy"]
    bal_acc = metrics["balanced_accuracy"]
    prec = metrics["precision"]
    rec = metrics["recall"]
    f1 = metrics["f1_score"]
    high_risk_sens = metrics["high_risk_sensitivity"]
    roc_auc_status = metrics.get("roc_auc_message", "N/A")

    # Format Per-Class Clinical Table
    clinical_table_lines = [
        f"{'Class':<18} {'Sensitivity (Recall)':<24} {'Specificity':<16} {'Support':<8}",
        "-" * 70,
    ]
    for c_name, sens in metrics["per_class_sensitivity"].items():
        spec = metrics["per_class_specificity"].get(c_name, 0.0)
        sup = metrics["per_class_support"].get(c_name, 0)
        clinical_table_lines.append(
            f"{c_name:<18} {sens:<24.4f} {spec:<16.4f} {sup:<8d}"
        )
    clinical_table_str = "\n".join(clinical_table_lines)

    report_content = f"""================================================================================
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 4: Baseline Image Classifier Comprehensive Evaluation Report
================================================================================

Timestamp               : {timestamp}
Evaluated Checkpoint    : {checkpoint_path}
Checkpoint Saved Epoch  : {epoch}

--------------------------------------------------------------------------------
Checkpoint Recorded Metrics
--------------------------------------------------------------------------------
Validation Loss         : {ckpt_val_loss_str}
Validation Accuracy     : {ckpt_val_acc_str}

--------------------------------------------------------------------------------
Validation Set Performance Metrics
--------------------------------------------------------------------------------
Overall Accuracy        : {overall_acc:.4f} ({overall_acc * 100:.2f}%)
Balanced Accuracy       : {bal_acc:.4f} ({bal_acc * 100:.2f}%)
Weighted Precision      : {prec:.4f}
Weighted Recall         : {rec:.4f}
Weighted F1 Score       : {f1:.4f}
High Risk Sensitivity   : {high_risk_sens:.4f} ({high_risk_sens * 100:.2f}%)
ROC-AUC Status          : {roc_auc_status}

--------------------------------------------------------------------------------
Per-Class Clinical Metrics (Sensitivity & Specificity)
--------------------------------------------------------------------------------
{clinical_table_str}

--------------------------------------------------------------------------------
Classification Report
--------------------------------------------------------------------------------
{metrics["classification_report"]}

--------------------------------------------------------------------------------
Confusion Matrix
--------------------------------------------------------------------------------
{metrics["confusion_matrix"]}

================================================================================
"""
    dest_path.write_text(report_content.strip() + "\n", encoding="utf-8")
    return dest_path


# =====================================================================
# Main Evaluation Pipeline Orchestrator
# =====================================================================
def run_evaluation(
    checkpoint_path: Union[Path, str] = "models/checkpoints/best_baseline_model.pth",
    report_path: Union[Path, str] = "models/reports/evaluation_report.txt",
    metrics_json_path: Union[Path, str] = "models/reports/evaluation_metrics.json",
    misclass_log_path: Union[Path, str] = "models/reports/misclassification_log.txt",
    curves_path: Union[Path, str] = "models/figures/training_curves.png",
    cm_path: Union[Path, str] = "models/figures/confusion_matrix.png",
) -> Dict[str, Any]:
    """
    Coordinates the comprehensive evaluation lifecycle:
    1. Loads trained checkpoint and metadata.
    2. Reconstructs MobileNetV3 model architecture and deterministic validation DataLoader.
    3. Runs model inference and extracts probability distribution matrices.
    4. Computes core metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1, Support).
    5. Computes clinical metrics (Per-class Sensitivity, Specificity, and High Risk Sensitivity).
    6. Safely attempts multiclass One-vs-Rest ROC-AUC calculation.
    7. Performs misclassification audit and generates misclassification_log.txt.
    8. Exports machine-readable structured evaluation_metrics.json.
    9. Generates publication-ready figures (training_curves.png, confusion_matrix.png).
    10. Writes comprehensive human-readable evaluation_report.txt.
    11. Prints professional summary with execution runtime.

    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics dictionary.
    """
    start_time = time.time()

    print("=" * 65)
    print("Multimodal Edge AI System - Milestone 4 Model Evaluation")
    print("=" * 65)

    # 1. Device and Checkpoint Loading
    device = get_device()
    print(f"[INFO] Compute Device: {device}")

    checkpoint = load_checkpoint(checkpoint_path)
    rel_ckpt_str = str(Path(checkpoint_path).relative_to(PROJECT_ROOT)) if Path(checkpoint_path).is_absolute() and Path(checkpoint_path).is_relative_to(PROJECT_ROOT) else str(checkpoint_path)
    print(f"[INFO] Loaded Checkpoint from: {rel_ckpt_str}")
    print(f"[INFO] Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"[INFO] Checkpoint Recorded Val Loss: {checkpoint.get('validation_loss', 'N/A'):.4f}")

    # 2. Reconstruct Model and Validation DataLoader
    model = load_trained_model(checkpoint=checkpoint, device=device)
    val_loader, classes = get_validation_loader(seed=42)
    sample_paths = get_validation_sample_paths(val_loader.dataset)
    print(f"[INFO] Validation Samples: {len(val_loader.dataset)} | Classes: {classes}")

    # 3. Predict on Validation Set
    y_true, y_pred, y_probs = evaluate_model(model=model, dataloader=val_loader, device=device)

    # 4. Compute Comprehensive Metrics
    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        target_names=classes,
    )

    # 5. Display Classification Report
    print("\n" + "-" * 65)
    print("CLASSIFICATION REPORT")
    print("-" * 65)
    print(metrics["classification_report"])
    print("-" * 65)

    # 6. Display Clinical Metrics Table
    print("\n" + "-" * 65)
    print("CLINICAL PERFORMANCE METRICS")
    print("-" * 65)
    print(f"{'Class':<18} {'Sensitivity (Recall)':<24} {'Specificity':<16} {'Support':<8}")
    print("-" * 65)
    for c_name, sens in metrics["per_class_sensitivity"].items():
        spec = metrics["per_class_specificity"].get(c_name, 0.0)
        sup = metrics["per_class_support"].get(c_name, 0)
        print(f"{c_name:<18} {sens:<24.4f} {spec:<16.4f} {sup:<8d}")
    print("-" * 65)
    print(f"High Risk Sensitivity : {metrics['high_risk_sensitivity']:.4f} ({metrics['high_risk_sensitivity'] * 100:.2f}%)")
    print(f"ROC-AUC               : {metrics['roc_auc_message']}")
    print("-" * 65)

    # 7. Generate Visualizations and Artifacts
    history = checkpoint.get("history", {})
    if history:
        plot_training_curves(history=history, output_path=curves_path)
    else:
        print("[WARNING] No training history found in checkpoint; skipping training curve generation.")

    plot_confusion_matrix(cm=metrics["confusion_matrix"], class_names=classes, output_path=cm_path)
    save_evaluation_report(
        checkpoint=checkpoint,
        metrics=metrics,
        checkpoint_path=rel_ckpt_str,
        output_path=report_path,
    )

    # 8. Misclassification Analysis
    save_misclassification_log(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        sample_paths=sample_paths,
        class_names=classes,
        checkpoint_path=rel_ckpt_str,
        output_path=misclass_log_path,
    )

    # 9. JSON Metrics Export
    runtime = time.time() - start_time
    model_version = checkpoint.get("model_version", "baseline_mobilenetv3_v1")
    save_metrics_json(
        metrics=metrics,
        checkpoint_path=rel_ckpt_str,
        model_version=model_version,
        evaluation_runtime=runtime,
        output_path=metrics_json_path,
    )

    # 10. Console Summary
    rel_curves = str(Path(curves_path).relative_to(PROJECT_ROOT)) if Path(curves_path).is_absolute() and Path(curves_path).is_relative_to(PROJECT_ROOT) else str(curves_path)
    rel_cm = str(Path(cm_path).relative_to(PROJECT_ROOT)) if Path(cm_path).is_absolute() and Path(cm_path).is_relative_to(PROJECT_ROOT) else str(cm_path)
    rel_rep = str(Path(report_path).relative_to(PROJECT_ROOT)) if Path(report_path).is_absolute() and Path(report_path).is_relative_to(PROJECT_ROOT) else str(report_path)
    rel_json = str(Path(metrics_json_path).relative_to(PROJECT_ROOT)) if Path(metrics_json_path).is_absolute() and Path(metrics_json_path).is_relative_to(PROJECT_ROOT) else str(metrics_json_path)
    rel_misclass = str(Path(misclass_log_path).relative_to(PROJECT_ROOT)) if Path(misclass_log_path).is_absolute() and Path(misclass_log_path).is_relative_to(PROJECT_ROOT) else str(misclass_log_path)

    print("\n" + "=" * 65)
    print("Evaluation Summary")
    print("=" * 65)
    print(f"Device                 : {device}")
    print(f"Validation Samples     : {metrics['support']}")
    print(f"Overall Accuracy       : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Balanced Accuracy      : {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy'] * 100:.2f}%)")
    print(f"High Risk Sensitivity  : {metrics['high_risk_sensitivity']:.4f} ({metrics['high_risk_sensitivity'] * 100:.2f}%)")
    print(f"ROC-AUC                : {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'Not computed (insufficient validation samples)'}")
    print(f"Evaluation Time        : {runtime:.2f} seconds")
    print(f"Checkpoint Location    : {rel_ckpt_str}")
    print("\nGenerated Artifacts:")
    print(f"  - Training curves       : {rel_curves}")
    print(f"  - Confusion matrix      : {rel_cm}")
    print(f"  - Evaluation report     : {rel_rep}")
    print(f"  - Metrics JSON          : {rel_json}")
    print(f"  - Misclassification log : {rel_misclass}")
    print("=" * 65)

    return metrics


def main() -> None:
    """Main execution function."""
    run_evaluation()
    print("\n[SUCCESS] Model evaluation pipeline completed successfully.\n")


if __name__ == "__main__":
    main()
