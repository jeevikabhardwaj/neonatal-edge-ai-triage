from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataset import OpenPOCUSMinorDataset
from model import create_model


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path.cwd()

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "minor"
    / "minor_train_manifest_240.csv"
)

VAL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "minor"
    / "minor_validation_manifest_240.csv"
)

TEST_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "minor_frame_manifest_clean.csv"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "minor"
    / "checkpoints"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "models"
    / "minor"
    / "reports"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "models"
    / "minor"
    / "figures"
)

HISTORY_DIR = (
    PROJECT_ROOT
    / "models"
    / "minor"
    / "history"
)

for directory in [
    CHECKPOINT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    HISTORY_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
RANDOM_SEED = 42


# ==========================================
# REPRODUCIBILITY
# ==========================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==========================================
# DEVICE
# ==========================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ==========================================
# DATASETS
# ==========================================

print("\n==========================================")
print("       MINOR MODEL TRAINING")
print("==========================================")

print("\nDevice:", DEVICE)

print("\nLoading datasets...")

train_dataset = OpenPOCUSMinorDataset(
    TRAIN_MANIFEST,
    split="train",
    image_size=IMAGE_SIZE,
    train=True,
)

val_dataset = OpenPOCUSMinorDataset(
    VAL_MANIFEST,
    split="validation",
    image_size=IMAGE_SIZE,
    train=False,
)

test_dataset = OpenPOCUSMinorDataset(
    TEST_MANIFEST,
    split="test",
    image_size=IMAGE_SIZE,
    train=False,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


print(
    f"\nTrain samples:      {len(train_dataset):,}"
)

print(
    f"Validation samples: {len(val_dataset):,}"
)

print(
    f"Test samples:       {len(test_dataset):,}"
)


# ==========================================
# CLASS WEIGHTS
# ==========================================

train_labels = train_dataset.df["label"].astype(int)

class_counts = np.bincount(
    train_labels,
    minlength=2,
)

total = class_counts.sum()

class_weights = total / (
    2.0 * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)

print("\nClass distribution:")

print(
    f"  Normal   (0): {class_counts[0]:,}"
)

print(
    f"  Abnormal (1): {class_counts[1]:,}"
)

print(
    "\nClass weights:",
    class_weights.detach().cpu().numpy()
)


# ==========================================
# MODEL
# ==========================================

model = create_model(
    num_classes=2,
    pretrained=True,
    device=DEVICE,
)


# ==========================================
# LOSS / OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_records = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                DEVICE
            )

            labels = batch["label"].to(
                DEVICE
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            predictions = (
                outputs.argmax(dim=1)
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_records.extend(
                batch["record_id"]
            )

    labels_np = np.array(all_labels)
    predictions_np = np.array(
        all_predictions
    )
    probabilities_np = np.array(
        all_probabilities
    )

    metrics = {}

    metrics["loss"] = (
        total_loss / len(loader.dataset)
    )

    metrics["accuracy"] = accuracy_score(
        labels_np,
        predictions_np
    )

    metrics["balanced_accuracy"] = (
        balanced_accuracy_score(
            labels_np,
            predictions_np
        )
    )

    metrics["precision"] = precision_score(
        labels_np,
        predictions_np,
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        labels_np,
        predictions_np,
        zero_division=0,
    )

    metrics["f1"] = f1_score(
        labels_np,
        predictions_np,
        zero_division=0,
    )

    try:
        metrics["roc_auc"] = roc_auc_score(
            labels_np,
            probabilities_np,
        )
    except ValueError:
        metrics["roc_auc"] = None

    return (
        metrics,
        labels_np,
        predictions_np,
        probabilities_np,
        all_records,
    )


# ==========================================
# TRAINING LOOP
# ==========================================

history = []

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

print("\n==========================================")
print("       TRAINING")
print("==========================================")

for epoch in range(1, EPOCHS + 1):

    model.train()

    epoch_loss = 0.0
    correct = 0
    total_samples = 0

    start_time = time.perf_counter()

    for batch in train_loader:

        images = batch["image"].to(
            DEVICE
        )

        labels = batch["label"].to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        epoch_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total_samples += images.size(0)

    train_loss = (
        epoch_loss /
        total_samples
    )

    train_accuracy = (
        correct /
        total_samples
    )

    (
        val_metrics,
        _,
        _,
        _,
        _,
    ) = evaluate(
        model,
        val_loader
    )

    scheduler.step(
        val_metrics["loss"]
    )

    epoch_time = (
        time.perf_counter()
        - start_time
    )

    current_lr = optimizer.param_groups[0]["lr"]

    history_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": (
            val_metrics["balanced_accuracy"]
        ),
        "val_precision": (
            val_metrics["precision"]
        ),
        "val_recall": (
            val_metrics["recall"]
        ),
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "learning_rate": current_lr,
        "epoch_seconds": epoch_time,
    }

    history.append(history_row)

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Train Loss: {train_loss:.4f}"
    )

    print(
        f"  Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"  Val Loss: "
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"  Val Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"  Val Balanced Accuracy: "
        f"{val_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Val Recall: "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"  Val F1: "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"  Val ROC-AUC: "
        f"{val_metrics['roc_auc']}"
    )

    print(
        f"  Learning Rate: "
        f"{current_lr:.2e}"
    )

    print(
        f"  Time: "
        f"{epoch_time:.1f}s"
    )

    # --------------------------------------
    # Save best checkpoint
    # --------------------------------------

    if val_metrics["loss"] < best_val_loss:

        best_val_loss = (
            val_metrics["loss"]
        )

        best_epoch = epoch
        epochs_without_improvement = 0

        checkpoint_path = (
            CHECKPOINT_DIR
            / "minor_baseline_best.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "val_loss": (
                    val_metrics["loss"]
                ),
                "val_metrics": val_metrics,
                "class_weights": (
                    class_weights.cpu()
                ),
                "config": {
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": (
                        LEARNING_RATE
                    ),
                    "weight_decay": (
                        WEIGHT_DECAY
                    ),
                    "num_classes": 2,
                    "random_seed": (
                        RANDOM_SEED
                    ),
                },
            },
            checkpoint_path,
        )

        print(
            "  ✓ Best checkpoint saved"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  No improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ==========================================
# SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(history)

history_path = (
    HISTORY_DIR
    / "minor_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
)


# ==========================================
# LOAD BEST CHECKPOINT
# ==========================================

checkpoint_path = (
    CHECKPOINT_DIR
    / "minor_baseline_best.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\n==========================================")
print("       FINAL TEST EVALUATION")
print("==========================================")

(
    test_metrics,
    test_labels,
    test_predictions,
    test_probabilities,
    test_records,
) = evaluate(
    model,
    test_loader
)

print(
    f"\nBest epoch: {best_epoch}"
)

for name, value in test_metrics.items():

    if value is None:
        print(f"{name}: N/A")
    else:
        print(
            f"{name}: {value:.4f}"
        )


# ==========================================
# RECORD-LEVEL AGGREGATION
# ==========================================

print("\n===== RECORD-LEVEL EVALUATION =====")

record_df = pd.DataFrame({
    "record_id": test_records,
    "label": test_labels,
    "probability": test_probabilities,
})

record_predictions = (
    record_df
    .groupby("record_id")
    .agg(
        label=("label", "first"),
        probability=("probability", "mean"),
    )
    .reset_index()
)

record_predictions["prediction"] = (
    record_predictions["probability"] >= 0.5
).astype(int)

record_labels = (
    record_predictions["label"].values
)

record_preds = (
    record_predictions["prediction"].values
)

record_probs = (
    record_predictions["probability"].values
)

record_metrics = {
    "records": len(record_predictions),
    "accuracy": accuracy_score(
        record_labels,
        record_preds,
    ),
    "balanced_accuracy": (
        balanced_accuracy_score(
            record_labels,
            record_preds,
        )
    ),
    "precision": precision_score(
        record_labels,
        record_preds,
        zero_division=0,
    ),
    "recall": recall_score(
        record_labels,
        record_preds,
        zero_division=0,
    ),
    "f1": f1_score(
        record_labels,
        record_preds,
        zero_division=0,
    ),
}

try:
    record_metrics["roc_auc"] = (
        roc_auc_score(
            record_labels,
            record_probs,
        )
    )
except ValueError:
    record_metrics["roc_auc"] = None

for name, value in record_metrics.items():

    if value is None:
        print(f"{name}: N/A")

    elif name == "records":
        print(f"{name}: {value}")

    else:
        print(
            f"{name}: {value:.4f}"
        )


# ==========================================
# SAVE REPORT
# ==========================================

report = {
    "task": "OpenPOCUS LUS Normal vs Abnormal",
    "classes": {
        "0": "Normal",
        "1": "Abnormal",
    },
    "device": str(DEVICE),
    "train_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "best_epoch": best_epoch,
    "image_level_test_metrics": test_metrics,
    "record_level_test_metrics": record_metrics,
    "config": {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "random_seed": RANDOM_SEED,
    },
}

report_path = (
    REPORT_DIR
    / "minor_training_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ==========================================
# SAVE RECORD PREDICTIONS
# ==========================================

record_predictions_path = (
    REPORT_DIR
    / "minor_test_record_predictions.csv"
)

record_predictions.to_csv(
    record_predictions_path,
    index=False,
)


print("\n==========================================")
print("       TRAINING COMPLETE")
print("==========================================")

print(
    f"\nBest checkpoint:\n"
    f"{checkpoint_path}"
)

print(
    f"\nHistory:\n"
    f"{history_path}"
)

print(
    f"\nReport:\n"
    f"{report_path}"
)

print(
    f"\nRecord predictions:\n"
    f"{record_predictions_path}"
)

print("\n==========================================\n")
