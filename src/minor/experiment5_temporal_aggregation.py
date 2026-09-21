import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from dataset import OpenPOCUSMinorDataset
from model import create_model

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "minor" / "minor_frame_manifest_clean.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "minor" / "checkpoints" / "minor_baseline_best.pth"
REPORTS_DIR = PROJECT_ROOT / "models" / "minor" / "reports"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
NUM_WORKERS = 0
IMAGE_SIZE = 224

def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["records"] = len(y_true)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["normal_recall"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["abnormal_recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    metrics["correctly_classified"] = int(tn + tp)
    metrics["incorrectly_classified"] = int(fp + fn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    
    return metrics

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading test dataset...")
    test_dataset = OpenPOCUSMinorDataset(
        manifest_path=MANIFEST_PATH,
        split="test",
        image_size=IMAGE_SIZE,
        train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = create_model(num_classes=2, pretrained=False, device=DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_record_ids = []
    all_labels = []
    all_probs = []
    
    print("Extracting frame-level predictions (this may take a few minutes)...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].numpy()
            record_ids = batch["record_id"]
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_record_ids.extend(record_ids)
            all_labels.extend(labels)
            all_probs.extend(probs)
            
    df = pd.DataFrame({
        "record_id": all_record_ids,
        "label": all_labels,
        "probability": all_probs
    })
    
    print("Computing record-level aggregations...")
    # Baseline: Mean Probability
    baseline_df = df.groupby("record_id").agg(
        label=("label", "first"),
        mean_probability=("probability", "mean"),
        num_frames=("probability", "count")
    ).reset_index()
    baseline_df["baseline_prediction"] = (baseline_df["mean_probability"] >= 0.5).astype(int)
    
    # Median Probability
    median_df = df.groupby("record_id").agg(
        median_probability=("probability", "median")
    ).reset_index()
    
    # Majority Vote
    df["frame_prediction"] = (df["probability"] >= 0.5).astype(int)
    majority_df = df.groupby("record_id").agg(
        majority_fraction=("frame_prediction", "mean")
    ).reset_index()
    # If exactly 0.5 fraction, default to 0 (Normal) to be conservative, or 1. We will use >= 0.5 as Abnormal.
    majority_df["majority_prediction"] = (majority_df["majority_fraction"] >= 0.5).astype(int)
    
    # Merge all
    record_df = baseline_df.merge(median_df, on="record_id").merge(majority_df, on="record_id")
    record_df["median_prediction"] = (record_df["median_probability"] >= 0.5).astype(int)
    
    # Save predictions
    predictions_path = REPORTS_DIR / "experiment5_temporal_aggregation_predictions.csv"
    record_df.to_csv(predictions_path, index=False)
    
    # Compute metrics
    y_true = record_df["label"].values
    
    baseline_metrics = compute_metrics(y_true, record_df["baseline_prediction"].values, record_df["mean_probability"].values)
    median_metrics = compute_metrics(y_true, record_df["median_prediction"].values, record_df["median_probability"].values)
    majority_metrics = compute_metrics(y_true, record_df["majority_prediction"].values, record_df["majority_fraction"].values)
    
    # Identify changed records
    changed_records = []
    for idx, row in record_df.iterrows():
        baseline_pred = row["baseline_prediction"]
        median_pred = row["median_prediction"]
        maj_pred = row["majority_prediction"]
        
        if median_pred != baseline_pred or maj_pred != baseline_pred:
            changed_records.append({
                "record_id": row["record_id"],
                "label": int(row["label"]),
                "num_frames": int(row["num_frames"]),
                "baseline_prediction": int(baseline_pred),
                "mean_probability": float(row["mean_probability"]),
                "median_prediction": int(median_pred),
                "median_probability": float(row["median_probability"]),
                "majority_prediction": int(maj_pred),
                "majority_fraction": float(row["majority_fraction"])
            })
            
    summary = {
        "experiment": "Record-level Temporal Aggregation (Experiment 5)",
        "dataset_split": "test",
        "records_evaluated": len(y_true),
        "total_frames_evaluated": len(df),
        "aggregation_strategies": {
            "baseline_mean_probability": baseline_metrics,
            "median_probability": median_metrics,
            "majority_vote": majority_metrics
        },
        "changed_records_vs_baseline": changed_records,
        "methodological_note": "The existing record-level baseline already uses Mean Probability aggregation. Median Probability and Majority-Vote frame aggregations were compared against it. All calculations are performed within independent records to prevent patient-level leakage."
    }
    
    summary_path = REPORTS_DIR / "experiment5_temporal_aggregation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Evaluated {len(y_true)} records across {len(df)} frames.")
    print("Baseline Mean Probability Accuracy:", baseline_metrics["accuracy"])
    print("Median Probability Accuracy:", median_metrics["accuracy"])
    print("Majority Vote Accuracy:", majority_metrics["accuracy"])
    print(f"Found {len(changed_records)} records with differing predictions.")

if __name__ == "__main__":
    main()
