import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADULT_CSV = PROJECT_ROOT / "models/minor/reports/minor_test_record_predictions.csv"
PED_CSV = PROJECT_ROOT / "models/minor/reports/pediatric_zero_shot_case_predictions.csv"
REPORTS_DIR = PROJECT_ROOT / "models/minor/reports"

def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    brier = brier_score_loss(y_true, y_prob)
    
    # ECE
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    
    # Metrics
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    correct = (y_true == y_pred)
    confidence = np.maximum(y_prob, 1 - np.array(y_prob))
    
    mean_conf = np.mean(confidence)
    mean_conf_correct = np.mean(confidence[correct]) if np.sum(correct) > 0 else 0.0
    mean_conf_incorrect = np.mean(confidence[~correct]) if np.sum(~correct) > 0 else 0.0
    
    return {
        "brier_score": float(brier),
        "ece": float(ece),
        "mean_confidence": float(mean_conf),
        "mean_conf_correct": float(mean_conf_correct),
        "mean_conf_incorrect": float(mean_conf_incorrect),
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_total": bin_total[nonzero]
    }

def entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

def plot_reliability_diagram(metrics_dict, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.plot(metrics_dict["prob_pred"], metrics_dict["prob_true"], "s-", label="Model")
    plt.xlabel("Mean predicted probability (Abnormal)")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Adult Data
    adult_df = pd.read_csv(ADULT_CSV)
    adult_y_true = adult_df['label'].values
    adult_y_prob = adult_df['probability'].values
    adult_y_pred = adult_df['prediction'].values
    
    adult_metrics = compute_calibration_metrics(adult_y_true, adult_y_prob, n_bins=10)
    
    adult_df['confidence'] = np.maximum(adult_y_prob, 1 - adult_y_prob)
    adult_df['entropy'] = entropy(adult_y_prob)
    adult_df['correct'] = (adult_y_true == adult_y_pred)
    adult_df.to_csv(REPORTS_DIR / "calibration_adult.csv", index=False)
    
    plot_reliability_diagram(adult_metrics, "Reliability Diagram - Adult In-Domain", REPORTS_DIR / "calibration_reliability_adult.png")
    
    # 2. Pediatric Data
    ped_df = pd.read_csv(PED_CSV)
    
    ped_y_true = []
    ped_y_prob = []
    ped_y_pred = []
    
    for idx, row in ped_df.iterrows():
        pathology = row['pathology']
        if pathology == 'Normal':
            gt = 0
        elif pathology in ['Pneumonia', 'Pneumothorax']:
            gt = 1
        else:
            continue
            
        ped_y_true.append(gt)
        ped_y_prob.append(row['mean_abnormal_probability'])
        pred = 1 if row['majority_prediction'] == 'Abnormal' else 0
        ped_y_pred.append(pred)
        
    ped_y_true = np.array(ped_y_true)
    ped_y_prob = np.array(ped_y_prob)
    ped_y_pred = np.array(ped_y_pred)
    
    ped_metrics = compute_calibration_metrics(ped_y_true, ped_y_prob, n_bins=10)
    
    # We will compute confidence and entropy for ALL pediatric cases to save in CSV
    all_ped_probs = ped_df['mean_abnormal_probability'].values
    ped_df['confidence'] = np.maximum(all_ped_probs, 1 - all_ped_probs)
    ped_df['entropy'] = entropy(all_ped_probs)
    
    ped_df.to_csv(REPORTS_DIR / "calibration_pediatric.csv", index=False)
    
    plot_reliability_diagram(ped_metrics, "Reliability Diagram - Pediatric External", REPORTS_DIR / "calibration_reliability_pediatric.png")
    
    # 3. Confidence Comparison Plot
    plt.figure(figsize=(10, 6))
    
    adult_corr = adult_df[adult_df['correct']]['confidence'].values
    adult_incorr = adult_df[~adult_df['correct']]['confidence'].values
    
    ped_corr_list = []
    ped_incorr_list = []
    
    for gt, prob, pred in zip(ped_y_true, ped_y_prob, ped_y_pred):
        conf = max(prob, 1-prob)
        if gt == pred:
            ped_corr_list.append(conf)
        else:
            ped_incorr_list.append(conf)
            
    # For plotting, if a list is empty we just pass an empty list
    data = [adult_corr, adult_incorr, ped_corr_list, ped_incorr_list]
    labels = ['Adult\nCorrect', 'Adult\nIncorrect', 'Pediatric\nCorrect', 'Pediatric\nIncorrect']
    
    plt.boxplot(data, labels=labels)
    plt.ylabel('Confidence')
    plt.title('Confidence Comparison: Adult In-Domain vs Pediatric External')
    plt.grid(True, axis='y')
    plt.savefig(REPORTS_DIR / "confidence_comparison.png", bbox_inches='tight')
    plt.close()
    
    # Summary JSON
    summary = {
        "adult_in_domain": {
            "cases": len(adult_y_true),
            "ece": adult_metrics["ece"],
            "brier_score": adult_metrics["brier_score"],
            "mean_confidence": adult_metrics["mean_confidence"],
            "mean_confidence_correct": adult_metrics["mean_conf_correct"],
            "mean_confidence_incorrect": adult_metrics["mean_conf_incorrect"],
            "mean_entropy": float(adult_df['entropy'].mean())
        },
        "pediatric_external": {
            "cases_evaluated_for_metrics": len(ped_y_true),
            "ece": ped_metrics["ece"],
            "brier_score": ped_metrics["brier_score"],
            "mean_confidence": ped_metrics["mean_confidence"],
            "mean_confidence_correct": ped_metrics["mean_conf_correct"],
            "mean_confidence_incorrect": ped_metrics["mean_conf_incorrect"],
            "mean_entropy": float(ped_df['entropy'].mean())
        },
        "uncertainty_method": "Confidence and Entropy analysis (MC Dropout not implemented to maintain strictly frozen evaluation pipeline on both datasets without retraining or changing inference scripts)"
    }
    
    with open(REPORTS_DIR / "calibration_uncertainty_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("Calibration & Uncertainty Analysis Complete.")
    print(f"Adult ECE: {adult_metrics['ece']:.4f}, Brier: {adult_metrics['brier_score']:.4f}")
    print(f"Adult Mean Confidence: {adult_metrics['mean_confidence']:.4f}")
    print(f"Pediatric ECE: {ped_metrics['ece']:.4f}, Brier: {ped_metrics['brier_score']:.4f}")
    print(f"Pediatric Mean Confidence: {ped_metrics['mean_confidence']:.4f}")

if __name__ == "__main__":
    main()
