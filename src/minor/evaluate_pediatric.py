import ast
import json
import argparse
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from infer import load_model, transform, CLASS_NAMES, DEVICE
from gradcam import GradCAM, generate_overlay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_CSV = PROJECT_ROOT / "data/external/pocus_atlas_pediatric/pocus_atlas_pediatric_audit.csv"
REPORTS_DIR = PROJECT_ROOT / "models/minor/reports"
VIS_DIR = PROJECT_ROOT / "models/minor/visualizations/pediatric"

NUM_FRAMES = 16

def extract_uniform_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    if total_frames == 0:
        return frames
        
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, Image.fromarray(frame_rgb)))
            
    cap.release()
    return frames

def extract_age_text(row):
    text = str(row.get('title', '')).lower() + ' ' + str(row.get('body', '')).lower()
    for word in ['old', 'year', 'month', 'day', 'wk']:
        if word in text:
            return text
    return "Unknown"

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(AUDIT_CSV)
    
    # Load model (frozen, eval mode)
    model = load_model()
    model.eval()
    
    target_layer = model.backbone.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    frame_results = []
    case_results = []
    
    # Track representative cases for Grad-CAM
    gradcam_targets = {'Normal': None, 'Pneumonia': None, 'Pneumothorax': None, 'unspecified': None}
    
    cases_evaluated = 0
    videos_evaluated = 0
    frames_evaluated = 0
    
    for idx, row in df.iterrows():
        try:
            video_dict = ast.literal_eval(row['video'])
            video_path = video_dict['path']
        except:
            continue
            
        if not Path(video_path).exists():
            continue
            
        pathology = row['pathology']
        case_id = row['id']
        source = row['source']
        age_text = extract_age_text(row)
        
        # Save one representative case per pathology for Grad-CAM
        if pathology in gradcam_targets and gradcam_targets[pathology] is None:
            gradcam_targets[pathology] = (case_id, video_path)
            
        frames = extract_uniform_frames(video_path, NUM_FRAMES)
        
        if not frames:
            continue
            
        cases_evaluated += 1
        videos_evaluated += 1
        
        case_probs_normal = []
        case_probs_abnormal = []
        case_preds = []
        
        for frame_idx, img in frames:
            # Step 4: Preprocessing exactly identical to infer.py
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                
            p_normal = float(probs[0].item())
            p_abnormal = float(probs[1].item())
            pred_class_idx = int(torch.argmax(probs).item())
            pred_class = CLASS_NAMES[pred_class_idx]
            
            frame_results.append({
                'case_id': case_id,
                'video_id': case_id,
                'frame_index': frame_idx,
                'source': source,
                'age_text': age_text,
                'pathology': pathology,
                'predicted_class': pred_class,
                'normal_probability': p_normal,
                'abnormal_probability': p_abnormal
            })
            
            case_probs_normal.append(p_normal)
            case_probs_abnormal.append(p_abnormal)
            case_preds.append(pred_class_idx)
            frames_evaluated += 1
            
        # Case aggregation
        num_frames = len(frames)
        mean_p_normal = float(np.mean(case_probs_normal))
        mean_p_abnormal = float(np.mean(case_probs_abnormal))
        max_p_abnormal = float(np.max(case_probs_abnormal))
        
        normal_frac = sum(1 for p in case_preds if p == 0) / num_frames
        abnormal_frac = sum(1 for p in case_preds if p == 1) / num_frames
        majority_vote = 'Normal' if normal_frac >= 0.5 else 'Abnormal'
        
        case_results.append({
            'case_id': case_id,
            'video_id': case_id,
            'source': source,
            'age_text': age_text,
            'pathology': pathology,
            'num_frames': num_frames,
            'mean_normal_probability': mean_p_normal,
            'mean_abnormal_probability': mean_p_abnormal,
            'normal_frame_fraction': normal_frac,
            'abnormal_frame_fraction': abnormal_frac,
            'majority_prediction': majority_vote,
            'max_abnormal_probability': max_p_abnormal
        })
        
    # Save CSVs
    frame_df = pd.DataFrame(frame_results)
    case_df = pd.DataFrame(case_results)
    
    frame_df.to_csv(REPORTS_DIR / "pediatric_zero_shot_frame_predictions.csv", index=False)
    case_df.to_csv(REPORTS_DIR / "pediatric_zero_shot_case_predictions.csv", index=False)
    
    # Ground truth mapping
    # Rule: Normal = 0 (Normal). Pneumonia & Pneumothorax = 1 (Abnormal). unspecified = Exclude from formal metrics.
    # This is defensible because Pneumonia (consolidation, B-lines) and Pneumothorax (absence of sliding) are widely accepted as pathological LUS findings.
    
    y_true = []
    y_pred = []
    confidence_all = []
    conflicts = []
    
    for res in case_results:
        majority = 1 if res['majority_prediction'] == 'Abnormal' else 0
        conf = max(res['mean_normal_probability'], res['mean_abnormal_probability'])
        confidence_all.append(conf)
        
        pathology = res['pathology']
        if pathology == 'Normal':
            gt = 0
        elif pathology in ['Pneumonia', 'Pneumothorax']:
            gt = 1
        else:
            gt = None
            
        if gt is not None:
            y_true.append(gt)
            y_pred.append(majority)
            if gt != majority and conf > 0.8:
                conflicts.append(res['case_id'])
                
    # Calculate metrics
    metrics = {
        "cases_evaluated": cases_evaluated,
        "videos_evaluated": videos_evaluated,
        "frames_evaluated": frames_evaluated,
        "pathology_distribution": df['pathology'].value_counts().to_dict(),
        "prediction_distribution": case_df['majority_prediction'].value_counts().to_dict(),
        "confidence_statistics": {
            "mean": float(np.mean(confidence_all)),
            "median": float(np.median(confidence_all)),
            "min": float(np.min(confidence_all)),
            "max": float(np.max(confidence_all))
        },
        "high_confidence_disagreements": conflicts
    }
    
    if len(y_true) > 0:
        metrics["formal_metrics"] = {
            "valid_cases_count": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
    else:
        metrics["formal_metrics"] = None
        
    with open(REPORTS_DIR / "pediatric_zero_shot_evaluation.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate Representative Grad-CAMs
    # We must ensure gradients can flow for Grad-CAM.
    for param in model.parameters():
        param.requires_grad = True
        
    gradcam_paths = []
    for path_class, target in gradcam_targets.items():
        if target is None: continue
        c_id, v_path = target
        
        frames = extract_uniform_frames(v_path, NUM_FRAMES)
        if not frames: continue
        
        # Take the middle frame
        mid_idx, mid_img = frames[len(frames)//2]
        
        tensor = transform(mid_img).unsqueeze(0).to(DEVICE)
        tensor.requires_grad = True
        
        cam, logits = grad_cam(tensor)
        
        # We need original img in cv2 format (BGR -> RGB)
        img_np = np.array(mid_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Temporary save to use generate_overlay (which takes path) or rewrite it
        tmp_path = VIS_DIR / f"tmp_{c_id}.jpg"
        cv2.imwrite(str(tmp_path), img_bgr)
        
        overlay = generate_overlay(tmp_path, cam)
        out_name = VIS_DIR / f"pediatric_zero_shot_gradcam_{path_class}_{c_id}.jpg"
        overlay.save(out_name)
        gradcam_paths.append(str(out_name))
        
        tmp_path.unlink()
        
    print(f"Evaluation Complete!")
    print(f"Cases Evaluated: {cases_evaluated}")
    print(f"Frames Evaluated: {frames_evaluated}")
    print(f"Pathology Distribution: {metrics['pathology_distribution']}")
    print(f"Prediction Distribution: {metrics['prediction_distribution']}")
    if metrics["formal_metrics"]:
        print(f"Accuracy: {metrics['formal_metrics']['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['formal_metrics']['balanced_accuracy']:.4f}")
        print(f"F1 Score: {metrics['formal_metrics']['f1']:.4f}")
    print(f"Mean Confidence: {metrics['confidence_statistics']['mean']:.4f}")
    print(f"Median Confidence: {metrics['confidence_statistics']['median']:.4f}")
    print(f"Grad-CAM Generated: {len(gradcam_paths)}")
    for p in gradcam_paths:
        print(f" - {p}")
        
if __name__ == '__main__':
    main()
