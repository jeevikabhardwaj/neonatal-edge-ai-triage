# Neonatal LUS Edge AI Triage
### Synthetic Neonatal Lung Ultrasound Classification System

> **No real neonatal patient data used.** All neonatal images are synthetically derived from publicly available adult LUS data (jannisborn/covid19_ultrasound, CC-BY 4.0). Clinical data is generated from published RDS cohort statistics.

---

## Status

| Phase | Component | Status |
|-------|-----------|--------|
| Phase 1 | OpenPOCUS dataset + manifest | ✅ Done |
| Phase 1 | MobileNetV3-Small binary model | ✅ Done |
| Phase 1 | Training (real data, 904 frames) | ✅ Done |
| Phase 1 | Evaluation | ✅ Done |
| Phase 2 | Synthetic neonatal images (904) | ✅ Done |
| Phase 2 | Synthetic clinical data (500 patients) | ✅ Done |
| Phase 2 | Multimodal fusion training | ✅ Done |
| Phase 2 | Evaluation | ✅ Done |
| Deploy | TorchScript export (edge models) | ✅ Done |
| Deploy | Gradio demo app | ✅ Done |

---

## Results

| Phase | Metric | Value | Notes |
|-------|--------|-------|-------|
| Phase 1 | F1 (weighted) | 0.88 | Binary Normal/Abnormal, stratified split |
| Phase 1 | Balanced Accuracy | 0.9030 | Normal recall=1.00, Abnormal recall=0.81 |
| Phase 1 | ROC-AUC | **0.9989** | Fixed from 0.394 via stratified patient split |
| Phase 2 | F1 (macro) | 0.75 | 3-class multimodal (eval) |
| Phase 2 | ROC-AUC (OvR) | **0.8489** | Meaningful learning on synthetic data |
| Phase 2 | High Risk F1 | 0.96 | Clinically critical class |

---

## Quick Start

```bash
pip install -r requirements.txt
pip install gradio

# 1. Build stratified manifest from real OpenPOCUS data
python scripts/build_manifest.py --input data/raw --output data/processed/phase1_manifest.csv

# 2. Train Phase 1 (binary)
python src/training/train_phase1.py --config config/phase1_config.yaml

# 3. Generate synthetic neonatal images
python scripts/generate_neonatal_images.py

# 4. Generate synthetic clinical data
python scripts/generate_clinical_data.py

# 5. Train Phase 2 (multimodal 3-class)
python src/training/train_phase2.py --config config/phase2_config.yaml

# 6. Export TorchScript edge models
python scripts/export_models.py

# 7. Run Gradio demo
python app.py
```

---

## Data

| Dataset | Type | Source | Frames |
|---------|------|--------|--------|
| covid19_ultrasound | Real adult LUS | [jannisborn/covid19_ultrasound](https://github.com/jannisborn/covid19_ultrasound) (CC-BY 4.0) | 904 |
| Synthetic neonatal LUS | Derived from above | `src/data/neonatal_synthesizer.py` | 904 |
| Synthetic clinical | Generated from literature | `scripts/generate_clinical_data.py` | 500 patients |

**Label mapping from source:** `Reg_*` → normal; `Cov_*`, `Pneu_*`, `Vir_*` → abnormal

---

## Project Structure

```
├── app.py                       ← Gradio demo (Phase 1 + Phase 2)
├── CLAUDE.md
├── config/
│   ├── phase1_config.yaml
│   └── phase2_config.yaml
├── src/
│   ├── data/
│   │   ├── openpocus_dataset.py
│   │   └── neonatal_synthesizer.py
│   ├── models/
│   │   ├── phase1_model.py
│   │   └── phase2_model.py
│   ├── training/
│   │   ├── train_phase1.py
│   │   └── train_phase2.py
│   └── evaluation/
│       ├── evaluate_phase1.py
│       └── evaluate_phase2.py
├── scripts/
│   ├── build_manifest.py        ← Stratified patient-level split
│   ├── export_models.py         ← TorchScript export
│   ├── generate_clinical_data.py
│   └── generate_neonatal_images.py
└── docs/
    └── experimental_log.md
```

---

## Model Architecture

**Phase 1:** MobileNetV3-Small → Binary head (2 classes), ~1.5M params, ~5 MB

**Phase 2:**
```
Image → MobileNetV3-Small encoder (576-dim)
                                              → Concat (608-dim) → FC(128) → 3 classes
Clinical → MLP encoder (8→32-dim, BN+Dropout)
```

---

## Known Issues

| Issue | Detail | Fix Status |
|-------|--------|-----------|
| Phase 1 ROC-AUC was 0.394 | Test split had 4.4:1 class imbalance (101 abnormal / 23 normal) | ✅ Fixed — stratified split in `build_manifest.py`; ROC-AUC now **0.9989** |
| Synthetic data only for Phase 2 | Phase 2 trained on synthetically augmented frames, not real neonatal LUS | By design — no real neonatal data available |

---

## Ethical Note

This system is a **research prototype**:
- No real neonatal patient data used at any stage
- Results cannot be extrapolated to clinical performance without validation on real neonatal LUS data
- Not a medical device
- Built for research and educational purposes only
