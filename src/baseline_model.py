"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 1: Baseline Architecture Module

Architecture & Design Rationale:
--------------------------------
1. Edge Deployment Suitability (MobileNetV3-Small):
   MobileNetV3-Small was specifically designed for ultra-lightweight, low-latency
   inference on resource-constrained edge hardware (e.g., embedded systems,
   handheld Point-of-Care Ultrasound (POCUS) scanners, and mobile triage units).
   By leveraging hardware-aware Neural Architecture Search (NAS), NetAdapt algorithms,
   depthwise separable convolutions, and efficient Squeeze-and-Excitation (SE) blocks,
   MobileNetV3-Small minimizes computational complexity (~0.06 GFLOPs, ~1.5M parameters)
   and memory footprint while maintaining high classification accuracy and low power
   consumption in critical neonatal care environments.

2. Transfer Learning for Small Medical Datasets:
   Clinical neonatal datasets—particularly lung ultrasound (LUS) and pediatric
   radiographs—are notoriously scarce, high-dimensional, and expensive to annotate
   by certified neonatologists and radiologists. Training a deep convolutional network
   from scratch on limited samples carries severe risks of catastrophic overfitting
   and brittle feature representations. Transfer learning from ImageNet-1k initializes
   the model with rich, general visual primitives (Gabor-like edge detectors, texture
   gradients, structural boundaries) that transfer effectively to medical ultrasound
   artifacts (such as A-lines, B-lines, pleural line thickening, and consolidations),
   substantially accelerating convergence and boosting triage generalization.

3. Freezing the Pretrained Backbone:
   Freezing the feature extraction backbone preserves the universal visual representations
   learned during large-scale pretraining and prevents destabilizing weight updates from
   high-variance gradients during early training epochs. Furthermore, freezing backbone
   parameters drastically reduces GPU/CPU memory consumption during backpropagation
   and speeds up training cycles on edge compute clusters.

4. Custom Classifier Head for 3-Class Neonatal Triage:
   The original 1000-class ImageNet output layer is replaced with a custom linear
   head projecting bottleneck feature representations (1024-d) to 3 discrete clinical
   triage risk tiers:
     - Class 0: Normal / Low Risk (Routine monitoring, no immediate escalation)
     - Class 1: Moderate Risk (Supplemental O2 / continuous clinical observation)
     - Class 2: High Risk (Urgent respiratory intervention / NICU transfer)
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# =====================================================================
# Global Constants
# =====================================================================
MODEL_NAME: str = "MobileNetV3-Small"
MODEL_WEIGHTS: str = "ImageNet-1K (DEFAULT)"
DEFAULT_NUM_CLASSES: int = 3
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# =====================================================================
# Device Selection
# =====================================================================
def get_device() -> torch.device:
    """
    Selects the optimal available compute device with priority:
    1. Apple Silicon GPU (MPS)
    2. NVIDIA GPU (CUDA)
    3. CPU fallback

    Returns:
        torch.device: The selected PyTorch device object.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =====================================================================
# Baseline Model Builder
# =====================================================================
def build_baseline_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Constructs the MobileNetV3-Small baseline architecture with pretrained weights
    and a custom classifier head configured for neonatal respiratory triage.

    Args:
        num_classes: Number of target classification categories (default: 3).
        freeze_backbone: If True, freezes feature extractor parameters and keeps
            only the classifier head trainable. If False, all parameters remain
            trainable for full fine-tuning (default: True).

    Returns:
        nn.Module: Configured MobileNetV3-Small PyTorch model.
    """
    # Load pretrained MobileNetV3-Small using recommended torchvision weights API
    weights = MobileNet_V3_Small_Weights.DEFAULT
    PRETRAINED_WEIGHTS = weights.name
    model = mobilenet_v3_small(weights=weights)

    # Replace ONLY the final linear classification layer in the classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        in_features=in_features,
        out_features=num_classes,
    )

    # Configure parameter freeze states
    if freeze_backbone:
        # Freeze only feature extractor parameters
        for param in model.features.parameters():
            param.requires_grad = False
        # Ensure classifier head parameters remain trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # All parameters remain trainable for end-to-end fine-tuning
        for param in model.parameters():
            param.requires_grad = True

    return model


# =====================================================================
# Metadata Extraction
# =====================================================================
def get_model_metadata(
    model: nn.Module,
    freeze_backbone: bool = True,
) -> Dict[str, Any]:
    """
    Extracts structural metadata, parameter statistics, and configuration
    details from the baseline model without hardcoded architecture names.

    Args:
        model: PyTorch model instance to inspect.
        freeze_backbone: Boolean flag indicating if backbone freezing is active.

    Returns:
        Dict[str, Any]: Metadata dictionary containing architecture details,
            parameter counts, and input specifications.
    """
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_parameters = total_parameters - trainable_parameters

    # Dynamically extract output classes from the final classification layer if possible
    num_classes = DEFAULT_NUM_CLASSES
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        if hasattr(last_layer, "out_features"):
            num_classes = last_layer.out_features

    return {
    "architecture": MODEL_NAME,
    "num_classes": num_classes,
    "input_size": DEFAULT_IMAGE_SIZE,
    "pretrained": True,
    "weights": MODEL_WEIGHTS,
    "freeze_backbone": freeze_backbone,
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_parameters,
    "frozen_parameters": frozen_parameters,
}


# =====================================================================
# Console Summary
# =====================================================================
def print_model_summary(
    metadata: Dict[str, Any],
    device: torch.device,
) -> None:
    """
    Prints a clean, professionally formatted summary of the model configuration,
    parameter counts, and compute runtime environment.

    Args:
        metadata: Metadata dictionary produced by get_model_metadata().
        device: Active PyTorch compute device.
    """
    pretrained_str = "Yes" if metadata.get("pretrained") else "No"
    frozen_str = "Yes" if metadata.get("freeze_backbone") else "No"
    input_size_str = str(metadata.get("input_size", DEFAULT_IMAGE_SIZE))

    print("==================================================")
    print("Baseline Model Summary")
    print("==================================================")
    print()
    print(f"Architecture         : {metadata.get('architecture')}")
    print(f"Device               : {device}")
    print(f"Pretrained           : {pretrained_str}")
    print(f"Weights              : {metadata.get('weights')}")
    print(f"Frozen Backbone      : {frozen_str}")
    print(f"Classes              : {metadata.get('num_classes')}")
    print(f"Input Size           : {input_size_str}")
    print()
    print(f"Total Parameters     : {metadata.get('total_parameters', 0):,}")
    print(f"Trainable Parameters : {metadata.get('trainable_parameters', 0):,}")
    print(f"Frozen Parameters    : {metadata.get('frozen_parameters', 0):,}")
    print()
    print("==================================================")


# =====================================================================
# Verification & Execution Pipeline
# =====================================================================
def main() -> None:
    """
    Main execution pipeline:
    1. Detect optimal compute device.
    2. Build baseline MobileNetV3-Small with frozen backbone.
    3. Extract metadata and print summary.
    4. Move model to device and switch to evaluation mode.
    5. Execute forward pass with dummy tensor.
    6. Assert output shapes and print verification report.
    """
    # 1. Device Configuration
    device = get_device()

    # 2. Build Baseline Model
    freeze_backbone = True
    model = build_baseline_model(
        num_classes=DEFAULT_NUM_CLASSES,
        freeze_backbone=freeze_backbone,
    )

    # 3. Model Metadata & Summary
    metadata = get_model_metadata(model=model, freeze_backbone=freeze_backbone)
    print_model_summary(metadata=metadata, device=device)

    # 4. Device Placement & Evaluation Mode
    model.to(device)
    model.eval()

    # 5. Dummy Input Creation (Batch of 4 RGB images: 4 x 3 x 224 x 224)
    dummy_input = torch.randn(
        4,
        3,
        *DEFAULT_IMAGE_SIZE,
        device=device,
        dtype=torch.float32,
    )

    # 6. Forward Pass (Inference Mode without gradients)
    with torch.no_grad():
        output = model(dummy_input)

    # 7. Verification Assertion
    assert output.shape == (4, DEFAULT_NUM_CLASSES), (
        f"Output shape mismatch: expected (4, {DEFAULT_NUM_CLASSES}), got {output.shape}"
    )

    # 8. Final Console Output
    print(f"Input Tensor Shape   : {dummy_input.shape}")
    print(f"Output Tensor Shape  : {output.shape}")
    print(f"Output Device        : {output.device}")
    print(f"Output Data Type     : {output.dtype}")
    print("\n[SUCCESS] Baseline MobileNetV3-Small verified successfully.")


if __name__ == "__main__":
    main()
