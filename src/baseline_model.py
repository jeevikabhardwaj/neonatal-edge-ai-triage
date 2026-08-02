"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 1: Baseline Architecture Setup & Pipeline Verification

This script loads a pretrained MobileNetV3-Small model, modifies the final
classifier head for 3 triage risk levels, and verifies inference with dummy inputs.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def get_device() -> torch.device:
    """
    Selects the best available compute device:
    1. Apple Silicon GPU (MPS)
    2. NVIDIA GPU (CUDA)
    3. CPU fallback
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def build_baseline_model(num_classes: int = 3) -> nn.Module:
    """
    Initializes a pretrained MobileNetV3-Small backbone and replaces
    only the final classification layer to output the specified number of classes.

    Class Mapping:
      - 0: Normal
      - 1: Moderate Risk
      - 2: High Risk
    """
    # Load pretrained MobileNetV3-Small using the latest torchvision weights API
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Replace ONLY the final linear classification layer
    # while keeping pretrained feature extraction layers unchanged.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        in_features=in_features,
        out_features=num_classes
    )

    return model


def main() -> None:
    """Main execution function."""

    # 1. Device Configuration
    device = get_device()
    print(f"[1] Selected Device: {device}")

    # 2. Model Initialization & Classifier Modification
    model = build_baseline_model(num_classes=3)
    model.to(device)

    # 3. Switch to Evaluation Mode
    model.eval()

    # 4. Dummy Input Creation (Batch of 4 RGB ultrasound images: 4 × 3 × 224 × 224)
    dummy_input = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
        dtype=torch.float32
    )

    print(f"[2] Input Tensor Shape: {dummy_input.shape}")

    # 5. Forward Pass (without gradient computation)
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output dimensions
    assert output.shape == (4, 3), (
        f"Unexpected output shape: {output.shape}"
    )

    # 6. Parameter Counting
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    # 7. Console Output & Verification
    print(f"[3] Output Tensor Shape: {output.shape}")
    print(f"[4] Output Device: {output.device}")
    print(f"[5] Output Data Type: {output.dtype}")
    print(
        f"[6] Total Trainable Parameters: "
        f"{trainable_params:,} (Total: {total_params:,})"
    )

    print(
        "\n[SUCCESS] Baseline MobileNetV3-Small model initialized "
        "and forward pass verified successfully."
    )


if __name__ == "__main__":
    main()
