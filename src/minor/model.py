import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class MinorMobileNetV3(nn.Module):
    """
    Lightweight MobileNetV3-Small binary classifier.

    Task:
        0 = Normal
        1 = Abnormal
    """

    def __init__(self, num_classes=2, pretrained=True):

        super().__init__()

        if pretrained:
            weights = (
                MobileNet_V3_Small_Weights.DEFAULT
            )
        else:
            weights = None

        self.backbone = mobilenet_v3_small(
            weights=weights
        )

        # Replace ImageNet classifier
        input_features = (
            self.backbone.classifier[-1].in_features
        )

        self.backbone.classifier[-1] = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)


def create_model(
    num_classes=2,
    pretrained=True,
    device=None,
):
    model = MinorMobileNetV3(
        num_classes=num_classes,
        pretrained=pretrained,
    )

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":

    print("\n==========================================")
    print("       MINOR MOBILENETV3 MODEL")
    print("==========================================")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = create_model(
        num_classes=2,
        pretrained=True,
        device=device,
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nDevice:", device)
    print("Classes: 2")
    print("Labels:")
    print("  0 = Normal")
    print("  1 = Abnormal")

    print(
        f"\nTotal parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # Test forward pass
    dummy = torch.randn(
        2, 3, 224, 224,
        device=device
    )

    with torch.no_grad():
        output = model(dummy)

    print(
        "\nInput shape:",
        tuple(dummy.shape)
    )

    print(
        "Output shape:",
        tuple(output.shape)
    )

    print("\n✓ Model forward pass successful")

    print("\n==========================================\n")
