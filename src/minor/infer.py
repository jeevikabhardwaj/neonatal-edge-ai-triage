from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms
from model import create_model

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / "models/minor/checkpoints/minor_baseline_best.pth"

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal",
}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Same normalization used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    model = create_model(
        num_classes=2,
        pretrained=False,
        device=DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    # Support either a raw state_dict or our training checkpoint format
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_class = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_class].item())

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(2)
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="OpenPOCUS Minor LUS inference"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to a lung ultrasound image",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    result = predict(image_path)

    print("\n" + "=" * 50)
    print("OpenPOCUS Minor LUS Inference")
    print("=" * 50)
    print(f"Device:     {DEVICE}")
    print(f"Image:      {image_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("\nClass probabilities:")

    for label, probability in result["probabilities"].items():
        print(f"  {label:10s}: {probability:.4f}")

    print("=" * 50)
    print(
        "\nResearch prototype only — this baseline is not "
        "a clinical diagnosis or neonatal risk classifier."
    )


if __name__ == "__main__":
    main()
