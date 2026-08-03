"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 5: Production-Ready Inference Engine Backend

This module serves as the central prediction engine API for lung ultrasound
image triage. It loads the trained MobileNetV3-Small checkpoint, preprocesses
input images, executes forward inference, computes class probability distributions,
and generates clinically grounded triage recommendations.

Designed for seamless integration with downstream modules (Streamlit Dashboard,
Multimodal Fusion Engine, Grad-CAM Explainability, and PDF Triage Reporting).
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Ensure project root and src/ are in sys.path for flexible execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.baseline_model import build_baseline_model, get_device
except ImportError:
    from baseline_model import build_baseline_model, get_device


# Model and Preprocessing Configuration Constants
MODEL_VERSION = "baseline_mobilenetv3_v1"
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(
    checkpoint_path: Union[Path, str] = "models/best_baseline_model.pth",
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, List[str], str, torch.device]:
    """
    Loads the trained MobileNetV3-Small checkpoint, dynamically determines
    class names, reconstructs architecture, and moves to target compute device.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Target compute device (if None, auto-detected).

    Returns:
        Tuple[torch.nn.Module, List[str], torch.device]:
            - Loaded PyTorch model in eval mode
            - List of class names
            - Active compute device

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    resolved_path = (PROJECT_ROOT / checkpoint_path).resolve() if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)

    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint file not found at: '{resolved_path}'\n"
            "Please train the baseline model first by running:\n"
            "    python src/train.py"
        )

    if device is None:
        device = get_device()

    checkpoint: Dict[str, Any] = torch.load(
        str(resolved_path),
        map_location="cpu"
    )

    # Extract class names and model metadata
    class_names: List[str] = checkpoint.get(
        "classes",
        ["high_risk", "moderate_risk", "normal"]
    )

    model_version: str = checkpoint.get(
        "model_version",
        MODEL_VERSION
    )

    num_classes = len(class_names)

    # Reconstruct architecture and load weights
    model = build_baseline_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, model_version, device


def preprocess_image(
    image: Union[str, Path, Image.Image],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """
    Preprocesses a single image for MobileNetV3 inference.

    Accepts either an image filesystem path or a PIL Image object.
    Ensures 3-channel RGB representation, resizes to target resolution,
    converts to PyTorch tensor, normalizes with ImageNet statistics,
    and prepends a batch dimension.

    Args:
        image: File path (str or Path) or PIL Image instance.
        image_size: Target (height, width) resolution.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, 224, 224).

    Raises:
        FileNotFoundError: If the specified image path does not exist.
        ValueError: If the input cannot be processed or is an unsupported format.
    """
    # Load PIL Image if path provided
    if isinstance(image, (str, Path)):
        img_path = Path(image).resolve()
        if not img_path.exists() or not img_path.is_file():
            raise FileNotFoundError(
                f"\n[ERROR] Input image file not found at: '{img_path}'"
            )
        try:
            pil_img = Image.open(img_path)
            pil_img.load()  # Verify image can be decoded
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"\n[ERROR] Unsupported or corrupted image format at: '{img_path}'"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"\n[ERROR] Failed to open image at '{img_path}': {exc}"
            ) from exc
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise ValueError(
            f"\n[ERROR] Expected image as file path (str, Path) or PIL.Image, got: {type(image)}"
        )

    # Convert to 3-channel RGB (handles grayscale 'L', RGBA, palette modes)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    transform_pipeline = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor: torch.Tensor = transform_pipeline(pil_img)

    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)
    return tensor


def get_recommendation(predicted_class: str, confidence: float) -> str:
    """
    Generates a clinical triage recommendation based on predicted risk level
    and prediction confidence.

    Independent of inference logic for modularity and clinical interpretability.

    Args:
        predicted_class: Predicted class label (e.g. 'normal', 'moderate_risk', 'high_risk').
        confidence: Prediction confidence score between 0.0 and 1.0.

    Returns:
        str: Human-readable clinical triage action.
    """
    class_key = predicted_class.lower().replace(" ", "_")

    recommendations = {
        "normal": "Continue routine monitoring.",
        "moderate_risk": "Clinical review recommended.",
        "high_risk": "Immediate clinical assessment recommended.",
    }

    base_recommendation = recommendations.get(
        class_key,
        "Clinical assessment recommended for unclassified risk level.",
    )

    if confidence < 0.60:
     base_recommendation += (
        " Prediction confidence is low. "
        "Interpret the result alongside clinical assessment."
    )
    return base_recommendation


def predict_single_sample(
    image: Union[str, Path, Image.Image],
    clinical_data: Optional[Dict[str, Any]] = None,
    model: Optional[torch.nn.Module] = None,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Union[Path, str] = "models/best_baseline_model.pth",
) -> Dict[str, Any]:
    """
    Central inference engine API for single-sample lung ultrasound triage.

    Loads the model (or reuses preloaded model), preprocesses the input image,
    executes forward pass inference, computes softmax class probabilities,
    and returns a structured prediction dictionary.

    Args:
        image: File path or PIL Image.
        clinical_data: Optional clinical vital signs dictionary (reserved for
                       future multimodal fusion milestone).
        model: Optional preloaded PyTorch model instance for batch/API reuse.
        class_names: Optional preloaded class label list.
        device: Optional preloaded compute device.
        checkpoint_path: Path to checkpoint if model needs to be loaded.

    Returns:
        Dict[str, Any]: Structured prediction dictionary containing:
            - predicted_class: str
            - confidence: float
            - probabilities: Dict[str, float]
            - recommended_action: str
            - prediction_timestamp: str
            - model_version: str
    """
    # 1. Model resolution
    if model is None or class_names is None or device is None:
        model, class_names, model_version, device = load_model(checkpoint_path=checkpoint_path, device=device)

    # 2. Preprocessing
    input_tensor = preprocess_image(image=image)
    input_tensor = input_tensor.to(device)

    # 3. Model Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities_tensor = F.softmax(logits, dim=1)

    # 4. Probabilities and Predictions Extraction
    probs_np = probabilities_tensor.cpu().squeeze(0).numpy()
    pred_index = int(probs_np.argmax())
    predicted_class = class_names[pred_index]
    confidence = float(probs_np[pred_index])

    probabilities_dict: Dict[str, float] = {
        class_name: round(float(probs_np[idx]), 4)
        for idx, class_name in enumerate(class_names)
    }

    # 5. Recommendation Generation
    recommendation = get_recommendation(
        predicted_class=predicted_class,
        confidence=confidence,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 6. Structured Output
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities_dict,
        "recommended_action": recommendation,
        "prediction_timestamp": timestamp,
        "model_version": model_version,
    }


def print_prediction_summary(result: Dict[str, Any]) -> None:
    """
    Formats and prints a clean, human-readable prediction report to the console.

    Args:
        result: Structured dictionary returned by predict_single_sample().
    """
    print("=" * 60)
    print("                Prediction Summary")
    print("=" * 60)

    display_name = result["predicted_class"].replace("_", " ").title()

    print(f"Predicted Class           : {display_name}")
    print(
        f"Confidence                : {result['confidence']:.4f} "
        f"({result['confidence'] * 100:.2f}%)"
    )

    print("Class Probabilities:")

    for class_name, prob in result["probabilities"].items():
        display_class = class_name.replace("_", " ").title()
        print(f"  - {display_class:<18}: {prob:.4f} ({prob * 100:.2f}%)")

    print(f"Clinical Recommendation   : {result['recommended_action']}")
    print(f"Prediction Timestamp      : {result['prediction_timestamp']}")
    print(f"Model Version             : {result['model_version']}")
    print("=" * 60)
def locate_sample_image() -> Path:
    """
    Locates the first available sample image from the raw dataset search paths:
    1. data/raw/high_risk/sample_01.png
    2. data/raw/moderate_risk/sample_01.png
    3. data/raw/normal/sample_01.png

    Returns:
        Path: Absolute path to the resolved sample image.

    Raises:
        FileNotFoundError: If none of the search paths contain an existing image.
    """
    search_candidates = [
        PROJECT_ROOT / "data" / "raw" / "high_risk" / "sample_01.png",
        PROJECT_ROOT / "data" / "raw" / "moderate_risk" / "sample_01.png",
        PROJECT_ROOT / "data" / "raw" / "normal" / "sample_01.png",
    ]

    for candidate in search_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "\n[ERROR] No test sample image found in candidate paths:\n"
        + "\n".join(f"  - {c}" for c in search_candidates)
        + "\nPlease generate the mock dataset first by running:\n"
        "    python src/generate_mock_data.py"
    )


def main() -> None:
    """Main execution function for standalone inference testing."""
    sample_image_path = locate_sample_image()
    print(f"[INFO] Performing inference on sample image: {sample_image_path.relative_to(PROJECT_ROOT)}")

    result = predict_single_sample(image=sample_image_path)
    print()
    print_prediction_summary(result)


if __name__ == "__main__":
    main()