"""
Multimodal Edge AI System for Neonatal Respiratory Triage
Milestone 5: Production-Grade Inference Engine & Multi-Run Latency Benchmark

This module serves as the central prediction engine API for lung ultrasound
image triage. It handles image validation, preprocessing, model weight restoration,
warm-up forward passes, multi-run steady-state latency benchmarking, clinical confidence
thresholding, and structured JSON-serializable output generation.

Designed as the core backend API for:
- Streamlit Clinical Dashboard
- Multimodal Fusion Engine (incorporating vital signs)
- Explainability (Grad-CAM saliency mapping)
- Automated PDF Clinical Report Generation
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
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


# Model, Preprocessing, and Clinical Threshold Constants
MODEL_VERSION = "baseline_mobilenetv3_v1"
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
CONFIDENCE_THRESHOLD: float = 0.60
DEFAULT_WARMUP_RUNS: int = 2
DEFAULT_BENCHMARK_RUNS: int = 20
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


# =====================================================================
# Device Synchronization Helper
# =====================================================================
def synchronize_device(device: Optional[Union[torch.device, str]] = None) -> None:
    """
    Synchronizes asynchronous GPU/accelerator execution queues (CUDA, MPS)
    to guarantee accurate and reproducible latency timing benchmarks.
    Safely no-ops on CPU or unsupported backends without throwing errors.

    Args:
        device: Target compute device (torch.device, str, or None).
    """
    if device is None:
        return

    device_type = device.type if isinstance(device, torch.device) else str(device).lower()

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


# =====================================================================
# Latency Statistics Helper
# =====================================================================
def compute_latency_stats(values: List[float]) -> Dict[str, float]:
    """
    Computes statistical metrics (average, minimum, maximum, standard deviation)
    for a list of latency timing values in milliseconds.

    Args:
        values: List of float latency values in milliseconds.

    Returns:
        Dict[str, float]: Dictionary containing average_ms, minimum_ms, maximum_ms, and std_ms.
    """
    if not values:
        return {
            "average_ms": 0.0,
            "minimum_ms": 0.0,
            "maximum_ms": 0.0,
            "std_ms": 0.0,
        }

    arr = np.array(values, dtype=float)
    avg = float(np.mean(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    std_val = float(np.std(arr)) if len(arr) > 1 else 0.0

    return {
        "average_ms": round(avg, 2),
        "minimum_ms": round(min_val, 2),
        "maximum_ms": round(max_val, 2),
        "std_ms": round(std_val, 2),
    }


# =====================================================================
# Model Loading & Reconstruction
# =====================================================================
def load_trained_model(
    checkpoint_path: Union[Path, str] = "models/checkpoints/best_baseline_model.pth",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, List[str], str, torch.device]:
    """
    Loads the trained MobileNetV3-Small checkpoint, dynamically restores
    metadata and class names, reconstructs the architecture, and prepares
    the model for production inference.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Target compute device (if None, auto-detected via get_device()).

    Returns:
        Tuple[nn.Module, List[str], str, torch.device]:
            - model: Reconstructed PyTorch model in eval mode on target device
            - class_names: List of class labels
            - model_version: Model version identifier string
            - device: Active compute device

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    resolved_path = (PROJECT_ROOT / checkpoint_path).resolve() if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)

    # Fallback check to legacy root path if default checkpoint path does not exist
    if not resolved_path.exists() or not resolved_path.is_file():
        legacy_path = (PROJECT_ROOT / "models" / "best_baseline_model.pth").resolve()
        if legacy_path.exists() and legacy_path.is_file():
            resolved_path = legacy_path

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
        map_location="cpu",
    )

    # Extract class names and model metadata
    class_names: List[str] = checkpoint.get(
        "classes",
        ["high_risk", "moderate_risk", "normal"],
    )

    model_version: str = checkpoint.get(
        "model_version",
        MODEL_VERSION,
    )

    num_classes = len(class_names)

    # Reconstruct architecture and load weights
    model = build_baseline_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, model_version, device


# Alias for backward compatibility
load_model = load_trained_model


# =====================================================================
# Image Validation & Preprocessing
# =====================================================================
def validate_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Validates input image source, ensures readable format, handles grayscale/RGBA,
    and converts the image into a standardized 3-channel RGB PIL Image.

    Args:
        image: File path (str or Path) or PIL Image instance.

    Returns:
        Image.Image: Standardized 3-channel RGB PIL Image.

    Raises:
        FileNotFoundError: If the specified image path does not exist.
        ValueError: If the input is corrupt, unreadable, or of an unsupported format.
    """
    if isinstance(image, (str, Path)):
        img_path = Path(image).resolve()
        if not img_path.exists() or not img_path.is_file():
            raise FileNotFoundError(
                f"\n[ERROR] Input image file not found at: '{img_path}'"
            )

        ext = img_path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"\n[ERROR] Unsupported image format '{ext}' for file: '{img_path}'.\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            )

        try:
            pil_img = Image.open(img_path)
            pil_img.load()  # Verify image can be decoded
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"\n[ERROR] Unsupported or corrupted image file at: '{img_path}'"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"\n[ERROR] Failed to read image file at '{img_path}': {exc}"
            ) from exc
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise ValueError(
            f"\n[ERROR] Expected image as file path (str, Path) or PIL.Image, received: {type(image)}"
        )

    # Convert to 3-channel RGB (handles grayscale 'L', 1-bit '1', RGBA, CMYK, Palette 'P')
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    return pil_img


def preprocess_image(
    image: Union[str, Path, Image.Image],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """
    Validates and preprocesses a single image for MobileNetV3 inference.

    Steps:
    1. Validates and converts image to 3-channel RGB PIL Image.
    2. Resizes image to target resolution (224x224).
    3. Converts image to PyTorch FloatTensor in range [0, 1].
    4. Normalizes with standard ImageNet mean and std.
    5. Prepends batch dimension -> shape (1, 3, 224, 224).

    Args:
        image: File path (str or Path) or PIL Image instance.
        image_size: Target (height, width) resolution.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
    """
    pil_img = validate_image(image)

    transform_pipeline = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor: torch.Tensor = transform_pipeline(pil_img)

    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)
    return tensor


# =====================================================================
# Clinical Recommendations & Confidence Thresholding
# =====================================================================
def get_recommendation(
    predicted_class: str,
    confidence: float,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[str, bool]:
    """
    Generates a clinically grounded triage recommendation based on predicted risk level
    and assesses confidence threshold status.

    Args:
        predicted_class: Predicted class label (e.g. 'normal', 'moderate_risk', 'high_risk').
        confidence: Prediction confidence score between 0.0 and 1.0.
        confidence_threshold: Minimum threshold required for high-confidence assessment.

    Returns:
        Tuple[str, bool]: (recommended_action, confidence_warning)
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

    confidence_warning = confidence < confidence_threshold
    if confidence_warning:
        full_recommendation = (
            f"{base_recommendation} Low confidence prediction. "
            "Interpret alongside full clinical assessment."
        )
    else:
        full_recommendation = base_recommendation

    return full_recommendation, confidence_warning


# =====================================================================
# Single Sample Inference API with Multi-Run Latency Benchmarking
# =====================================================================
def predict_single_sample(
    image: Union[str, Path, Image.Image],
    clinical_data: Optional[Dict[str, Any]] = None,
    model: Optional[nn.Module] = None,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Union[Path, str] = "models/checkpoints/best_baseline_model.pth",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
) -> Dict[str, Any]:
    """
    Central production inference engine API for neonatal respiratory triage.

    Executes full pipeline:
    1. Pre-execution warm-up passes (2 runs) to compile GPU/MPS kernels.
    2. Synchronized preprocessing & validation latency measurement.
    3. Multi-run (20 runs) synchronized steady-state inference benchmark.
    4. Postprocessing (class resolution, clinical recommendation, confidence check).
    5. Computes statistical summary (mean, min, max, std) for inference and total pipeline.
    6. Generates a clean, JSON-serializable structured dictionary.

    Args:
        image: File path or PIL Image instance.
        clinical_data: Optional clinical vitals dictionary (reserved for future multimodal fusion).
        model: Optional preloaded PyTorch model instance for batch/API reuse.
        class_names: Optional preloaded class label list.
        device: Optional preloaded compute device.
        checkpoint_path: Path to checkpoint if model needs to be loaded.
        confidence_threshold: Confidence threshold for clinical alerts.
        warmup_runs: Number of unmeasured warm-up forward passes to execute (default: 2).
        benchmark_runs: Number of measured steady-state inference runs (default: 20).

    Returns:
        Dict[str, Any]: Structured, JSON-serializable prediction result dictionary:
            - predicted_class: str (e.g. "moderate_risk")
            - display_label: str (e.g. "Moderate Risk")
            - confidence: float (e.g. 0.4409)
            - confidence_warning: bool (True if confidence < threshold)
            - probabilities: Dict[str, float]
            - recommended_action: str
            - latency_ms: Dict[str, float] (preprocess, inference, postprocess, total)
            - benchmark: Dict[str, Any] (warmup_runs, benchmark_runs, device_synchronized, inference stats, total_pipeline stats)
            - device: str
            - model_version: str
            - timestamp: str
    """
    # 1. Model Resolution
    if model is None or class_names is None or device is None:
        model, class_names, model_version, device = load_trained_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    else:
        model_version = MODEL_VERSION

    # 2. Warm-up forward passes (Not included in reported benchmark statistics)
    if warmup_runs > 0:
        warmup_tensor = preprocess_image(image=image).to(device)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(warmup_tensor)
                synchronize_device(device)

    # 3. Image Preprocessing & Latency Measurement
    synchronize_device(device)
    t_pre_start = time.perf_counter()
    input_tensor = preprocess_image(image=image)
    input_tensor = input_tensor.to(device)
    synchronize_device(device)
    t_pre_end = time.perf_counter()
    preprocess_ms = (t_pre_end - t_pre_start) * 1000.0

    # 4. Multi-Run Steady-State Inference Benchmarking
    num_runs = max(benchmark_runs, 1)
    inference_times_ms: List[float] = []
    probabilities_tensor: Optional[torch.Tensor] = None

    with torch.no_grad():
        for _ in range(num_runs):
            synchronize_device(device)
            t_inf_start = time.perf_counter()
            logits = model(input_tensor)
            probabilities_tensor = F.softmax(logits, dim=1)
            synchronize_device(device)
            t_inf_end = time.perf_counter()
            inference_times_ms.append((t_inf_end - t_inf_start) * 1000.0)

    # 5. Post-processing & Latency Measurement
    synchronize_device(device)
    t_post_start = time.perf_counter()

    probs_np = probabilities_tensor.cpu().squeeze(0).numpy()
    pred_index = int(probs_np.argmax())
    predicted_class = str(class_names[pred_index])
    confidence = float(probs_np[pred_index])
    display_label = predicted_class.replace("_", " ").title()

    probabilities_dict: Dict[str, float] = {
        str(class_name): round(float(probs_np[idx]), 4)
        for idx, class_name in enumerate(class_names)
    }

    # Clinical Recommendation & Confidence Warning
    recommendation, confidence_warning = get_recommendation(
        predicted_class=predicted_class,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    synchronize_device(device)
    t_post_end = time.perf_counter()
    postprocess_ms = (t_post_end - t_post_start) * 1000.0

    # 6. Compute Total Pipeline Times for each benchmark iteration
    total_pipeline_times_ms = [
        preprocess_ms + inf_time + postprocess_ms
        for inf_time in inference_times_ms
    ]

    # 7. Statistical Aggregations
    inference_stats = compute_latency_stats(inference_times_ms)
    total_pipeline_stats = compute_latency_stats(total_pipeline_times_ms)

    legacy_latency_dict: Dict[str, float] = {
        "preprocess": round(preprocess_ms, 2),
        "inference": inference_stats["average_ms"],
        "postprocess": round(postprocess_ms, 2),
        "total": total_pipeline_stats["average_ms"],
    }

    benchmark_metadata: Dict[str, Any] = {
        "warmup_runs": warmup_runs,
        "benchmark_runs": num_runs,
        "device_synchronized": True,
        "inference": inference_stats,
        "total_pipeline": total_pipeline_stats,
    }

    return {
        "predicted_class": predicted_class,
        "display_label": display_label,
        "confidence": round(confidence, 4),
        "confidence_warning": confidence_warning,
        "probabilities": probabilities_dict,
        "recommended_action": recommendation,
        "latency_ms": legacy_latency_dict,
        "benchmark": benchmark_metadata,
        "device": str(device),
        "model_version": model_version,
        "timestamp": timestamp,
    }


# =====================================================================
# Console Summary Presentation
# =====================================================================
def print_prediction_summary(result: Dict[str, Any]) -> None:
    """
    Formats and prints a comprehensive, human-readable prediction report to the console.

    Args:
        result: Structured dictionary returned by predict_single_sample().
    """
    print("=" * 65)
    print("                 Neonatal Triage Prediction Summary")
    print("=" * 65)

    print(f"Predicted Class           : {result['display_label']}")
    print(
        f"Confidence                : {result['confidence']:.4f} "
        f"({result['confidence'] * 100:.2f}%)"
    )

    if result.get("confidence_warning", False):
        print("Confidence Warning        : ⚠️ Low confidence (< 60.00%)")
    else:
        print("Confidence Warning        : None (High confidence)")

    print("\nClass Probabilities:")
    for class_name, prob in result["probabilities"].items():
        display_class = class_name.replace("_", " ").title()
        print(f"  - {display_class:<18}: {prob:.4f} ({prob * 100:.2f}%)")

    print(f"\nClinical Recommendation   : {result['recommended_action']}")

    lat = result.get("latency_ms", {})
    bm = result.get("benchmark", {})
    inf_stats = bm.get("inference", {})
    tot_stats = bm.get("total_pipeline", {})

    print("\n" + "=" * 50)
    print("Latency Benchmark")
    print("=" * 50)

    print(f"Warm-up Runs           : {bm.get('warmup_runs', 0)}")
    print(f"Benchmark Runs         : {bm.get('benchmark_runs', 0)}")
    print(f"Device Synchronization : {'Enabled' if bm.get('device_synchronized', False) else 'Disabled'}\n")

    print(f"Preprocessing          : {lat.get('preprocess', 0.0):.2f} ms")
    print(f"Post-processing        : {lat.get('postprocess', 0.0):.2f} ms\n")

    print("Inference Statistics\n")
    print(f"Average               : {inf_stats.get('average_ms', 0.0):.2f} ms")
    print(f"Minimum               : {inf_stats.get('minimum_ms', 0.0):.2f} ms")
    print(f"Maximum               : {inf_stats.get('maximum_ms', 0.0):.2f} ms")
    print(f"Std. Dev.             : {inf_stats.get('std_ms', 0.0):.2f} ms\n")

    print("Total Pipeline Statistics\n")
    print(f"Average               : {tot_stats.get('average_ms', 0.0):.2f} ms")
    print(f"Minimum               : {tot_stats.get('minimum_ms', 0.0):.2f} ms")
    print(f"Maximum               : {tot_stats.get('maximum_ms', 0.0):.2f} ms")
    print(f"Std. Dev.             : {tot_stats.get('std_ms', 0.0):.2f} ms")
    print("=" * 50)

    print(f"\nCompute Device            : {result.get('device', 'N/A')}")
    print(f"Model Version             : {result.get('model_version', 'N/A')}")
    print(f"Timestamp                 : {result.get('timestamp', 'N/A')}")
    print("=" * 65)


# =====================================================================
# Sample Image Discovery Helper
# =====================================================================
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

    # Search for any PNG file inside data/raw if specific samples are not found
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_dir.exists():
        found = list(raw_dir.glob("*/*.png"))
        if found:
            return found[0].resolve()

    raise FileNotFoundError(
        "\n[ERROR] No test sample image found in candidate paths:\n"
        + "\n".join(f"  - {c}" for c in search_candidates)
        + "\nPlease generate the mock dataset first by running:\n"
        "    python src/generate_mock_data.py"
    )


# =====================================================================
# Main Execution Entry Point
# =====================================================================
def main() -> None:
    """Main execution function for standalone inference testing."""
    sample_image_path = locate_sample_image()
    rel_sample = sample_image_path.relative_to(PROJECT_ROOT) if sample_image_path.is_relative_to(PROJECT_ROOT) else sample_image_path
    print(f"[INFO] Performing inference on sample image: {rel_sample}")

    result = predict_single_sample(image=sample_image_path)
    print()
    print_prediction_summary(result)
    print("\n[SUCCESS] Prediction pipeline completed successfully.\n")


if __name__ == "__main__":
    main()