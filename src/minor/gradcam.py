import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

import sys
# Make sure we can import from src/minor
sys.path.append(str(Path(__file__).resolve().parent))

from infer import load_model, transform, CLASS_NAMES, DEVICE

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIS_DIR = PROJECT_ROOT / "models/minor/visualizations"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, logits

def generate_overlay(image_path, cam):
    original_img = cv2.imread(str(image_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 0.6 image, 0.4 heatmap to ensure original image is visible
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(overlay)

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for Minor LUS")
    parser.add_argument("image", type=str, help="Path to a lung ultrasound image")
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    model = load_model()
    model.eval()

    target_layer = model.backbone.features[-1]
    
    grad_cam = GradCAM(model, target_layer)
    
    cam, logits = grad_cam(tensor)
    
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_class = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_class].item())
    pred_label = CLASS_NAMES[predicted_class]
    
    overlay_img = generate_overlay(image_path, cam)
    
    filename = f"{image_path.stem}_gradcam.jpg"
    out_path = VIS_DIR / filename
    overlay_img.save(out_path)
    
    print("\n" + "=" * 50)
    print("OpenPOCUS Minor LUS Grad-CAM")
    print("=" * 50)
    print(f"Device:            {DEVICE}")
    print(f"Image:             {image_path}")
    print(f"Predicted Class:   {pred_label}")
    print(f"Confidence:        {confidence * 100:.2f}%")
    print(f"Visualization:     {out_path}")
    print("=" * 50)
    print("\nGrad-CAM is an interpretability visualization and does not establish clinical validity or diagnostic correctness.")

if __name__ == "__main__":
    main()
