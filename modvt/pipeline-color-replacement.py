#!/usr/bin/env python3
"""
Quick test to verify the segmentation works and create basic mask blending
"""

import cv2
import numpy as np
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch

print("🔧 Quick Test - Segmentation + Basic Blending")

# Check device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load images
person_img = cv2.imread("tst.jpg")
garment_img = cv2.imread("Soggetto.png", cv2.IMREAD_UNCHANGED)

if person_img is None or garment_img is None:
    print("❌ Could not load images")
    exit()

print(f"✅ Images loaded: Person {person_img.shape}, Garment {garment_img.shape}")

# Load segmentation model
print("Loading SegFormer...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = model.to(device)
model.eval()
print("✅ SegFormer loaded")

# Create segmentation
print("Creating segmentation...")
image_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
from PIL import Image
image_pil = Image.fromarray(image_rgb)

inputs = processor(images=image_pil, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits.cpu()
upsampled_logits = torch.nn.functional.interpolate(
    logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
)
predicted_segmentation = upsampled_logits.argmax(dim=1)[0].numpy()

print("✅ Segmentation completed")

# Create clothing mask (class 4 = upper clothes)
clothing_mask = np.zeros_like(predicted_segmentation, dtype=np.uint8)
clothing_pixels = np.sum(predicted_segmentation == 4)
if clothing_pixels > 100:
    clothing_mask[predicted_segmentation == 4] = 255
    print(f"✅ Upper clothes found: {clothing_pixels} pixels")
else:
    print("❌ No upper clothes detected")

# Create face mask (class 11 = face)
face_mask = np.zeros_like(predicted_segmentation, dtype=np.uint8)
face_pixels = np.sum(predicted_segmentation == 11)
if face_pixels > 50:
    face_mask[predicted_segmentation == 11] = 255
    # Expand face protection
    kernel = np.ones((30, 30), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel, iterations=1)
    print(f"✅ Face protection: {face_pixels} pixels")

# Remove face from clothing mask
clothing_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(face_mask))

# Blur mask for smooth transitions
clothing_mask = cv2.GaussianBlur(clothing_mask, (5, 5), 0)

print(f"Final clothing mask: {np.sum(clothing_mask > 0)} pixels")

# Simple color-based garment blending
def extract_garment_color(garment_img):
    """Extract average color from garment"""
    if garment_img.shape[2] == 4:
        # Handle alpha channel
        alpha = garment_img[:,:,3] / 255.0
        rgb = garment_img[:,:,:3]
        # Only consider non-transparent pixels
        valid_mask = alpha > 0.5
        if np.sum(valid_mask) > 0:
            avg_color = np.mean(rgb[valid_mask], axis=0)
        else:
            avg_color = [128, 128, 128]  # Gray fallback
    else:
        avg_color = np.mean(garment_img, axis=(0,1))
    
    return avg_color[::-1]  # BGR to RGB

garment_color = extract_garment_color(garment_img)
print(f"✅ Garment color extracted: {garment_color}")

# Create simple replacement
result = person_img.copy()
mask_norm = clothing_mask.astype(np.float32) / 255.0

# Apply garment color to clothing area
for i in range(3):
    channel = result[:,:,i].astype(np.float32)
    channel = channel * (1 - mask_norm) + garment_color[i] * mask_norm
    result[:,:,i] = np.clip(channel, 0, 255).astype(np.uint8)

# Create output directory
os.makedirs("outputs_fixed", exist_ok=True)

# Save results
cv2.imwrite("outputs_fixed/debug_clothing_mask.png", clothing_mask)
cv2.imwrite("outputs_fixed/debug_face_mask.png", face_mask)
cv2.imwrite("outputs_fixed/quick_test_result.png", result)

# Create comparison
h, w = person_img.shape[:2]
scale = min(300 / w, 400 / h)
new_w, new_h = int(w * scale), int(h * scale)

orig_small = cv2.resize(person_img, (new_w, new_h))
result_small = cv2.resize(result, (new_w, new_h))
mask_small = cv2.resize(clothing_mask, (new_w, new_h))
mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)

comparison = np.hstack([orig_small, mask_colored, result_small])

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "ORIGINAL", (10, 30), font, 0.7, (255,255,255), 2)
cv2.putText(comparison, "MASK", (new_w + 10, 30), font, 0.7, (255,255,255), 2)
cv2.putText(comparison, "RESULT", (2*new_w + 10, 30), font, 0.7, (255,255,255), 2)

cv2.imwrite("outputs_fixed/quick_test_comparison.png", comparison)

print("✅ Quick test completed!")
print("📁 Check outputs_fixed/ for results")
print("🎯 This shows if segmentation works - next step is proper inpainting")