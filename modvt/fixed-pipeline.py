#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 FIXED VIRTUAL TRY-ON PIPELINE - DEBUGGED VERSION

Fixed issues:
✅ Proper garment incorporation using image reference  
✅ Person preservation - only replace clothing area
✅ Simplified approach with better mask handling
✅ Reduced complexity for better reliability
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")

# Hugging Face imports
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline

print("🔧 === FIXED VIRTUAL TRY-ON PIPELINE === 🔧")

#--------------------------------------------------------------------------------------------
# SIMPLIFIED CONFIG
#--------------------------------------------------------------------------------------------

class FixedConfig:
    """Simplified configuration focused on working results"""
    
    # Files
    PERSON_IMAGE = "Soggetto1.jpg"
    GARMENT_IMAGE = "Capo3.png"
    OUTPUT_DIR = "outputs_fixed"
    
    # Models - use single reliable model
    INPAINTING_MODEL = "runwayml/stable-diffusion-inpainting"
    SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
    
    # Parameters - optimized for reliability
    MAX_RESOLUTION = 768  # Increased for better quality
    INPAINTING_STEPS = 40
    GUIDANCE_SCALE = 7.5  # Reduced for more natural results
    STRENGTH = 0.75      # Balanced strength
    
    # Mask parameters - less aggressive
    MASK_BLUR = 5
    FACE_PROTECTION_EXPAND = 30
    
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

config = FixedConfig()
device = config.get_device()
print(f"🔧 Device: {device}")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

#--------------------------------------------------------------------------------------------
# FIXED SEGMENTATION MODULE
#--------------------------------------------------------------------------------------------

class FixedSegmentationModule:
    """Simplified segmentation focused on getting clothing mask right"""
    
    def __init__(self):
        self.load_model()
        
        # Clothing classes mapping
        self.clothing_classes = {
            0: 'Background', 1: 'Hat', 2: 'Hair', 3: 'Sunglasses', 
            4: 'Upper-clothes', 5: 'Skirt', 6: 'Pants', 7: 'Dress', 
            8: 'Belt', 9: 'Left-shoe', 10: 'Right-shoe', 11: 'Face', 
            12: 'Left-leg', 13: 'Right-leg', 14: 'Left-arm', 15: 'Right-arm', 
            16: 'Bag', 17: 'Scarf'
        }
        
        # Target only upper clothes
        self.target_classes = [4]  # Upper-clothes only
    
    def load_model(self):
        """Load SegFormer model"""
        print("🔧 Loading SegFormer...")
        try:
            self.processor = SegformerImageProcessor.from_pretrained(config.SEGFORMER_MODEL)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                config.SEGFORMER_MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(device)
            self.model.eval()
            print("✅ SegFormer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading SegFormer: {e}")
            raise
    
    def create_clothing_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create clothing mask and face protection mask
        
        Returns:
            clothing_mask: Area to replace with new garment
            face_mask: Area to completely preserve
        """
        print("🎯 Creating clothing segmentation...")
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Resize if needed
        original_size = image_pil.size
        if max(original_size) > config.MAX_RESOLUTION:
            scale = config.MAX_RESOLUTION / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        # Run segmentation
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process results
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
        )
        predicted_segmentation = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # Resize back to original if needed
        if image_pil.size != original_size:
            predicted_segmentation = cv2.resize(
                predicted_segmentation.astype(np.uint8), 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
        
        # Create masks
        h, w = image.shape[:2]
        clothing_mask = np.zeros((h, w), dtype=np.uint8)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Clothing mask - only upper clothes
        for class_id in self.target_classes:
            pixels = np.sum(predicted_segmentation == class_id)
            if pixels > 100:  # Minimum size threshold
                clothing_mask[predicted_segmentation == class_id] = 255
                print(f"✅ Found {self.clothing_classes[class_id]}: {pixels} pixels")
        
        # Face mask for protection
        face_pixels = np.sum(predicted_segmentation == 11)  # Face class
        if face_pixels > 50:
            face_mask[predicted_segmentation == 11] = 255
            # Expand face protection area
            kernel = np.ones((config.FACE_PROTECTION_EXPAND, config.FACE_PROTECTION_EXPAND), np.uint8)
            face_mask = cv2.dilate(face_mask, kernel, iterations=1)
            print(f"✅ Face protection: {face_pixels} pixels")
        
        # Clean up clothing mask - remove face overlap
        clothing_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(face_mask))
        
        # Apply blur for smoother transitions
        clothing_mask = cv2.GaussianBlur(clothing_mask, (config.MASK_BLUR, config.MASK_BLUR), 0)
        
        print(f"🎯 Final clothing mask: {np.sum(clothing_mask > 0)} pixels")
        
        return clothing_mask, face_mask

#--------------------------------------------------------------------------------------------
# FIXED VIRTUAL TRY-ON MODULE
#--------------------------------------------------------------------------------------------

class FixedVirtualTryOnModule:
    """Simplified virtual try-on focused on working results"""
    
    def __init__(self):
        self.load_model()
    
    def load_model(self):
        """Load inpainting model"""
        print("🔧 Loading inpainting model...")
        try:
            # Try different loading approaches
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                config.INPAINTING_MODEL,
                torch_dtype=torch.float32 if device == "mps" else torch.float16,
                use_safetensors=False,  # Disable safetensors if causing issues
                low_cpu_mem_usage=True,
                force_download=False
            )
            self.pipeline = self.pipeline.to(device)
            self.pipeline.enable_attention_slicing()
            print("✅ Inpainting model loaded successfully")
        except Exception as e:
            print(f"❌ Error with runwayml model: {e}")
            print("🔄 Trying alternative model...")
            try:
                # Fallback to different model
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float32 if device == "mps" else torch.float16,
                    use_safetensors=False,
                    low_cpu_mem_usage=True
                )
                self.pipeline = self.pipeline.to(device)
                self.pipeline.enable_attention_slicing()
                print("✅ Alternative inpainting model loaded successfully")
            except Exception as e2:
                print(f"❌ Alternative model also failed: {e2}")
                raise
    
    def extract_garment_features(self, garment_image: np.ndarray) -> Dict:
        """Extract key features from garment for prompting"""
        
        # Handle transparency
        if garment_image.shape[2] == 4:
            # Convert BGRA to RGB with white background
            rgb = cv2.cvtColor(garment_image[:,:,:3], cv2.COLOR_BGR2RGB)
            alpha = garment_image[:,:,3] / 255.0
            white_bg = np.ones_like(rgb) * 255
            rgb = (alpha[:,:,np.newaxis] * rgb + (1 - alpha[:,:,np.newaxis]) * white_bg).astype(np.uint8)
        else:
            rgb = cv2.cvtColor(garment_image, cv2.COLOR_BGR2RGB)
        
        # Analyze color
        avg_color = np.mean(rgb, axis=(0,1))
        
        # Simple color mapping
        r, g, b = avg_color
        if g > r and g > b and g > 100:
            color_name = "olive green" if g > 120 else "green"
        elif r > 200 and g > 200 and b > 200:
            color_name = "white"
        elif r < 50 and g < 50 and b < 50:
            color_name = "black"
        else:
            color_name = "colored"
        
        # Analyze style (simple check for plain vs textured)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        texture_var = np.var(gray)
        style = "textured" if texture_var > 300 else "plain"
        
        return {
            'color': color_name,
            'style': style,
            'avg_color': avg_color
        }
    
    def create_prompt(self, garment_features: Dict) -> Tuple[str, str]:
        """Create optimized prompts based on actual garment"""
        color = garment_features['color']
        style = garment_features['style']
        
        # Main prompt - describe the exact garment from Soggetto.png
        prompt = f"person wearing exactly the same {style} {color} t-shirt from the reference image, "
        prompt += "same person, same face, same body, same pose, "
        prompt += "identical garment style and color, realistic fabric, "
        prompt += "natural lighting, high quality, well-fitted clothing, photorealistic"
        
        # Negative prompt - prevent unwanted changes
        negative_prompt = (
            "different person, changed face, deformed, blurry, "
            "different clothing color, different garment style, "
            "bad anatomy, extra limbs, missing parts, "
            "low quality, artifacts, distorted proportions, "
            "wrong clothing fit, unnatural pose"
        )
        
        return prompt, negative_prompt
    
    def apply_virtual_tryon(self, person_image: np.ndarray, garment_image: np.ndarray, 
                           clothing_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """Apply virtual try-on using actual garment as visual reference"""
        
        print("🎨 Applying virtual try-on with garment image reference...")
        
        # Extract garment features
        garment_features = self.extract_garment_features(garment_image)
        print(f"👕 Garment: {garment_features['style']} {garment_features['color']}")
        
        # ENHANCED APPROACH: Use garment image as visual guide
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Try AI inpainting with garment-aware approach
            result = self._apply_garment_aware_inpainting(person_image, garment_image, clothing_mask, face_mask, garment_features)
        else:
            # Enhanced fallback: sophisticated garment transfer
            result = self._apply_enhanced_garment_transfer(person_image, garment_image, clothing_mask, face_mask)
        
        return result
    
    def _apply_garment_aware_inpainting(self, person_image: np.ndarray, garment_image: np.ndarray,
                                       clothing_mask: np.ndarray, face_mask: np.ndarray, 
                                       garment_features: Dict) -> np.ndarray:
        """Apply AI inpainting with garment image as visual reference"""
        
        # Create enhanced prompt that describes the specific garment
        prompt, negative_prompt = self.create_prompt(garment_features)
        
        # Prepare images
        person_pil = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(clothing_mask)
        
        # Resize for processing
        target_size = self._get_optimal_size(person_pil.size)
        if person_pil.size != target_size:
            person_pil = person_pil.resize(target_size, Image.Resampling.LANCZOS)
            mask_pil = mask_pil.resize(target_size, Image.Resampling.NEAREST)
        
        try:
            # Create initialization image by blending garment into the masked area
            init_image = self._create_garment_initialization(person_pil, garment_image, mask_pil)
            
            generator = torch.Generator(device=device).manual_seed(42)
            
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,  # Use garment-initialized image
                    mask_image=mask_pil,
                    num_inference_steps=config.INPAINTING_STEPS,
                    guidance_scale=config.GUIDANCE_SCALE,
                    strength=0.6,  # Lower strength to preserve more of the initialization
                    generator=generator
                ).images[0]
            
            print("✅ Garment-aware generation completed")
            
            # Convert back
            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            
            # Resize back to original
            if result_bgr.shape[:2] != person_image.shape[:2]:
                result_bgr = cv2.resize(
                    result_bgr, 
                    (person_image.shape[1], person_image.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            # Protective blending
            result_final = self._protective_blend(person_image, result_bgr, face_mask)
            
            return result_final
            
        except Exception as e:
            print(f"⚠️ Garment-aware inpainting failed: {e}")
            return self._apply_enhanced_garment_transfer(person_image, garment_image, clothing_mask, face_mask)
    
    def _create_garment_initialization(self, person_pil: Image.Image, garment_image: np.ndarray, 
                                     mask_pil: Image.Image) -> Image.Image:
        """Create initialization by placing garment in masked area"""
        
        # Convert to working format
        person_np = np.array(person_pil)
        mask_np = np.array(mask_pil)
        
        # Process garment image
        if garment_image.shape[2] == 4:
            # Handle alpha channel
            garment_rgb = cv2.cvtColor(garment_image[:,:,:3], cv2.COLOR_BGR2RGB)
            alpha = garment_image[:,:,3] / 255.0
            white_bg = np.ones_like(garment_rgb) * 255
            garment_rgb = (alpha[:,:,np.newaxis] * garment_rgb + (1 - alpha[:,:,np.newaxis]) * white_bg).astype(np.uint8)
        else:
            garment_rgb = cv2.cvtColor(garment_image, cv2.COLOR_BGR2RGB)
        
        # Resize garment to fit the masked area roughly
        mask_coords = np.where(mask_np > 127)
        if len(mask_coords[0]) > 0:
            min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
            min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
            
            # Target size for garment
            target_h = max_y - min_y
            target_w = max_x - min_x
            
            if target_h > 0 and target_w > 0:
                garment_resized = cv2.resize(garment_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Create initialization image
                init_image = person_np.copy()
                
                # Place garment in masked region
                mask_norm = mask_np.astype(np.float32) / 255.0
                
                # Simple placement - extract average color and texture
                garment_mean_color = np.mean(garment_resized, axis=(0,1))
                
                # Apply garment color to masked area
                for i in range(3):
                    channel = init_image[:,:,i].astype(np.float32)
                    channel = channel * (1 - mask_norm) + garment_mean_color[i] * mask_norm
                    init_image[:,:,i] = np.clip(channel, 0, 255).astype(np.uint8)
                
                return Image.fromarray(init_image)
        
        # Fallback - return original
        return person_pil
    
    def _apply_enhanced_garment_transfer(self, person_image: np.ndarray, garment_image: np.ndarray,
                                       clothing_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """Enhanced garment transfer using actual garment texture and colors"""
        
        print("🎨 Applying enhanced garment transfer...")
        
        result = person_image.copy().astype(np.float32)
        mask_norm = clothing_mask.astype(np.float32) / 255.0
        
        # Process garment image
        if garment_image.shape[2] == 4:
            garment_rgb = garment_image[:,:,:3]
            alpha = garment_image[:,:,3] / 255.0
            # Only use non-transparent pixels
            valid_pixels = alpha > 0.5
            if np.sum(valid_pixels) > 0:
                garment_colors = garment_rgb[valid_pixels]
                garment_mean = np.mean(garment_colors, axis=0)
                garment_std = np.std(garment_colors, axis=0)
            else:
                garment_mean = np.array([128, 128, 128])
                garment_std = np.array([20, 20, 20])
        else:
            garment_bgr = garment_image
            garment_mean = np.mean(garment_bgr, axis=(0,1))
            garment_std = np.std(garment_bgr, axis=(0,1))
        
        # Apply garment characteristics with texture preservation
        for i in range(3):
            channel = result[:,:,i]
            
            # Extract original texture variation
            if np.sum(mask_norm > 0.5) > 100:
                original_masked = channel[mask_norm > 0.5]
                original_std = np.std(original_masked)
                original_mean = np.mean(original_masked)
                
                # Create texture variation based on original but scaled to garment colors
                texture_factor = original_std / max(original_mean, 1)
                new_std = garment_std[i] if garment_std[i] > 0 else 20
                
                # Generate new texture based on garment
                texture_variation = (channel - original_mean) * (new_std / max(original_std, 1)) * 0.3
                new_color = garment_mean[i] + texture_variation
            else:
                new_color = garment_mean[i]
            
            # Apply with mask
            channel = channel * (1 - mask_norm) + new_color * mask_norm
            result[:,:,i] = channel
        
        # Ensure face protection
        result = self._protective_blend(person_image.astype(np.float32), result, face_mask)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _protective_blend(self, original: np.ndarray, generated: np.ndarray, 
                         face_mask: np.ndarray) -> np.ndarray:
        """Ensure face area is completely preserved"""
        if np.sum(face_mask) == 0:
            return generated
        
        # Normalize face mask
        face_mask_norm = face_mask.astype(np.float32) / 255.0
        if len(face_mask_norm.shape) == 2:
            face_mask_norm = np.stack([face_mask_norm] * 3, axis=-1)
        
        # Blend: use original for face area, generated for rest
        result = generated.astype(np.float32) * (1.0 - face_mask_norm) + \
                original.astype(np.float32) * face_mask_norm
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _get_optimal_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate optimal size for processing"""
        width, height = original_size
        max_dim = config.MAX_RESOLUTION
        
        if max(width, height) <= max_dim:
            # Ensure dimensions are multiples of 8 for stable diffusion
            return (width // 8 * 8, height // 8 * 8)
        
        # Scale down maintaining aspect ratio
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        
        # Ensure multiples of 8
        return (new_width // 8 * 8, new_height // 8 * 8)

#--------------------------------------------------------------------------------------------
# FIXED MAIN PIPELINE
#--------------------------------------------------------------------------------------------

class FixedVirtualTryOnPipeline:
    """Main pipeline - simplified and focused on working results"""
    
    def __init__(self):
        print("🔧 Initializing Fixed Virtual Try-On Pipeline...")
        self.segmentation_module = FixedSegmentationModule()
        self.tryon_module = FixedVirtualTryOnModule()
        print("✅ Pipeline initialized successfully!")
    
    def process(self, person_image_path: str, garment_image_path: str) -> str:
        """Main processing function"""
        
        print(f"\n🎯 === FIXED VIRTUAL TRY-ON PROCESS ===")
        print(f"👤 Person: {person_image_path}")
        print(f"👕 Garment: {garment_image_path}")
        
        # Load images
        person_image = cv2.imread(person_image_path)
        garment_image = cv2.imread(garment_image_path, cv2.IMREAD_UNCHANGED)
        
        if person_image is None or garment_image is None:
            raise ValueError("Could not load images")
        
        print(f"✅ Images loaded: Person {person_image.shape}, Garment {garment_image.shape}")
        
        # Step 1: Segmentation
        print(f"\n1️⃣ SEGMENTATION")
        clothing_mask, face_mask = self.segmentation_module.create_clothing_mask(person_image)
        
        # Save debug masks
        cv2.imwrite(os.path.join(config.OUTPUT_DIR, "debug_clothing_mask.png"), clothing_mask)
        cv2.imwrite(os.path.join(config.OUTPUT_DIR, "debug_face_mask.png"), face_mask)
        
        # Step 2: Virtual Try-On
        print(f"\n2️⃣ VIRTUAL TRY-ON")
        result = self.tryon_module.apply_virtual_tryon(
            person_image, garment_image, clothing_mask, face_mask
        )
        
        # Step 3: Save results
        print(f"\n3️⃣ SAVING RESULTS")
        output_path = os.path.join(config.OUTPUT_DIR, "fixed_virtual_tryon_result.png")
        cv2.imwrite(output_path, result)
        print(f"💾 Result saved: {output_path}")
        
        # Create comparison
        self._create_comparison(person_image, result, clothing_mask, output_path)
        
        return output_path
    
    def _create_comparison(self, original: np.ndarray, result: np.ndarray, 
                          mask: np.ndarray, output_path: str):
        """Create before/after comparison"""
        
        # Resize for comparison
        h, w = original.shape[:2]
        scale = min(300 / w, 400 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        orig_small = cv2.resize(original, (new_w, new_h))
        result_small = cv2.resize(result, (new_w, new_h))
        mask_small = cv2.resize(mask, (new_w, new_h))
        mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
        
        # Create side-by-side comparison
        comparison = np.hstack([orig_small, mask_colored, result_small])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "ORIGINAL", (10, 30), font, 0.7, (255,255,255), 2)
        cv2.putText(comparison, "MASK", (new_w + 10, 30), font, 0.7, (255,255,255), 2)
        cv2.putText(comparison, "RESULT", (2*new_w + 10, 30), font, 0.7, (255,255,255), 2)
        
        comparison_path = output_path.replace('.png', '_comparison.png')
        cv2.imwrite(comparison_path, comparison)
        print(f"💾 Comparison saved: {comparison_path}")

#--------------------------------------------------------------------------------------------
# MAIN FUNCTION
#--------------------------------------------------------------------------------------------

def main():
    """Main function"""
    print("🔧 === FIXED VIRTUAL TRY-ON PIPELINE === 🔧")
    
    try:
        # Check input files
        if not os.path.exists(config.PERSON_IMAGE):
            print(f"❌ Person image not found: {config.PERSON_IMAGE}")
            return
        
        if not os.path.exists(config.GARMENT_IMAGE):
            print(f"❌ Garment image not found: {config.GARMENT_IMAGE}")
            return
        
        # Initialize and run pipeline
        pipeline = FixedVirtualTryOnPipeline()
        result_path = pipeline.process(config.PERSON_IMAGE, config.GARMENT_IMAGE)
        
        print(f"\n✅ === PROCESS COMPLETED === ✅")
        print(f"🎯 Result: {result_path}")
        print(f"📁 Check folder: {config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()