#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 PIPELINE VIRTUAL TRY-ON MODULARE - HUGGING FACE MODELS 🚀

Pipeline modulare che utilizza i migliori modelli Hugging Face per virtual try-on:
- SegFormer B2 per segmentazione clothing precisa
- Stable Diffusion Inpainting con IP-Adapter via T2IAdapter
- Ottimizzazioni specifiche per Mac M2 Pro (16GB RAM)

Autore: Assistente AI
Ottimizzato per: MacBook Pro M2 Pro, 16GB RAM
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Hugging Face imports
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline, T2IAdapter

#--------------------------------------------------------------------------------------------
# CONFIGURAZIONE E COSTANTI
#--------------------------------------------------------------------------------------------
class Config:
    PERSON_IMAGE    = "tst2.jpg"
    GARMENT_IMAGE   = "Soggetto.png"
    OUTPUT_DIR      = "outputs"

    SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
    INPAINT_MODEL   = "stabilityai/stable-diffusion-2-inpainting"
    IP_ADAPTER      = "shi-labs/ip-adapter-vit-b-16"
    MAX_RESOLUTION  = 768

    INPAINT_STEPS   = 30
    GUIDANCE_SCALE  = 7.5
    STRENGTH        = 0.99

    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

# Initial setup
config = Config()
device = config.get_device()
dtype = torch.float32 if device.type == "mps" else (torch.float16 if device.type == "cuda" else torch.float32)
if device.type == "mps": print("🍎 MPS rilevato: forzando float32 per compatibilità")
print(f"🚀 === PIPELINE VIRTUAL TRY-ON MODULARE === 🚀")
print(f"Device: {device}, Dtype: {dtype}, Max Res: {config.MAX_RESOLUTION}\n")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

#--------------------------------------------------------------------------------------------
# SEGMENTATION MODULE
#--------------------------------------------------------------------------------------------
class ClothingSegmentationModule:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained(config.SEGFORMER_MODEL)
        self.model     = SegformerForSemanticSegmentation.from_pretrained(
            config.SEGFORMER_MODEL,
            torch_dtype=torch.float32 if device.type=="mps" else dtype
        ).to(device)
        self.model.eval()
        self.target_ids = [4, 7]  # upper-clothes, dress
        self.id2label   = self.model.config.id2label

    def segment_clothing(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        h0, w0 = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        scale = min(config.MAX_RESOLUTION / max(w0, h0), 1.0)
        tw, th = int(w0*scale), int(h0*scale)
        pil_res = pil.resize((tw, th), Image.Resampling.LANCZOS)

        inputs = self.processor(pil_res, return_tensors="pt").to(device)
        if device.type == "mps": inputs = {k: v.float() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(th, tw), mode="bilinear", align_corners=False
        )
        seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        seg_full = cv2.resize(seg, (w0, h0), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros_like(seg_full, dtype=np.uint8)
        detected = []
        for cid in self.target_ids:
            if np.any(seg_full == cid):
                mask[seg_full == cid] = 255
                detected.append(self.id2label[cid])
        if mask.sum() > 0:
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.GaussianBlur(mask, (7,7), 0)

        info = {'detected_classes': detected, 'total_pixels': int(mask.sum()//255), 'confidence': 'high' if detected else 'low'}
        return mask, info

#--------------------------------------------------------------------------------------------
# VIRTUAL TRY-ON MODULE
#--------------------------------------------------------------------------------------------
class VirtualTryOnModule:
    def __init__(self):
        print(f"\n🔧 Caricamento Inpainting: {config.INPAINT_MODEL}")
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config.INPAINT_MODEL,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        self.pipeline.enable_attention_slicing()

        print(f"🔧 Caricamento IP-Adapter: {config.IP_ADAPTER}")
        adapter = T2IAdapter.from_pretrained(
            config.IP_ADAPTER,
            torch_dtype=dtype
        )
        self.pipeline.register_modules(ip_adapters=adapter)
        self.pipeline.set_ip_adapter_scale(1.0)
        print("✅ Inpainting + IP-Adapter pronti")

    def apply_garment(self, person: np.ndarray, garment_path: str, mask: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert("L")
        garment = Image.open(garment_path).convert("RGBA")

        w,h = pil.size
        tw,th = (w//8*8, h//8*8)
        pil     = pil.resize((tw,th),  Image.Resampling.LANCZOS)
        mask_pil= mask_pil.resize((tw,th), Image.Resampling.LANCZOS)
        garment = garment.resize((tw,th), Image.Resampling.LANCZOS)

        result = self.pipeline(
            image=pil,
            mask_image=mask_pil,
            ip_adapter_images=[garment],
            prompt="photorealistic, high quality clothing, natural lighting, well-fitted garment, detailed fabric texture",
            negative_prompt="blurry, lowres, deformed, text, watermark, artifacts",
            guidance_scale=config.GUIDANCE_SCALE,
            strength=config.STRENGTH,
            num_inference_steps=config.INPAINT_STEPS
        ).images[0]

        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

#--------------------------------------------------------------------------------------------
# MAIN PIPELINE
#--------------------------------------------------------------------------------------------
class ModularVirtualTryOnPipeline:
    def __init__(self):
        print("\n🚀 Inizializzo Pipeline...")
        self.seg  = ClothingSegmentationModule()
        self.vto  = VirtualTryOnModule()

    def process(self, person_path: str, garment_path: str, output_path: str=None) -> str:
        print(f"\n⏳ Elaboro: {person_path} + {garment_path}")
        person = cv2.imread(person_path)
        mask, info = self.seg.segment_clothing(person)
        dbg = os.path.join(config.OUTPUT_DIR, "debug_mask.png")
        cv2.imwrite(dbg, mask)
        print(f"💾 Mask debug: {dbg}")

        result = self.vto.apply_garment(person, garment_path, mask)
        # Post-process
        result = cv2.resize(result, (person.shape[1], person.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharp = cv2.filter2D(result, -1, kernel)
        final = cv2.addWeighted(result,0.8,sharp,0.2,0)

        out = output_path or os.path.join(config.OUTPUT_DIR, "virtual_tryon_result.png")
        cv2.imwrite(out, final)
        print(f"💾 Risultato: {out}")
        print(f"🎯 Rilevato: {', '.join(info['detected_classes'])} | Pixel: {info['total_pixels']} | Conf: {info['confidence']}")
        return out

if __name__ == "__main__":
    print(f"✅ PyTorch {torch.__version__} | Device: {device}")
    pipeline = ModularVirtualTryOnPipeline()
    pipeline.process(config.PERSON_IMAGE, config.GARMENT_IMAGE)
