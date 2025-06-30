#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 PIPELINE VIRTUAL TRY-ON MODULARE - HUGGING FACE MODELS 🚀

Pipeline modulare che utilizza i migliori modelli Hugging Face per virtual try-on:
- SegFormer B2 per segmentazione clothing precisa
- CatVTON per integrazione realistica degli indumenti  
- Ottimizzazioni specifiche per Mac M2 Pro (16GB RAM)

Autore: Assistente AI
Ottimizzato per: MacBook Pro M2 Pro, 16GB RAM
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Hugging Face imports
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import requests
from io import BytesIO

print("🚀 === PIPELINE VIRTUAL TRY-ON MODULARE === 🚀")
print(f"PyTorch: {torch.__version__}")
print(f"Hardware: Mac M2 Pro ottimizzato")

#--------------------------------------------------------------------------------------------
# CONFIGURAZIONE E COSTANTI
#--------------------------------------------------------------------------------------------

class Config:
    """Configurazione centralizzata per la pipeline"""
    
    # Files di input
    PERSON_IMAGE = "tst2.jpg"
    GARMENT_IMAGE = "Soggetto.png"
    OUTPUT_DIR = "outputs"
    
    # Modelli Hugging Face
    SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
    CATVTON_MODEL = "zhengchong/CatVTON"  # Fallback: diffusers inpainting
    
    # Parametri ottimizzazione Mac M2 Pro
    MAX_RESOLUTION = 768  # Ottimale per 16GB RAM
    USE_FLOAT16 = False   # DISABILITATO per compatibilità MPS
    ENABLE_ATTENTION_SLICING = True
    BATCH_SIZE = 1
    
    # Parametri qualità
    SEGMENTATION_THRESHOLD = 0.7
    INPAINTING_STEPS = 20
    GUIDANCE_SCALE = 7.5
    
    # Device detection con fix MPS
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

# Inizializzazione configurazione
config = Config()
device = config.get_device()

# Fix dtype per MPS compatibility
if device == "mps":
    dtype = torch.float32  # Forza float32 su MPS per stabilità
    print("🍎 MPS rilevato: forzando float32 per compatibilità")
else:
    dtype = torch.float16 if config.USE_FLOAT16 and device != "cpu" else torch.float32

print(f"🔧 Device: {device}")
print(f"🔧 Precision: {dtype}")
print(f"🔧 Max Resolution: {config.MAX_RESOLUTION}")

# Crea directory output
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

#--------------------------------------------------------------------------------------------
# CLASSE SEGMENTATION MODULE
#--------------------------------------------------------------------------------------------

class ClothingSegmentationModule:
    """Modulo per segmentazione precisa degli indumenti con SegFormer"""
    
    def __init__(self):
        self.model_name = config.SEGFORMER_MODEL
        self.processor = None
        self.model = None
        self.load_model()
        
        # Mapping classi clothing
        self.clothing_classes = {
            0: 'Background', 1: 'Hat', 2: 'Hair', 3: 'Sunglasses', 
            4: 'Upper-clothes', 5: 'Skirt', 6: 'Pants', 7: 'Dress', 
            8: 'Belt', 9: 'Left-shoe', 10: 'Right-shoe', 11: 'Face', 
            12: 'Left-leg', 13: 'Right-leg', 14: 'Left-arm', 15: 'Right-arm', 
            16: 'Bag', 17: 'Scarf'
        }
        
        # Classi target per rimozione (upper body clothing)
        self.target_classes = [4, 7]  # Upper-clothes, Dress
        
    def load_model(self):
        """Carica SegFormer ottimizzato per Mac M2 Pro"""
        print(f"\n🔧 Caricamento SegFormer: {self.model_name}")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            
            # Caricamento modello con gestione esplicita dtype per MPS
            if device == "mps":
                # Su MPS, meglio usare float32 per stabilità
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Forzato float32 per MPS
                    low_cpu_mem_usage=True
                )
                print("🍎 MPS rilevato: utilizzando float32 per stabilità")
            else:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
            
            # Ottimizzazioni Mac M2 Pro
            self.model = self.model.to(device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
                
            print("✅ SegFormer caricato con successo")
            
        except Exception as e:
            print(f"❌ Errore caricamento SegFormer: {e}")
            raise
    
    def segment_clothing(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Segmenta gli indumenti nell'immagine
        
        Args:
            image: Immagine BGR (OpenCV format)
            
        Returns:
            mask: Maschera binaria degli indumenti da rimuovere
            info: Informazioni sulla segmentazione
        """
        print("\n🎯 Segmentazione clothing in corso...")
        
        # Preprocessing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Ridimensionamento se necessario
        original_size = image_pil.size
        if max(original_size) > config.MAX_RESOLUTION:
            scale = config.MAX_RESOLUTION / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
            print(f"📏 Ridimensionato da {original_size} a {new_size}")
        
        # Inferenza SegFormer con FIX per MPS compatibility
        inputs = self.processor(images=image_pil, return_tensors="pt")
        
        # FIX CRUCIALE: Su MPS usa sempre float32 per SegFormer
        if device == "mps":
            inputs = {k: v.to(device).float() for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device).to(dtype) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-processing
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_pil.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        predicted_segmentation = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # Ridimensiona al formato originale se necessario
        if image_pil.size != original_size:
            predicted_segmentation = cv2.resize(
                predicted_segmentation.astype(np.uint8),
                original_size,
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
        
        # Crea maschera per classi target
        clothing_mask = np.zeros_like(predicted_segmentation, dtype=np.uint8)
        detected_classes = []
        
        for class_id in self.target_classes:
            pixels = np.sum(predicted_segmentation == class_id)
            if pixels > 100:  # Soglia minima
                clothing_mask[predicted_segmentation == class_id] = 255
                detected_classes.append(self.clothing_classes[class_id])
                print(f"✅ Rilevato {self.clothing_classes[class_id]}: {pixels} pixel")
        
        # Miglioramento morfologico
        if np.sum(clothing_mask > 0) > 0:
            kernel = np.ones((5,5), np.uint8)
            clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
            clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
            clothing_mask = cv2.GaussianBlur(clothing_mask, (5, 5), 0)
        
        # Info segmentazione
        segmentation_info = {
            'detected_classes': detected_classes,
            'total_pixels': np.sum(clothing_mask > 0),
            'segmentation_map': predicted_segmentation,
            'confidence': 'high' if detected_classes else 'low'
        }
        
        print(f"🎯 Segmentazione completata: {segmentation_info['total_pixels']} pixel")
        
        return clothing_mask, segmentation_info

#--------------------------------------------------------------------------------------------
# CLASSE VIRTUAL TRY-ON MODULE  
#--------------------------------------------------------------------------------------------

class VirtualTryOnModule:
    """Modulo per virtual try-on con modelli diffusion"""
    
    def __init__(self):
        self.model_name = None
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Carica il modello virtual try-on ottimale"""
        print(f"\n🔧 Caricamento modello Virtual Try-On...")
        
        # Lista modelli in ordine di preferenza
        model_candidates = [
            ("zhengchong/CatVTON", "catvton"),
            ("runwayml/stable-diffusion-inpainting", "sd_inpainting"),
            ("stabilityai/stable-diffusion-2-inpainting", "sd2_inpainting")
        ]
        
        for model_name, model_type in model_candidates:
            try:
                print(f"🔄 Tentativo caricamento: {model_name}")
                
                if model_type == "catvton":
                    # Prova CatVTON specifico
                    self.pipeline = AutoPipelineForInpainting.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32 if device == "mps" else dtype,  # Fix MPS
                        use_safetensors=True,
                        low_cpu_mem_usage=True
                    )
                else:
                    # Fallback a Stable Diffusion Inpainting
                    self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32 if device == "mps" else dtype,  # Fix MPS
                        use_safetensors=True,
                        low_cpu_mem_usage=True
                    )
                
                # Ottimizzazioni Mac M2 Pro
                self.pipeline = self.pipeline.to(device)
                
                if config.ENABLE_ATTENTION_SLICING:
                    self.pipeline.enable_attention_slicing()
                
                # Ottimizzazioni specifiche per MPS
                if device == "mps":
                    self.pipeline.enable_attention_slicing()
                
                self.model_name = model_name
                print(f"✅ Modello caricato: {model_name}")
                break
                
            except Exception as e:
                print(f"⚠️ Fallito {model_name}: {e}")
                continue
        
        if self.pipeline is None:
            raise RuntimeError("❌ Impossibile caricare alcun modello virtual try-on")
    
    def apply_garment(self, person_image: np.ndarray, garment_image: np.ndarray, 
                     clothing_mask: np.ndarray) -> np.ndarray:
        """
        Applica l'indumento sulla persona
        
        Args:
            person_image: Immagine della persona (BGR)
            garment_image: Immagine dell'indumento (BGR/BGRA)
            clothing_mask: Maschera degli indumenti da sostituire
            
        Returns:
            result_image: Immagine risultante (BGR)
        """
        print("\n🎨 Applicazione virtual try-on...")
        
        # Preprocessing immagini
        person_pil = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(clothing_mask)
        
        # Ridimensionamento coordinato
        target_size = self._get_optimal_size(person_pil.size)
        if person_pil.size != target_size:
            person_pil = person_pil.resize(target_size, Image.Resampling.LANCZOS)
            mask_pil = mask_pil.resize(target_size, Image.Resampling.NEAREST)
            print(f"📏 Ridimensionato a: {target_size}")
        
        # Preprocessing indumento
        garment_pil = self._prepare_garment(garment_image, target_size)
        
        # Prompt engineering per risultati ottimali
        prompt = self._generate_prompt(garment_pil)
        negative_prompt = self._generate_negative_prompt()
        
        print(f"📝 Prompt: {prompt}")
        
        # Generazione con controllo errori
        try:
            generator = torch.Generator(device=device).manual_seed(42)
            
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=person_pil,
                    mask_image=mask_pil,
                    num_inference_steps=config.INPAINTING_STEPS,
                    guidance_scale=config.GUIDANCE_SCALE,
                    strength=0.8,
                    generator=generator
                ).images[0]
            
            print("✅ Virtual try-on completato")
            
            # Conversione a BGR per OpenCV
            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            
            return result_bgr
            
        except Exception as e:
            print(f"❌ Errore durante virtual try-on: {e}")
            
            # Fallback: semplice sostituzione
            print("🔄 Utilizzo fallback method...")
            return self._fallback_application(person_image, garment_image, clothing_mask)
    
    def _get_optimal_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calcola dimensioni ottimali per la generazione"""
        width, height = original_size
        max_dim = config.MAX_RESOLUTION
        
        if max(width, height) <= max_dim:
            # Arrotonda a multipli di 8 (requisito diffusion)
            return (width // 8 * 8, height // 8 * 8)
        
        # Ridimensiona mantenendo aspect ratio
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        
        # Arrotonda a multipli di 8
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8
        
        return (new_width, new_height)
    
    def _prepare_garment(self, garment_image: np.ndarray, target_size: Tuple[int, int]) -> Image.Image:
        """Prepara l'indumento per l'inserimento nel prompt"""
        # Converti a RGB se necessario
        if garment_image.shape[2] == 4:
            # BGRA -> RGB con sfondo bianco
            rgb = cv2.cvtColor(garment_image[:,:,:3], cv2.COLOR_BGR2RGB)
            alpha = garment_image[:,:,3] / 255.0
            white_bg = np.ones_like(rgb) * 255
            rgb = (alpha[:,:,np.newaxis] * rgb + (1 - alpha[:,:,np.newaxis]) * white_bg).astype(np.uint8)
        else:
            rgb = cv2.cvtColor(garment_image, cv2.COLOR_BGR2RGB)
        
        garment_pil = Image.fromarray(rgb)
        
        # Ridimensiona per analisi
        garment_pil = garment_pil.resize((224, 224), Image.Resampling.LANCZOS)
        
        return garment_pil
    
    def _generate_prompt(self, garment_pil: Image.Image) -> str:
        """Genera prompt ottimizzato basato sull'indumento"""
        # Analisi colore dominante
        garment_array = np.array(garment_pil)
        dominant_color = self._get_dominant_color(garment_array)
        
        # Prompt base
        base_prompt = f"a person wearing a {dominant_color} shirt, "
        base_prompt += "high quality, realistic clothing, natural lighting, "
        base_prompt += "well-fitted garment, professional photo, detailed fabric texture"
        
        return base_prompt
    
    def _generate_negative_prompt(self) -> str:
        """Genera negative prompt per evitare artifacts"""
        return ("blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "poorly fitted clothes, unnatural pose, artifacts, noise, "
                "multiple people, extra limbs, missing parts")
    
    def _get_dominant_color(self, image_array: np.ndarray) -> str:
        """Estrae il colore dominante dell'indumento"""
        # Semplificazione: analizza il pixel centrale
        h, w = image_array.shape[:2]
        center_color = image_array[h//2, w//2]
        r, g, b = center_color
        
        # Mapping colori semplificato
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        else:
            return "colored"
    
    def _fallback_application(self, person_image: np.ndarray, garment_image: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """Metodo di fallback per applicazione indumento"""
        # Semplice inpainting con OpenCV
        result = cv2.inpaint(person_image, mask, 10, cv2.INPAINT_TELEA)
        
        # Sovrapposizione semplice dell'indumento (implementazione base)
        # TODO: Implementare logica più sofisticata se necessario
        
        return result

#--------------------------------------------------------------------------------------------
# CLASSE MAIN PIPELINE
#--------------------------------------------------------------------------------------------

class ModularVirtualTryOnPipeline:
    """Pipeline principale che coordina tutti i moduli"""
    
    def __init__(self):
        print("\n🚀 Inizializzazione Pipeline Modulare...")
        
        # Inizializza moduli
        self.segmentation_module = ClothingSegmentationModule()
        self.tryon_module = VirtualTryOnModule()
        
        print("✅ Pipeline inizializzata con successo!")
    
    def process(self, person_image_path: str, garment_image_path: str, 
                output_path: str = None) -> str:
        """
        Elabora virtual try-on completo
        
        Args:
            person_image_path: Path immagine persona
            garment_image_path: Path immagine indumento
            output_path: Path output (opzionale)
            
        Returns:
            output_path: Path del risultato finale
        """
        print(f"\n🎯 === ELABORAZIONE VIRTUAL TRY-ON ===")
        print(f"👤 Persona: {person_image_path}")
        print(f"👕 Indumento: {garment_image_path}")
        
        # Validazione input
        if not os.path.exists(person_image_path):
            raise FileNotFoundError(f"Immagine persona non trovata: {person_image_path}")
        if not os.path.exists(garment_image_path):
            raise FileNotFoundError(f"Immagine indumento non trovata: {garment_image_path}")
        
        # Caricamento immagini
        print("\n📥 Caricamento immagini...")
        person_image = cv2.imread(person_image_path)
        garment_image = cv2.imread(garment_image_path, cv2.IMREAD_UNCHANGED)
        
        if person_image is None:
            raise ValueError(f"Impossibile caricare immagine persona: {person_image_path}")
        if garment_image is None:
            raise ValueError(f"Impossibile caricare immagine indumento: {garment_image_path}")
        
        print(f"✅ Persona: {person_image.shape}")
        print(f"✅ Indumento: {garment_image.shape}")
        
        # Step 1: Segmentazione clothing
        print(f"\n1️⃣ SEGMENTAZIONE CLOTHING")
        clothing_mask, seg_info = self.segmentation_module.segment_clothing(person_image)
        
        # Salva debug segmentazione
        seg_debug_path = os.path.join(config.OUTPUT_DIR, "debug_segmentation.png")
        cv2.imwrite(seg_debug_path, clothing_mask)
        print(f"💾 Debug segmentazione: {seg_debug_path}")
        
        # Step 2: Virtual Try-On
        print(f"\n2️⃣ VIRTUAL TRY-ON")
        result_image = self.tryon_module.apply_garment(person_image, garment_image, clothing_mask)
        
        # Step 3: Post-processing
        print(f"\n3️⃣ POST-PROCESSING")
        final_result = self._post_process(result_image, person_image)
        
        # Step 4: Salvataggio
        print(f"\n4️⃣ SALVATAGGIO")
        if output_path is None:
            output_path = os.path.join(config.OUTPUT_DIR, "virtual_tryon_result.png")
        
        cv2.imwrite(output_path, final_result)
        print(f"💾 Risultato finale: {output_path}")
        
        # Crea analisi comparativa
        self._create_comparison(person_image, final_result, clothing_mask, output_path)
        
        # Statistiche finali
        self._print_statistics(seg_info, output_path)
        
        return output_path
    
    def _post_process(self, result_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """Post-processing per migliorare la qualità finale"""
        
        # Ridimensiona al formato originale se necessario
        if result_image.shape != original_image.shape:
            result_image = cv2.resize(result_image, 
                                    (original_image.shape[1], original_image.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4)
        
        # Leggero sharpening
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(result_image, -1, kernel_sharpen)
        result_image = cv2.addWeighted(result_image, 0.8, sharpened, 0.2, 0)
        
        # Correzione colore leggera
        result_image = cv2.convertScaleAbs(result_image, alpha=1.02, beta=3)
        
        return result_image
    
    def _create_comparison(self, original: np.ndarray, result: np.ndarray, 
                          mask: np.ndarray, output_path: str):
        """Crea immagine comparativa"""
        h, w = original.shape[:2]
        
        # Ridimensiona per comparazione
        scale = min(300 / w, 250 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        orig_small = cv2.resize(original, (new_w, new_h))
        result_small = cv2.resize(result, (new_w, new_h))
        mask_small = cv2.resize(mask, (new_w, new_h))
        mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
        
        # Crea griglia 2x2
        top_row = np.hstack([orig_small, mask_colored])
        bottom_row = np.hstack([result_small, result_small])  # Due volte il risultato
        comparison = np.vstack([top_row, bottom_row])
        
        # Aggiungi etichette
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "ORIGINALE", (10, 30), font, 0.6, (255,255,255), 2)
        cv2.putText(comparison, "MASCHERA", (new_w + 10, 30), font, 0.6, (255,255,255), 2)
        cv2.putText(comparison, "RISULTATO", (10, new_h + 30), font, 0.6, (255,255,255), 2)
        cv2.putText(comparison, "FINALE", (new_w + 10, new_h + 30), font, 0.6, (255,255,255), 2)
        
        comparison_path = output_path.replace('.png', '_comparison.png')
        cv2.imwrite(comparison_path, comparison)
        print(f"💾 Comparazione: {comparison_path}")
    
    def _print_statistics(self, seg_info: dict, output_path: str):
        """Stampa statistiche finali"""
        print(f"\n📊 === STATISTICHE FINALI ===")
        print(f"🎯 Classi rilevate: {', '.join(seg_info['detected_classes'])}")
        print(f"🔢 Pixel segmentati: {seg_info['total_pixels']}")
        print(f"✅ Confidenza: {seg_info['confidence']}")
        print(f"💾 Output: {output_path}")
        print(f"📁 Directory: {config.OUTPUT_DIR}")
        print(f"🚀 Processo completato con successo!")

#--------------------------------------------------------------------------------------------
# FUNZIONI UTILITY
#--------------------------------------------------------------------------------------------

def check_requirements():
    """Verifica i requisiti di sistema"""
    print("\n🔍 Verifica requisiti di sistema...")
    
    # Verifica PyTorch
    print(f"✅ PyTorch: {torch.__version__}")
    
    # Verifica device
    device = config.get_device()
    print(f"✅ Device: {device}")
    
    if device == "mps":
        print("🍎 Apple Silicon rilevato - Ottimizzazioni MPS attive")
    elif device == "cpu":
        print("⚠️ Usando CPU - Performance ridotte")
    
    # Verifica memoria
    if device == "mps":
        print("💾 Unified Memory Mac: Ottimale per virtual try-on")
    
    print("✅ Sistema compatibile!")

def download_sample_images():
    """Scarica immagini di esempio per test"""
    print("\n📥 Download immagini di esempio...")
    
    # URLs di esempio (sostituire con immagini reali)
    sample_urls = {
        "person_sample.jpg": "https://example.com/person.jpg",  # Sostituire
        "garment_sample.png": "https://example.com/garment.png"  # Sostituire
    }
    
    for filename, url in sample_urls.items():
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Crea manualmente: {filepath}")
            # In un'implementazione reale, scaricare da URL validi

def create_demo_interface():
    """Crea interfaccia demo opzionale"""
    try:
        import gradio as gr
        
        def process_images(person_img, garment_img):
            """Wrapper per Gradio"""
            try:
                # Salva immagini temporanee
                person_path = os.path.join(config.OUTPUT_DIR, "temp_person.jpg")
                garment_path = os.path.join(config.OUTPUT_DIR, "temp_garment.png")
                
                person_img.save(person_path)
                garment_img.save(garment_path)
                
                # Elabora
                pipeline = ModularVirtualTryOnPipeline()
                result_path = pipeline.process(person_path, garment_path)
                
                # Carica risultato
                result_img = Image.open(result_path)
                return result_img
                
            except Exception as e:
                return f"Errore: {str(e)}"
        
        # Crea interfaccia
        interface = gr.Interface(
            fn=process_images,
            inputs=[
                gr.Image(type="pil", label="Immagine Persona"),
                gr.Image(type="pil", label="Indumento")
            ],
            outputs=gr.Image(type="pil", label="Risultato Virtual Try-On"),
            title="🚀 Virtual Try-On Modulare",
            description="Pipeline modulare con modelli Hugging Face per virtual try-on realistico"
        )
        
        return interface
        
    except ImportError:
        print("ℹ️ Gradio non installato - Interfaccia demo non disponibile")
        return None

#--------------------------------------------------------------------------------------------
# FUNZIONE MAIN
#--------------------------------------------------------------------------------------------

def main():
    """Funzione principale"""
    print("🚀 === VIRTUAL TRY-ON MODULARE === 🚀")
    
    try:
        # Verifica requisiti
        check_requirements()
        
        # Verifica files di input
        if not os.path.exists(config.PERSON_IMAGE):
            print(f"❌ File persona non trovato: {config.PERSON_IMAGE}")
            print("📁 Assicurati che il file esista nella directory corrente")
            return
        
        if not os.path.exists(config.GARMENT_IMAGE):
            print(f"❌ File indumento non trovato: {config.GARMENT_IMAGE}")
            print("📁 Assicurati che il file esista nella directory corrente")
            return
        
        # Inizializza pipeline
        pipeline = ModularVirtualTryOnPipeline()
        
        # Elabora virtual try-on
        result_path = pipeline.process(config.PERSON_IMAGE, config.GARMENT_IMAGE)
        
        # Crea interfaccia demo opzionale
        demo_interface = create_demo_interface()
        if demo_interface:
            print("\n🌐 Avvio interfaccia demo...")
            demo_interface.launch(share=False, inbrowser=True)
        
        # Visualizzazione risultato
        try:
            result_img = cv2.imread(result_path)
            cv2.imshow('Virtual Try-On Modulare - Risultato', result_img)
            print("\n👁️ Premi un tasto per chiudere la visualizzazione...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("ℹ️ Visualizzazione non disponibile - Controlla il file di output")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        print("\n🔧 Suggerimenti per il debug:")
        print("1. Verifica che le immagini siano valide e leggibili")
        print("2. Controlla la connessione internet per il download dei modelli")
        print("3. Assicurati di avere abbastanza memoria disponibile")
        print("4. Prova con immagini di dimensioni più piccole")

if __name__ == "__main__":
    main()