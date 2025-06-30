#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 PIPELINE VIRTUAL TRY-ON MIGLIORATA - VERSIONE 2.0 🚀

Pipeline migliorata per virtual try-on più preciso e realistico:
- Face preservation per mantenere identità
- Segmentazione più precisa con erosione maschera
- ControlNet per maggiore controllo
- Blending avanzato per risultati naturali
- Ottimizzato per sostituire solo l'indumento target

Miglioramenti principali:
✅ Preservazione volto e identità
✅ Maschera più precisa (solo torso)
✅ Prompt engineering migliorato
✅ Post-processing avanzato
✅ Fallback con OpenCV per casi edge
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")

# Hugging Face imports
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
try:
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False

# Face detection
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    try:
        import mediapipe as mp
        MEDIAPIPE_AVAILABLE = True
    except ImportError:
        MEDIAPIPE_AVAILABLE = False
        FACE_RECOGNITION_AVAILABLE = False

print("🚀 === PIPELINE VIRTUAL TRY-ON MIGLIORATA V2.0 === 🚀")
print(f"PyTorch: {torch.__version__}")
print(f"ControlNet disponibile: {CONTROLNET_AVAILABLE}")
print(f"Face detection: {FACE_RECOGNITION_AVAILABLE or MEDIAPIPE_AVAILABLE}")

#--------------------------------------------------------------------------------------------
# CONFIGURAZIONE MIGLIORATA
#--------------------------------------------------------------------------------------------

class ImprovedConfig:
    """Configurazione ottimizzata per virtual try-on preciso"""
    
    # Files di input
    PERSON_IMAGE = "tst2.jpg"  # Aggiornato al tuo riferimento
    GARMENT_IMAGE = "Soggetto.png"
    OUTPUT_DIR = "outputs_v2"
    
    # Modelli Hugging Face (ordine di preferenza)
    PREFERRED_MODELS = [
        "stabilityai/stable-diffusion-2-inpainting",  # Più stabile
        "runwayml/stable-diffusion-inpainting",
        "zhengchong/CatVTON"
    ]
    
    SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
    
    # ControlNet per maggiore controllo (se disponibile)
    CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_inpaint" if CONTROLNET_AVAILABLE else None
    
    # Parametri ottimizzazione
    MAX_RESOLUTION = 512  # Ridotto per maggiore stabilità
    USE_FLOAT16 = False   # Disabilitato per Mac M2
    ENABLE_ATTENTION_SLICING = True
    
    # Parametri qualità migliorati
    SEGMENTATION_THRESHOLD = 0.8  # Più restrittivo
    INPAINTING_STEPS = 25  # Aumentato per qualità
    GUIDANCE_SCALE = 12.0  # Aumentato per maggiore aderenza al prompt
    STRENGTH = 0.85  # Ridotto per preservare meglio l'originale
    
    # Nuovi parametri per preservazione identità
    FACE_PRESERVATION = True
    MASK_EROSION_KERNEL_SIZE = 15  # Erosione maschera per area più conservativa
    MASK_BLUR_RADIUS = 8  # Blur della maschera per transizioni naturali
    COLOR_MATCHING_STRENGTH = 0.3  # Forza del color matching
    
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

config = ImprovedConfig()
device = config.get_device()
dtype = torch.float32 if device == "mps" else torch.float16 if config.USE_FLOAT16 else torch.float32

print(f"🔧 Device: {device}")
print(f"🔧 Precision: {dtype}")
print(f"🔧 Max Resolution: {config.MAX_RESOLUTION}")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

#--------------------------------------------------------------------------------------------
# CLASSE FACE PRESERVATION MODULE
#--------------------------------------------------------------------------------------------

class FacePreservationModule:
    """Modulo per preservare il volto durante il virtual try-on"""
    
    def __init__(self):
        self.face_detector = None
        self.setup_face_detection()
    
    def setup_face_detection(self):
        """Inizializza il rilevamento volti"""
        if FACE_RECOGNITION_AVAILABLE:
            print("✅ Utilizzo face_recognition per rilevamento volti")
            self.detection_method = "face_recognition"
        elif MEDIAPIPE_AVAILABLE:
            print("✅ Utilizzo MediaPipe per rilevamento volti")
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self.detection_method = "mediapipe"
        else:
            print("⚠️ Nessun modulo face detection disponibile - utilizzo Haar Cascades")
            # Fallback a OpenCV Haar Cascades
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.detection_method = "opencv"
            except:
                print("❌ Face detection non disponibile")
                self.detection_method = None
    
    def detect_face_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Rileva la regione del volto per preservarla
        
        Args:
            image: Immagine BGR
            
        Returns:
            face_mask: Maschera del volto da preservare (None se non rilevato)
        """
        if self.detection_method is None:
            return None
            
        h, w = image.shape[:2]
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            if self.detection_method == "face_recognition":
                # Converti BGR a RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                for (top, right, bottom, left) in face_locations:
                    # Espandi leggermente l'area del volto
                    padding = 30
                    top = max(0, top - padding)
                    left = max(0, left - padding)
                    bottom = min(h, bottom + padding)
                    right = min(w, right + padding)
                    
                    face_mask[top:bottom, left:right] = 255
                    
            elif self.detection_method == "mediapipe":
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb_image)
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Espandi area
                        padding = 40
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        width = min(w - x, width + 2 * padding)
                        height = min(h - y, height + 2 * padding)
                        
                        face_mask[y:y+height, x:x+width] = 255
                        
            elif self.detection_method == "opencv":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
                
                for (x, y, w_face, h_face) in faces:
                    # Espandi area del volto
                    padding = 50
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w_face = min(w - x, w_face + 2 * padding)
                    h_face = min(h - y, h_face + 2 * padding)
                    
                    face_mask[y:y+h_face, x:x+w_face] = 255
            
            if np.sum(face_mask) > 0:
                # Applica gaussian blur per transizioni naturali
                face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
                print(f"✅ Volto rilevato: {np.sum(face_mask > 0)} pixel")
                return face_mask
                
        except Exception as e:
            print(f"⚠️ Errore rilevamento volto: {e}")
        
        return None

#--------------------------------------------------------------------------------------------
# CLASSE IMPROVED SEGMENTATION MODULE
#--------------------------------------------------------------------------------------------

class ImprovedClothingSegmentationModule:
    """Modulo di segmentazione migliorato per virtual try-on preciso"""
    
    def __init__(self):
        self.model_name = config.SEGFORMER_MODEL
        self.processor = None
        self.model = None
        self.face_preservation = FacePreservationModule()
        self.load_model()
        
        # Mapping classi clothing (stesso del codice originale)
        self.clothing_classes = {
            0: 'Background', 1: 'Hat', 2: 'Hair', 3: 'Sunglasses', 
            4: 'Upper-clothes', 5: 'Skirt', 6: 'Pants', 7: 'Dress', 
            8: 'Belt', 9: 'Left-shoe', 10: 'Right-shoe', 11: 'Face', 
            12: 'Left-leg', 13: 'Right-leg', 14: 'Left-arm', 15: 'Right-arm', 
            16: 'Bag', 17: 'Scarf'
        }
        
        # FOCUS: Solo upper-clothes per sostituire t-shirt
        self.target_classes = [4]  # Solo Upper-clothes
    
    def load_model(self):
        """Carica SegFormer"""
        print(f"\n🔧 Caricamento SegFormer migliorato...")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if device == "mps" else dtype,
                low_cpu_mem_usage=True
            )
            
            self.model = self.model.to(device)
            self.model.eval()
            
            print("✅ SegFormer migliorato caricato")
            
        except Exception as e:
            print(f"❌ Errore caricamento SegFormer: {e}")
            raise
    
    def create_precise_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Crea maschera precisa per virtual try-on conservativo
        
        Args:
            image: Immagine BGR
            
        Returns:
            clothing_mask: Maschera dell'indumento da sostituire
            face_mask: Maschera del volto da preservare
            info: Informazioni sulla segmentazione
        """
        print("\n🎯 Segmentazione precisa in corso...")
        
        # Segmentazione clothing standard
        clothing_mask, seg_info = self.segment_clothing_base(image)
        
        # Rilevamento volto per preservazione
        face_mask = self.face_preservation.detect_face_region(image)
        if face_mask is None:
            face_mask = np.zeros_like(clothing_mask)
        
        # MIGLIORAMENTO: Erosione della maschera clothing per essere più conservativi
        if np.sum(clothing_mask) > 0:
            # Applica erosione per ridurre l'area di intervento
            erosion_kernel = np.ones((config.MASK_EROSION_KERNEL_SIZE, config.MASK_EROSION_KERNEL_SIZE), np.uint8)
            clothing_mask = cv2.erode(clothing_mask, erosion_kernel, iterations=1)
            
            # Rimuovi la parte che si sovrappone al volto
            clothing_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(face_mask))
            
            # NUOVA TECNICA: Limitazione area torso
            clothing_mask = self._limit_to_torso_area(clothing_mask, image)
            
            # Blur per transizioni naturali
            clothing_mask = cv2.GaussianBlur(clothing_mask, (config.MASK_BLUR_RADIUS, config.MASK_BLUR_RADIUS), 0)
            
            print(f"🎯 Maschera raffinata: {np.sum(clothing_mask > 0)} pixel")
        
        seg_info['mask_refinement'] = 'applied'
        seg_info['face_detected'] = np.sum(face_mask) > 0
        
        return clothing_mask, face_mask, seg_info
    
    def segment_clothing_base(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Segmentazione base (stessa logica del codice originale)"""
        # [Stessa implementazione del codice originale]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Ridimensionamento
        original_size = image_pil.size
        if max(original_size) > config.MAX_RESOLUTION:
            scale = config.MAX_RESOLUTION / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        # Inferenza
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device).float() if device == "mps" else v.to(device).to(dtype) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-processing
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
        )
        
        predicted_segmentation = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # Ridimensiona all'originale
        if image_pil.size != original_size:
            predicted_segmentation = cv2.resize(
                predicted_segmentation.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
        
        # Crea maschera per target classes
        clothing_mask = np.zeros_like(predicted_segmentation, dtype=np.uint8)
        detected_classes = []
        
        for class_id in self.target_classes:
            pixels = np.sum(predicted_segmentation == class_id)
            if pixels > 100:
                clothing_mask[predicted_segmentation == class_id] = 255
                detected_classes.append(self.clothing_classes[class_id])
                print(f"✅ Rilevato {self.clothing_classes[class_id]}: {pixels} pixel")
        
        segmentation_info = {
            'detected_classes': detected_classes,
            'total_pixels': np.sum(clothing_mask > 0),
            'segmentation_map': predicted_segmentation,
            'confidence': 'high' if detected_classes else 'low'
        }
        
        return clothing_mask, segmentation_info
    
    def _limit_to_torso_area(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Limita la maschera all'area del torso (parte superiore del corpo)
        """
        h, w = mask.shape
        
        # Definisci area approssimativa del torso (parte centrale superiore)
        # Questa è una euristica che può essere migliorata
        top_limit = int(h * 0.15)  # Evita la testa
        bottom_limit = int(h * 0.65)  # Limita alla parte superiore del corpo
        left_limit = int(w * 0.2)   # Margini laterali
        right_limit = int(w * 0.8)
        
        # Crea maschera area torso
        torso_mask = np.zeros_like(mask)
        torso_mask[top_limit:bottom_limit, left_limit:right_limit] = 255
        
        # Applica limitazione
        limited_mask = cv2.bitwise_and(mask, torso_mask)
        
        print(f"🎯 Limitazione area torso: {np.sum(mask > 0)} -> {np.sum(limited_mask > 0)} pixel")
        
        return limited_mask

#--------------------------------------------------------------------------------------------
# CLASSE IMPROVED VIRTUAL TRY-ON MODULE
#--------------------------------------------------------------------------------------------

class ImprovedVirtualTryOnModule:
    """Modulo virtual try-on migliorato per risultati più realistici"""
    
    def __init__(self):
        self.pipeline = None
        self.controlnet_pipeline = None
        self.load_models()
    
    def load_models(self):
        """Carica i modelli ottimali per virtual try-on"""
        print(f"\n🔧 Caricamento modelli Virtual Try-On migliorati...")
        
        # Prova modelli in ordine di preferenza
        for model_name in config.PREFERRED_MODELS:
            try:
                print(f"🔄 Tentativo: {model_name}")
                
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32 if device == "mps" else dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
                
                self.pipeline = self.pipeline.to(device)
                
                # Ottimizzazioni
                if config.ENABLE_ATTENTION_SLICING:
                    self.pipeline.enable_attention_slicing()
                
                print(f"✅ Modello principale caricato: {model_name}")
                break
                
            except Exception as e:
                print(f"⚠️ Fallito {model_name}: {e}")
                continue
        
        # Prova ControlNet se disponibile
        if CONTROLNET_AVAILABLE and config.CONTROLNET_MODEL:
            try:
                print("🔄 Caricamento ControlNet...")
                controlnet = ControlNetModel.from_pretrained(
                    config.CONTROLNET_MODEL, torch_dtype=dtype, use_safetensors=True
                )
                self.controlnet_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
                self.controlnet_pipeline = self.controlnet_pipeline.to(device)
                print("✅ ControlNet caricato per maggiore controllo")
            except Exception as e:
                print(f"⚠️ ControlNet non disponibile: {e}")
        
        if self.pipeline is None:
            raise RuntimeError("❌ Impossibile caricare alcun modello")
    
    def apply_garment_improved(self, person_image: np.ndarray, garment_image: np.ndarray, 
                              clothing_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """
        Applica l'indumento con tecniche migrate per preservare identità
        
        Args:
            person_image: Immagine persona (BGR)
            garment_image: Immagine indumento (BGR/BGRA)
            clothing_mask: Maschera indumento da sostituire
            face_mask: Maschera volto da preservare
            
        Returns:
            result_image: Risultato migliorato (BGR)
        """
        print("\n🎨 Applicazione Virtual Try-On migliorata...")
        
        # Preprocessing coordinato
        person_pil, mask_pil, target_size = self._prepare_images(person_image, clothing_mask)
        
        # Analisi indumento per prompt intelligente
        garment_analysis = self._analyze_garment(garment_image)
        
        # Prompt engineering migliorato
        prompt = self._create_enhanced_prompt(garment_analysis)
        negative_prompt = self._create_enhanced_negative_prompt()
        
        print(f"📝 Prompt migliorato: {prompt}")
        
        try:
            # Generazione con parametri ottimizzati
            generator = torch.Generator(device=device).manual_seed(42)
            
            # Usa ControlNet se disponibile per maggiore controllo
            if self.controlnet_pipeline is not None:
                result = self._generate_with_controlnet(
                    person_pil, mask_pil, prompt, negative_prompt, generator
                )
            else:
                result = self._generate_standard(
                    person_pil, mask_pil, prompt, negative_prompt, generator
                )
            
            print("✅ Generazione completata")
            
            # Post-processing avanzato
            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            
            # Ridimensiona all'originale
            if result_bgr.shape[:2] != person_image.shape[:2]:
                result_bgr = cv2.resize(result_bgr, 
                                      (person_image.shape[1], person_image.shape[0]),
                                      interpolation=cv2.INTER_LANCZOS4)
            
            # MIGLIORAMENTO: Blending avanzato per preservare dettagli
            final_result = self._advanced_blending(person_image, result_bgr, clothing_mask, face_mask)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Errore generazione: {e}")
            return self._fallback_method(person_image, garment_image, clothing_mask)
    
    def _prepare_images(self, person_image: np.ndarray, mask: np.ndarray) -> Tuple:
        """Prepara le immagini per il processing"""
        person_pil = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Dimensioni ottimali
        target_size = self._get_optimal_size(person_pil.size)
        
        if person_pil.size != target_size:
            person_pil = person_pil.resize(target_size, Image.Resampling.LANCZOS)
            mask_pil = mask_pil.resize(target_size, Image.Resampling.NEAREST)
            print(f"📏 Ridimensionato a: {target_size}")
        
        return person_pil, mask_pil, target_size
    
    def _analyze_garment(self, garment_image: np.ndarray) -> Dict:
        """Analizza l'indumento per generare prompt accurato"""
        # Preprocessing indumento
        if garment_image.shape[2] == 4:
            # BGRA -> RGB con sfondo bianco
            rgb = cv2.cvtColor(garment_image[:,:,:3], cv2.COLOR_BGR2RGB)
            alpha = garment_image[:,:,3] / 255.0
            white_bg = np.ones_like(rgb) * 255
            rgb = (alpha[:,:,np.newaxis] * rgb + (1 - alpha[:,:,np.newaxis]) * white_bg).astype(np.uint8)
        else:
            rgb = cv2.cvtColor(garment_image, cv2.COLOR_BGR2RGB)
        
        # Analisi colore dominante più sofisticata
        colors = rgb.reshape(-1, 3)
        unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
        dominant_color_idx = np.argmax(counts)
        dominant_color = unique_colors[dominant_color_idx]
        
        # Mappatura colori migliorata
        color_name = self._map_color_to_name(dominant_color)
        
        # Analisi texture/pattern (semplificata)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        texture_variance = np.var(gray)
        is_textured = texture_variance > 500  # Soglia empirica
        
        return {
            'dominant_color': color_name,
            'rgb_values': dominant_color,
            'is_textured': is_textured,
            'brightness': np.mean(rgb)
        }
    
    def _map_color_to_name(self, rgb_color: np.ndarray) -> str:
        """Mappa colore RGB a nome descrittivo"""
        r, g, b = rgb_color
        
        # Mapping colori migliorato
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 150 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 150 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 150:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r < 100 and g > 150 and b > 150:
            return "cyan"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        elif 80 < r < 140 and 80 < g < 140 and 80 < b < 140:
            return "gray"
        else:
            return "colored"
    
    def _create_enhanced_prompt(self, garment_analysis: Dict) -> str:
        """Crea prompt migliorato per virtual try-on preciso"""
        color = garment_analysis['dominant_color']
        
        # Prompt base specifico per t-shirt
        if garment_analysis['is_textured']:
            garment_desc = f"textured {color} t-shirt"
        else:
            garment_desc = f"plain {color} t-shirt"
        
        # Prompt ottimizzato per preservare identità
        prompt = f"a person wearing a {garment_desc}, "
        prompt += "same face, same hair, same body proportions, "
        prompt += "realistic clothing fit, natural lighting, "
        prompt += "high quality, detailed fabric texture, "
        prompt += "keep original person identity, photorealistic"
        
        return prompt
    
    def _create_enhanced_negative_prompt(self) -> str:
        """Negative prompt migliorato per evitare alterazioni indesiderate"""
        return ("blurry, low quality, distorted face, deformed body, different person, "
                "changed face, altered identity, bad anatomy, poorly fitted clothes, "
                "unnatural pose, artifacts, noise, multiple people, extra limbs, "
                "missing parts, wrong proportions, inconsistent lighting, "
                "face modification, identity change")
    
    def _generate_standard(self, person_pil, mask_pil, prompt, negative_prompt, generator):
        """Generazione standard con Stable Diffusion"""
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person_pil,
                mask_image=mask_pil,
                num_inference_steps=config.INPAINTING_STEPS,
                guidance_scale=config.GUIDANCE_SCALE,
                strength=config.STRENGTH,
                generator=generator
            ).images[0]
        return result
    
    def _generate_with_controlnet(self, person_pil, mask_pil, prompt, negative_prompt, generator):
        """Generazione con ControlNet per maggiore controllo"""
        # Crea control image (edges o pose)
        control_image = person_pil  # Semplificato, in produzione si userebbe edge/pose detection
        
        with torch.no_grad():
            result = self.controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person_pil,
                mask_image=mask_pil,
                control_image=control_image,
                num_inference_steps=config.INPAINTING_STEPS,
                guidance_scale=config.GUIDANCE_SCALE,
                controlnet_conditioning_scale=0.8,
                generator=generator
            ).images[0]
        return result
    
    def _advanced_blending(self, original: np.ndarray, generated: np.ndarray, 
                          clothing_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """
        Blending avanzato per preservare dettagli originali
        """
        print("🎨 Applicazione blending avanzato...")
        
        # Normalizza maschere
        clothing_mask_norm = clothing_mask.astype(np.float32) / 255.0
        face_mask_norm = face_mask.astype(np.float32) / 255.0
        
        # Combina maschere per area di preservazione totale
        preserve_mask = np.maximum(face_mask_norm, 
                                 cv2.dilate(face_mask_norm, np.ones((20,20), np.uint8)))
        
        # Crea maschera blend area (dove sostituire)
        blend_mask = clothing_mask_norm * (1.0 - preserve_mask)
        
        # Espandi a 3 canali
        if len(blend_mask.shape) == 2:
            blend_mask = np.stack([blend_mask] * 3, axis=-1)
        if len(preserve_mask.shape) == 2:
            preserve_mask = np.stack([preserve_mask] * 3, axis=-1)
        
        # Blending intelligente
        result = original.astype(np.float32)
        
        # Applica generated solo nell'area clothing (non volto)
        result = result * (1.0 - blend_mask) + generated.astype(np.float32) * blend_mask
        
        # Assicura preservazione totale del volto
        result = result * (1.0 - preserve_mask) + original.astype(np.float32) * preserve_mask
        
        # Color matching leggero per naturalezza
        if config.COLOR_MATCHING_STRENGTH > 0:
            result = self._apply_color_matching(result, original, blend_mask)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_color_matching(self, result: np.ndarray, original: np.ndarray, 
                            blend_mask: np.ndarray) -> np.ndarray:
        """Applica color matching per risultati più naturali"""
        # Estrai colori medi delle aree non modificate
        preserve_area = blend_mask < 0.1
        if np.any(preserve_area):
            original_mean = np.mean(original[preserve_area], axis=0)
            result_mean = np.mean(result[preserve_area], axis=0)
            
            # Applica correzione colore leggera
            color_correction = original_mean - result_mean
            result += color_correction * config.COLOR_MATCHING_STRENGTH * blend_mask
        
        return result
    
    def _get_optimal_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calcola dimensioni ottimali (stessa logica originale)"""
        width, height = original_size
        max_dim = config.MAX_RESOLUTION
        
        if max(width, height) <= max_dim:
            return (width // 8 * 8, height // 8 * 8)
        
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        
        return (new_width // 8 * 8, new_height // 8 * 8)
    
    def _fallback_method(self, person_image: np.ndarray, garment_image: np.ndarray, 
                        clothing_mask: np.ndarray) -> np.ndarray:
        """Metodo fallback migliorato"""
        print("🔄 Utilizzo metodo fallback migliorato...")
        
        # Inpainting OpenCV
        inpainted = cv2.inpaint(person_image, clothing_mask, 10, cv2.INPAINT_TELEA)
        
        # Tentativo di sovrapposizione intelligente dell'indumento
        # (implementazione semplificata)
        
        return inpainted

#--------------------------------------------------------------------------------------------
# CLASSE MAIN PIPELINE MIGLIORATA
#--------------------------------------------------------------------------------------------

class ImprovedVirtualTryOnPipeline:
    """Pipeline principale migliorata per virtual try-on preciso"""
    
    def __init__(self):
        print("\n🚀 Inizializzazione Pipeline Virtual Try-On Migliorata...")
        
        self.segmentation_module = ImprovedClothingSegmentationModule()
        self.tryon_module = ImprovedVirtualTryOnModule()
        
        print("✅ Pipeline migliorata inizializzata!")
    
    def process_improved(self, person_image_path: str, garment_image_path: str, 
                        output_path: str = None) -> str:
        """
        Processo virtual try-on migliorato con preservazione identità
        """
        print(f"\n🎯 === VIRTUAL TRY-ON MIGLIORATO ===")
        print(f"👤 Persona: {person_image_path}")
        print(f"👕 Indumento: {garment_image_path}")
        
        # Validazione
        if not os.path.exists(person_image_path):
            raise FileNotFoundError(f"Immagine non trovata: {person_image_path}")
        if not os.path.exists(garment_image_path):
            raise FileNotFoundError(f"Indumento non trovato: {garment_image_path}")
        
        # Caricamento
        person_image = cv2.imread(person_image_path)
        garment_image = cv2.imread(garment_image_path, cv2.IMREAD_UNCHANGED)
        
        if person_image is None or garment_image is None:
            raise ValueError("Impossibile caricare le immagini")
        
        print(f"✅ Immagini caricate: Persona {person_image.shape}, Indumento {garment_image.shape}")
        
        # Step 1: Segmentazione migliorata
        print(f"\n1️⃣ SEGMENTAZIONE MIGLIORATA")
        clothing_mask, face_mask, seg_info = self.segmentation_module.create_precise_mask(person_image)
        
        # Debug masks
        debug_clothing_path = os.path.join(config.OUTPUT_DIR, "debug_clothing_mask.png")
        debug_face_path = os.path.join(config.OUTPUT_DIR, "debug_face_mask.png")
        cv2.imwrite(debug_clothing_path, clothing_mask)
        cv2.imwrite(debug_face_path, face_mask)
        print(f"💾 Debug maschere: {debug_clothing_path}, {debug_face_path}")
        
        # Step 2: Virtual Try-On migliorato
        print(f"\n2️⃣ VIRTUAL TRY-ON MIGLIORATO")
        result_image = self.tryon_module.apply_garment_improved(
            person_image, garment_image, clothing_mask, face_mask
        )
        
        # Step 3: Post-processing finale
        print(f"\n3️⃣ POST-PROCESSING FINALE")
        final_result = self._final_post_processing(result_image, person_image, face_mask)
        
        # Step 4: Salvataggio e analisi
        print(f"\n4️⃣ SALVATAGGIO E ANALISI")
        if output_path is None:
            output_path = os.path.join(config.OUTPUT_DIR, "improved_virtual_tryon_result.png")
        
        cv2.imwrite(output_path, final_result)
        print(f"💾 Risultato finale: {output_path}")
        
        # Analisi comparativa migliorata
        self._create_detailed_comparison(person_image, final_result, clothing_mask, face_mask, output_path)
        
        # Statistiche
        self._print_improved_statistics(seg_info, output_path)
        
        return output_path
    
    def _final_post_processing(self, result: np.ndarray, original: np.ndarray, 
                              face_mask: np.ndarray) -> np.ndarray:
        """Post-processing finale per rifinitura"""
        
        # Assicura preservazione volto al 100%
        if np.sum(face_mask) > 0:
            face_mask_norm = face_mask.astype(np.float32) / 255.0
            if len(face_mask_norm.shape) == 2:
                face_mask_norm = np.stack([face_mask_norm] * 3, axis=-1)
            
            result = result.astype(np.float32) * (1.0 - face_mask_norm) + \
                    original.astype(np.float32) * face_mask_norm
            result = result.astype(np.uint8)
        
        # Sharpening leggero
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 8
        sharpened = cv2.filter2D(result, -1, kernel_sharpen)
        result = cv2.addWeighted(result, 0.85, sharpened, 0.15, 0)
        
        # Correzione gamma leggera
        result = cv2.convertScaleAbs(result, alpha=1.01, beta=2)
        
        return result
    
    def _create_detailed_comparison(self, original: np.ndarray, result: np.ndarray,
                                   clothing_mask: np.ndarray, face_mask: np.ndarray, 
                                   output_path: str):
        """Crea comparazione dettagliata"""
        h, w = original.shape[:2]
        scale = min(250 / w, 200 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Ridimensiona tutto
        orig_small = cv2.resize(original, (new_w, new_h))
        result_small = cv2.resize(result, (new_w, new_h))
        clothing_small = cv2.resize(clothing_mask, (new_w, new_h))
        face_small = cv2.resize(face_mask, (new_w, new_h))
        
        # Colora maschere
        clothing_colored = cv2.applyColorMap(clothing_small, cv2.COLORMAP_HOT)
        face_colored = cv2.applyColorMap(face_small, cv2.COLORMAP_COOL)
        
        # Crea griglia 2x3
        top_row = np.hstack([orig_small, clothing_colored, face_colored])
        bottom_row = np.hstack([result_small, result_small, result_small])
        comparison = np.vstack([top_row, bottom_row])
        
        # Etichette
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "ORIGINALE", (5, 25), font, 0.5, (255,255,255), 1)
        cv2.putText(comparison, "CLOTHING MASK", (new_w + 5, 25), font, 0.5, (255,255,255), 1)
        cv2.putText(comparison, "FACE MASK", (2*new_w + 5, 25), font, 0.5, (255,255,255), 1)
        cv2.putText(comparison, "RISULTATO FINALE", (5, new_h + 25), font, 0.5, (255,255,255), 1)
        
        comparison_path = output_path.replace('.png', '_detailed_comparison.png')
        cv2.imwrite(comparison_path, comparison)
        print(f"💾 Comparazione dettagliata: {comparison_path}")
    
    def _print_improved_statistics(self, seg_info: Dict, output_path: str):
        """Statistiche migliorate"""
        print(f"\n📊 === STATISTICHE VIRTUAL TRY-ON MIGLIORATO ===")
        print(f"🎯 Classi rilevate: {', '.join(seg_info.get('detected_classes', []))}")
        print(f"🔢 Pixel clothing: {seg_info.get('total_pixels', 0)}")
        print(f"👤 Volto rilevato: {'Sì' if seg_info.get('face_detected', False) else 'No'}")
        print(f"🔧 Raffinamento maschera: {seg_info.get('mask_refinement', 'none')}")
        print(f"✅ Confidenza: {seg_info.get('confidence', 'unknown')}")
        print(f"💾 Output finale: {output_path}")
        print(f"📁 Directory: {config.OUTPUT_DIR}")
        print(f"🚀 Virtual Try-On migliorato completato!")

#--------------------------------------------------------------------------------------------
# FUNZIONE MAIN MIGLIORATA
#--------------------------------------------------------------------------------------------

def main_improved():
    """Funzione principale migliorata"""
    print("🚀 === VIRTUAL TRY-ON MIGLIORATO V2.0 === 🚀")
    
    try:
        # Verifica file
        if not os.path.exists(config.PERSON_IMAGE):
            print(f"❌ File persona non trovato: {config.PERSON_IMAGE}")
            print("💡 Modifica config.PERSON_IMAGE per puntare al file corretto")
            return
        
        if not os.path.exists(config.GARMENT_IMAGE):
            print(f"❌ File indumento non trovato: {config.GARMENT_IMAGE}")
            print("💡 Modifica config.GARMENT_IMAGE per puntare al file corretto")
            return
        
        # Inizializza pipeline migliorata
        pipeline = ImprovedVirtualTryOnPipeline()
        
        # Processo migliorato
        result_path = pipeline.process_improved(config.PERSON_IMAGE, config.GARMENT_IMAGE)
        
        print(f"\n✅ === PROCESSO COMPLETATO === ✅")
        print(f"🎯 Risultato: {result_path}")
        print(f"📁 Controlla la cartella: {config.OUTPUT_DIR}")
        
        # Visualizzazione opzionale
        try:
            result_img = cv2.imread(result_path)
            cv2.imshow('Virtual Try-On Migliorato - Risultato Finale', result_img)
            print("\n👁️ Premi un tasto per chiudere...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("ℹ️ Visualizzazione non disponibile")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        print("\n🔧 Suggerimenti debug migliorati:")
        print("1. Verifica path dei file di input")
        print("2. Controlla memoria disponibile")
        print("3. Verifica connessione per download modelli")
        print("4. Prova con immagini più piccole se necessario")

if __name__ == "__main__":
    main_improved()