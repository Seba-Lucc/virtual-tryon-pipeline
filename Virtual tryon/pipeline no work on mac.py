# Installazione per Mac:
# pip install torch torchvision diffusers transformers accelerate mediapipe opencv-python pillow

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import mediapipe as mp
import os

# Verifiche di compatibilità Mac
print(f"Dispositivo: {'Apple Silicon' if torch.backends.mps.is_available() else 'Intel/AMD'}")
print(f"PyTorch version: {torch.__version__}")

class CatVTONMacPipeline:
    def __init__(self):
        # Selezione device ottimizzata per Mac
        if torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
            self.dtype = torch.float32  # MPS supporta meglio float32
            print("Usando Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
            print("Usando NVIDIA GPU")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            print("Usando CPU (più lento ma funziona)")
        
        print("Caricamento modello Stable Diffusion Inpainting...")
        
        # Caricamento modello ottimizzato per Mac
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True  # Importante per Mac
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Ottimizzazioni specifiche per Mac
        if self.device == "mps":
            # Apple Silicon ottimizzazioni
            self.pipe.enable_attention_slicing()  # Riduce uso memoria
        elif self.device == "cpu":
            # CPU ottimizzazioni
            self.pipe.enable_attention_slicing()
            
        # Non usiamo xformers su Mac (causa problemi)
        
        # MediaPipe inizializzazione
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.5,
            model_complexity=1  # Leggero per Mac
        )
        
        print("Pipeline inizializzata con successo!")

    def detect_pose_and_create_mask(self, image):
        """Rileva pose e crea maschera - ottimizzato per Mac"""
        height, width = image.shape[:2]
        
        # Ridimensiona se troppo grande (ottimizzazione Mac)
        if width > 1024 or height > 1024:
            scale = min(1024/width, 1024/height)
            new_width, new_height = int(width * scale), int(height * scale)
            image_resized = cv2.resize(image, (new_width, new_height))
            scale_factor = (width/new_width, height/new_height)
        else:
            image_resized = image
            scale_factor = (1, 1)
            new_width, new_height = width, height
        
        # MediaPipe processing
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            raise ValueError("Pose non rilevata nell'immagine")
        
        # Estrazione landmarks con scaling
        landmarks = results.pose_landmarks.landmark
        
        def landmark_to_pixel(landmark):
            x = int(landmark.x * new_width * scale_factor[0])
            y = int(landmark.y * new_height * scale_factor[1])
            return max(0, min(x, width-1)), max(0, min(y, height-1))
        
        # Punti chiave del torso
        left_shoulder = landmark_to_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = landmark_to_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        left_hip = landmark_to_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_hip = landmark_to_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        
        # Calcoli dimensioni
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        torso_height = max(left_hip[1], right_hip[1]) - min(left_shoulder[1], right_shoulder[1])
        
        # Creazione maschera
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Margini generosi per cattura completa
        margin_x = int(shoulder_width * 0.35)
        margin_y_top = int(torso_height * 0.15)
        margin_y_bottom = int(torso_height * 0.25)
        
        # Poligono del torso
        torso_points = np.array([
            [left_shoulder[0] - margin_x, left_shoulder[1] - margin_y_top],
            [right_shoulder[0] + margin_x, right_shoulder[1] - margin_y_top],
            [right_hip[0] + margin_x, right_hip[1] + margin_y_bottom],
            [left_hip[0] - margin_x, left_hip[1] + margin_y_bottom]
        ], dtype=np.int32)
        
        # Clip ai bordi
        torso_points[:, 0] = np.clip(torso_points[:, 0], 0, width - 1)
        torso_points[:, 1] = np.clip(torso_points[:, 1], 0, height - 1)
        
        cv2.fillPoly(mask, [torso_points], 255)
        
        # Miglioramento morfologico
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        
        return mask, (left_shoulder, right_shoulder, left_hip, right_hip, shoulder_width, torso_height)

    def prepare_garment(self, garment_path, target_size):
        """Prepara indumento - ottimizzato per Mac"""
        garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
        if garment is None:
            raise ValueError(f"Impossibile caricare: {garment_path}")
        
        h, w = garment.shape[:2]
        target_w, target_h = target_size
        
        # Mantieni proporzioni
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale * 0.9), int(h * scale * 0.9)  # Leggermente più piccolo
        
        garment_resized = cv2.resize(garment, (new_w, new_h))
        
        # Background appropriato
        if garment_resized.shape[2] == 4:
            result = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        else:
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
        # Centra
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = garment_resized
        
        return result

    def generate_tryon(self, person_image_path, garment_image_path, output_path="mac_catvton_result.png"):
        """Virtual try-on ottimizzato per Mac"""
        
        # Caricamento immagine
        person_image = cv2.imread(person_image_path)
        if person_image is None:
            raise ValueError(f"Impossibile caricare: {person_image_path}")
        
        print("Rilevamento pose...")
        
        # Pose e maschera
        mask, pose_info = self.detect_pose_and_create_mask(person_image)
        left_shoulder, right_shoulder, left_hip, right_hip, shoulder_width, torso_height = pose_info
        
        # Preparazione indumento
        target_size = (int(shoulder_width * 1.5), int(torso_height * 0.9))
        garment_prepared = self.prepare_garment(garment_image_path, target_size)
        
        # Ridimensionamento per performance su Mac
        max_dimension = 768  # Limite per Mac
        h, w = person_image.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            person_image = cv2.resize(person_image, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h))
        
        # Conversioni PIL
        person_pil = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        print("Generazione con Stable Diffusion...")
        
        # Prompt ottimizzato
        prompt = "a person wearing a well-fitted shirt, natural lighting, high quality, realistic clothing texture, professional photo"
        negative_prompt = "blurry, low quality, distorted, poorly fitted, unrealistic, bad anatomy, artifacts, noise"
        
        # Generazione con gestione memoria Mac
        try:
            # Configurazione ottimizzata per Mac
            generator = torch.Generator(device=self.device).manual_seed(42)
            
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=person_pil,
                    mask_image=mask_pil,
                    num_inference_steps=25,  # Ridotto per Mac
                    guidance_scale=7.5,
                    strength=0.85,
                    generator=generator
                ).images[0]
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Memoria insufficiente, provo con impostazioni ridotte...")
                # Riduzione ulteriore
                person_small = person_pil.resize((384, 512))
                mask_small = mask_pil.resize((384, 512))
                
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=person_small,
                        mask_image=mask_small,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        strength=0.8,
                        generator=generator
                    ).images[0]
                    
                # Ridimensiona al formato originale
                result = result.resize(person_pil.size)
            else:
                raise e
        
        # Salvataggio
        result.save(output_path)
        print(f"✓ Risultato salvato: {output_path}")
        
        # Comparazione
        comparison = Image.new('RGB', (person_pil.width * 2, person_pil.height))
        comparison.paste(person_pil, (0, 0))
        comparison.paste(result, (person_pil.width, 0))
        comparison.save(output_path.replace('.png', '_comparison.png'))
        
        # Debug files
        mask_pil.save(output_path.replace('.png', '_mask.png'))
        
        return result

def main_mac():
    """Funzione principale per Mac"""
    print("=== CatVTON per Mac ===")
    
    # Verifica sistema
    if torch.backends.mps.is_available():
        print("✓ Apple Silicon GPU rilevato")
    elif torch.cuda.is_available():
        print("✓ NVIDIA GPU rilevato")
    else:
        print("⚠ Usando CPU (sarà più lento)")
        proceed = input("Continuare? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Inizializzazione
    try:
        catvton = CatVTONMacPipeline()
    except Exception as e:
        print(f"Errore inizializzazione: {e}")
        print("Prova a installare le dipendenze:")
        print("pip install torch torchvision diffusers transformers mediapipe opencv-python pillow")
        return
    
    # Input files
    person_image = "tst.jpg"
    garment_image = "Soggetto.png"
    output_file = "mac_catvton_result.png"
    
    # Verifica files
    if not os.path.exists(person_image):
        print(f"❌ File non trovato: {person_image}")
        return
    if not os.path.exists(garment_image):
        print(f"❌ File non trovato: {garment_image}")
        return
    
    # Esecuzione
    try:
        print("Avvio virtual try-on...")
        result = catvton.generate_tryon(person_image, garment_image, output_file)
        
        print("✅ Virtual try-on completato!")
        print(f"📁 Risultato: {output_file}")
        print(f"📁 Comparazione: {output_file.replace('.png', '_comparison.png')}")
        
        # Visualizzazione opzionale
        try:
            result_cv = cv2.imread(output_file)
            cv2.imshow('Risultato Mac CatVTON', result_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("(Visualizzazione non disponibile)")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        print("\nSuggerimenti:")
        print("1. Verifica che le immagini siano valide")
        print("2. Prova con immagini più piccole")
        print("3. Chiudi altre applicazioni per liberare memoria")
        print("4. Considera l'Opzione 1 (SegFormer) come alternativa più leggera")

if __name__ == "__main__":
    main_mac()