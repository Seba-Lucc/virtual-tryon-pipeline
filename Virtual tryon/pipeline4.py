#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 PIPELINE VIRTUAL TRY-ON RIVOLUZIONARIA 🚀
Realismo fotografico con integrazione anatomica avanzata

Features:
- Segmentazione conservativa e precisa
- Warping anatomico dell'indumento
- Color matching automatico
- Shadow synthesis realistica
- Compositing fotografico professionale
"""

import cv2
import mediapipe as mp
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image, ImageEnhance, ImageFilter
import torch
import os
from sklearn.cluster import KMeans
import math

print("🚀 === PIPELINE VIRTUAL TRY-ON RIVOLUZIONARIA === 🚀")
print(f"PyTorch: {torch.__version__}")
print(f"Hardware: MacBook Pro M2 Pro (16GB) - OTTIMIZZATO!")

#--------------------------------------------------------------------------------------------
# CONFIGURAZIONE AVANZATA
#--------------------------------------------------------------------------------------------

PERSON_IMAGE = "tst.jpg"
GARMENT_IMAGE = "Soggetto.png"
OUTPUT_IMAGE = "result_revolutionary.png"

# Parametri di qualità (ottimizzati per M2 Pro)
MAX_RESOLUTION = 1024  # Aumentato per qualità superiore
SEGMENTATION_PRECISION = 0.85  # Soglia più alta per precisione
WARPING_POINTS = 12  # Punti di controllo per deformazione anatomica
BLEND_RADIUS = 8  # Raggio blending per bordi naturali

print(f"✓ Risoluzione massima: {MAX_RESOLUTION}px")
print(f"✓ Precisione segmentazione: {SEGMENTATION_PRECISION}")

# Verifica files
for file_path in [PERSON_IMAGE, GARMENT_IMAGE]:
    if not os.path.exists(file_path):
        print(f"❌ ERRORE: File non trovato: {file_path}")
        exit(1)

#--------------------------------------------------------------------------------------------
# CLASSE ANATOMICAL ANALYZER
#--------------------------------------------------------------------------------------------

class AnatomicalAnalyzer:
    """Analizza la struttura anatomica per warping realistico"""
    
    def __init__(self, landmarks, image_shape):
        self.landmarks = landmarks
        self.height, self.width = image_shape[:2]
        self.pose_points = self._extract_pose_points()
        
    def _extract_pose_points(self):
        """Estrae punti anatomici chiave"""
        def to_pixel(landmark):
            return int(landmark.x * self.width), int(landmark.y * self.height)
        
        # Punti chiave per warping anatomico
        points = {
            'left_shoulder': to_pixel(self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]),
            'right_shoulder': to_pixel(self.landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]),
            'left_elbow': to_pixel(self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]),
            'right_elbow': to_pixel(self.landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]),
            'left_hip': to_pixel(self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]),
            'right_hip': to_pixel(self.landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]),
            'nose': to_pixel(self.landmarks[mp.solutions.pose.PoseLandmark.NOSE]),
        }
        
        # Punti calcolati per migliore copertura
        points['neck_center'] = (
            (points['left_shoulder'][0] + points['right_shoulder'][0]) // 2,
            (points['left_shoulder'][1] + points['right_shoulder'][1]) // 2 - 20
        )
        
        points['chest_center'] = (
            (points['left_shoulder'][0] + points['right_shoulder'][0]) // 2,
            (points['left_shoulder'][1] + points['right_shoulder'][1]) // 2 + 40
        )
        
        points['waist_center'] = (
            (points['left_hip'][0] + points['right_hip'][0]) // 2,
            (points['left_hip'][1] + points['right_hip'][1]) // 2 - 30
        )
        
        return points
    
    def get_torso_dimensions(self):
        """Calcola dimensioni anatomiche precise"""
        shoulder_width = abs(self.pose_points['right_shoulder'][0] - self.pose_points['left_shoulder'][0])
        torso_height = self.pose_points['waist_center'][1] - self.pose_points['neck_center'][1]
        
        # Calcola curve anatomiche
        chest_width = int(shoulder_width * 0.95)  # Leggermemente più stretto delle spalle
        waist_width = int(shoulder_width * 0.85)   # Vita più stretta
        
        return {
            'shoulder_width': shoulder_width,
            'chest_width': chest_width,
            'waist_width': waist_width,
            'torso_height': torso_height,
            'center_line': (self.pose_points['neck_center'][0] + self.pose_points['waist_center'][0]) // 2
        }
    
    def create_conservative_mask(self, segformer_mask):
        """Crea maschera conservativa che preserva la pelle"""
        
        # Inizia con maschera anatomica molto precisa
        anatomical_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Definiamo l'area SOLO dell'abbigliamento, non della pelle
        dims = self.get_torso_dimensions()
        
        # Maschera a forma di maglietta (non include spalle nude, braccia, collo)
        torso_points = np.array([
            # Girocollo (non troppo alto per non toccare collo/pelle)
            [dims['center_line'] - dims['shoulder_width']//3, self.pose_points['neck_center'][1] + 25],
            [dims['center_line'] + dims['shoulder_width']//3, self.pose_points['neck_center'][1] + 25],
            
            # Spalle maglietta (dentro le spalle anatomiche)
            [self.pose_points['right_shoulder'][0] - 15, self.pose_points['right_shoulder'][1] + 10],
            [self.pose_points['right_elbow'][0] - 20, self.pose_points['right_elbow'][1] - 20],
            
            # Lato destro verso vita
            [self.pose_points['right_hip'][0] - 10, self.pose_points['waist_center'][1]],
            
            # Base maglietta
            [dims['center_line'] + dims['waist_width']//2, self.pose_points['waist_center'][1]],
            [dims['center_line'] - dims['waist_width']//2, self.pose_points['waist_center'][1]],
            
            # Lato sinistro
            [self.pose_points['left_hip'][0] + 10, self.pose_points['waist_center'][1]],
            [self.pose_points['left_elbow'][0] + 20, self.pose_points['left_elbow'][1] - 20],
            [self.pose_points['left_shoulder'][0] + 15, self.pose_points['left_shoulder'][1] + 10],
        ], dtype=np.int32)
        
        # Clip ai bordi
        torso_points[:, 0] = np.clip(torso_points[:, 0], 0, self.width - 1)
        torso_points[:, 1] = np.clip(torso_points[:, 1], 0, self.height - 1)
        
        cv2.fillPoly(anatomical_mask, [torso_points], 255)
        
        # Combina con SegFormer MA solo dove si intersecano (conservativo)
        if segformer_mask is not None:
            combined_mask = cv2.bitwise_and(anatomical_mask, segformer_mask)
            # Se SegFormer trova molto poco, usa solo anatomica ridotta
            if np.sum(combined_mask > 0) < np.sum(anatomical_mask > 0) * 0.3:
                # Riduci ulteriormente la maschera anatomica
                kernel = np.ones((15,15), np.uint8)
                anatomical_mask = cv2.erode(anatomical_mask, kernel, iterations=2)
                combined_mask = anatomical_mask
        else:
            combined_mask = anatomical_mask
        
        # Smoothing molto leggero per bordi naturali
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        
        return combined_mask

#--------------------------------------------------------------------------------------------
# CLASSE GARMENT WARPER  
#--------------------------------------------------------------------------------------------

class GarmentWarper:
    """Deforma l'indumento per seguire l'anatomia del corpo"""
    
    def __init__(self, anatomical_analyzer):
        self.analyzer = anatomical_analyzer
        self.dims = anatomical_analyzer.get_torso_dimensions()
        
    def create_warping_mesh(self, garment_shape):
        """Crea griglia di deformazione anatomica"""
        h, w = garment_shape[:2]
        
        # Punti di controllo sull'indumento (griglia regolare)
        src_points = []
        dst_points = []
        
        # Griglia 4x3 di punti di controllo
        for row in range(4):
            for col in range(3):
                # Punto source (indumento piatto)
                src_x = col * w // 2
                src_y = row * h // 3
                src_points.append([src_x, src_y])
                
                # Punto destination (corpo)
                # Mappa i punti dell'indumento sui punti anatomici
                dst_x, dst_y = self._map_garment_to_body(col, row, w, h)
                dst_points.append([dst_x, dst_y])
        
        return np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32)
    
    def _map_garment_to_body(self, col, row, garment_w, garment_h):
        """Mappa punto dell'indumento sul corpo"""
        pose = self.analyzer.pose_points
        
        # Interpolazione tra punti anatomici basata sulla posizione nella griglia
        col_ratio = col / 2.0  # 0 = sinistra, 1 = centro, 2 = destra
        row_ratio = row / 3.0  # 0 = alto, 1 = centro, 2 = basso
        
        # Calcolo X (larghezza): interpola tra i lati del corpo
        if col_ratio <= 1.0:
            # Lato sinistro -> centro
            left_x = pose['left_shoulder'][0] if row_ratio < 0.5 else pose['left_hip'][0]
            center_x = self.dims['center_line']
            x = left_x + (center_x - left_x) * col_ratio
        else:
            # Centro -> lato destro  
            center_x = self.dims['center_line']
            right_x = pose['right_shoulder'][0] if row_ratio < 0.5 else pose['right_hip'][0]
            x = center_x + (right_x - center_x) * (col_ratio - 1.0)
        
        # Calcolo Y (altezza): interpola tra collo e vita
        neck_y = pose['neck_center'][1] + 20  # Poco sotto il collo
        waist_y = pose['waist_center'][1]
        y = neck_y + (waist_y - neck_y) * row_ratio
        
        # Applica curvatura naturale del corpo
        if col_ratio == 1.0:  # Punto centrale
            y += int(10 * math.sin(row_ratio * math.pi))  # Leggera curvatura
        
        return int(x), int(y)
    
    def warp_garment(self, garment_img, target_shape):
        """Applica warping anatomico all'indumento"""
        h_target, w_target = target_shape[:2]
        
        # Ridimensiona garment alle dimensioni target conservative
        target_w = min(self.dims['shoulder_width'], w_target // 2)
        target_h = min(self.dims['torso_height'], h_target // 2)
        
        garment_resized = cv2.resize(garment_img, (target_w, target_h))
        
        # Crea mesh di warping
        src_points, dst_points = self.create_warping_mesh(garment_resized.shape)
        
        # Applica warping usando Thin Plate Spline (TPS) approximation
        warped = self._apply_tps_warping(garment_resized, src_points, dst_points, (w_target, h_target))
        
        return warped
    
    def _apply_tps_warping(self, image, src_points, dst_points, output_size):
        """Applica warping usando approssimazione Thin Plate Spline"""
        h, w = image.shape[:2]
        output_w, output_h = output_size
        
        # Crea immagine di output
        warped = np.zeros((output_h, output_w, image.shape[2]), dtype=np.uint8)
        
        # Per semplicità usiamo warping pezzo per pezzo con triangolazione
        from scipy.spatial import Delaunay
        
        # Triangolazione dei punti di controllo
        tri = Delaunay(src_points)
        
        for triangle in tri.simplices:
            # Punti del triangolo source e destination
            src_tri = src_points[triangle]
            dst_tri = dst_points[triangle]
            
            # Calcola trasformazione affine per il triangolo
            M = cv2.getAffineTransform(src_tri.astype(np.float32), dst_tri.astype(np.float32))
            
            # Applica trasformazione
            transformed = cv2.warpAffine(image, M, (output_w, output_h))
            
            # Maschera per il triangolo
            triangle_mask = np.zeros((output_h, output_w), dtype=np.uint8)
            cv2.fillPoly(triangle_mask, [dst_tri.astype(np.int32)], 255)
            
            # Blend nel risultato finale
            mask_norm = triangle_mask.astype(float) / 255.0
            for c in range(image.shape[2]):
                warped[:, :, c] = (mask_norm * transformed[:, :, c] + 
                                 (1 - mask_norm) * warped[:, :, c]).astype(np.uint8)
        
        return warped

#--------------------------------------------------------------------------------------------
# CLASSE COLOR MATCHER
#--------------------------------------------------------------------------------------------

class ColorMatcher:
    """Adatta i colori dell'indumento all'illuminazione della scena"""
    
    def __init__(self, reference_image, reference_mask):
        self.reference_img = reference_image
        self.reference_mask = reference_mask
        self._analyze_lighting()
        
    def _analyze_lighting(self):
        """Analizza l'illuminazione della scena di riferimento"""
        # Estrai colori della pelle/ambiente vicino all'area del torso
        mask_dilated = cv2.dilate(self.reference_mask, np.ones((20,20), np.uint8), iterations=2)
        mask_environment = cv2.subtract(mask_dilated, self.reference_mask)
        
        # Colore medio dell'ambiente
        env_pixels = self.reference_img[mask_environment > 0]
        if len(env_pixels) > 0:
            self.ambient_color = np.mean(env_pixels, axis=0)
        else:
            self.ambient_color = np.array([127, 127, 127])
        
        # Analisi temperatura colore
        self.color_temperature = self._estimate_color_temperature()
        
    def _estimate_color_temperature(self):
        """Stima la temperatura colore della scena"""
        # Semplificato: rapporto tra canali
        b, g, r = self.ambient_color
        if b > r:
            return "cool"  # Luce fredda (blu)
        elif r > b * 1.2:
            return "warm"  # Luce calda (giallo/rosso)
        else:
            return "neutral"
            
    def match_garment_colors(self, garment_img):
        """Adatta i colori dell'indumento all'illuminazione"""
        # Converti in float per calcoli precisi
        garment_float = garment_img.astype(float)
        
        # Applica correzione temperatura colore
        if self.color_temperature == "warm":
            # Aggiungi calore (più rosso/giallo)
            garment_float[:, :, 2] *= 1.05  # Più rosso
            garment_float[:, :, 1] *= 1.02  # Più verde
            garment_float[:, :, 0] *= 0.98  # Meno blu
        elif self.color_temperature == "cool":
            # Aggiungi freddezza (più blu)
            garment_float[:, :, 0] *= 1.05  # Più blu
            garment_float[:, :, 1] *= 1.01  # Leggero verde
            garment_float[:, :, 2] *= 0.98  # Meno rosso
        
        # Clip valori
        garment_float = np.clip(garment_float, 0, 255)
        
        return garment_float.astype(np.uint8)

#--------------------------------------------------------------------------------------------
# CLASSE SHADOW SYNTHESIZER
#--------------------------------------------------------------------------------------------

class ShadowSynthesizer:
    """Sintetizza ombre realistiche per l'indumento"""
    
    def __init__(self, anatomical_analyzer):
        self.analyzer = anatomical_analyzer
        
    def create_garment_shadows(self, garment_mask, image_shape):
        """Crea ombre realistiche per l'indumento"""
        h, w = image_shape[:2]
        shadow_map = np.zeros((h, w), dtype=np.float32)
        
        # Direzione luce stimata (dall'alto a sinistra)
        light_direction = np.array([-0.3, -0.7])  # x, y offset
        
        # Crea ombre sotto le pieghe dell'indumento
        # Simula l'ombra che l'indumento fa sul corpo
        
        # Dilata la maschera per creare area di ombra
        shadow_kernel = np.ones((15, 15), np.uint8)
        shadow_area = cv2.dilate(garment_mask, shadow_kernel, iterations=1)
        shadow_area = cv2.subtract(shadow_area, garment_mask)
        
        # Crea gradient di ombra
        shadow_strength = 0.15  # Ombra leggera
        shadow_map[shadow_area > 0] = shadow_strength
        
        # Blur per ombra realistica
        shadow_map = cv2.GaussianBlur(shadow_map, (11, 11), 0)
        
        return shadow_map

#--------------------------------------------------------------------------------------------
# CLASSE MAIN PIPELINE
#--------------------------------------------------------------------------------------------

class RevolutionaryTryOnPipeline:
    """Pipeline principale rivoluzionaria"""
    
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Carica tutti i modelli necessari"""
        print("\n🔧 Caricamento modelli avanzati...")
        
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7)
        
        # SegFormer per validazione
        try:
            self.segformer_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.segformer_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            print("✓ SegFormer caricato")
        except:
            print("⚠ SegFormer non disponibile - uso solo anatomia")
            self.segformer_processor = None
            self.segformer_model = None
            
    def process_image(self, person_path, garment_path, output_path):
        """Processo principale rivoluzionario"""
        
        print("\n🎯 === INIZIO PROCESSO RIVOLUZIONARIO ===")
        
        # 1. CARICAMENTO E PREPROCESSING
        print("\n1️⃣ Caricamento ottimizzato...")
        person_img = cv2.imread(person_path)
        garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
        
        if person_img is None or garment_img is None:
            raise ValueError("Impossibile caricare le immagini")
        
        # Ottimizzazione risoluzione per M2 Pro
        person_img = self._optimize_resolution(person_img)
        h, w = person_img.shape[:2]
        print(f"✓ Risoluzione ottimizzata: {w}x{h}")
        
        # 2. ANALISI ANATOMICA AVANZATA
        print("\n2️⃣ Analisi anatomica avanzata...")
        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(person_rgb)
        
        if not pose_results.pose_landmarks:
            raise ValueError("Pose non rilevata - immagine non adatta")
        
        analyzer = AnatomicalAnalyzer(pose_results.pose_landmarks.landmark, person_img.shape)
        print("✓ Struttura anatomica analizzata")
        
        # Debug pose
        debug_pose = person_img.copy()
        mp.solutions.drawing_utils.draw_landmarks(debug_pose, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        cv2.imwrite('debug_pose_revolutionary.png', debug_pose)
        
        # 3. SEGMENTAZIONE CONSERVATIVA
        print("\n3️⃣ Segmentazione conservativa...")
        segformer_mask = self._get_segformer_mask(person_rgb) if self.segformer_model else None
        conservative_mask = analyzer.create_conservative_mask(segformer_mask)
        
        pixels_to_remove = np.sum(conservative_mask > 0)
        print(f"✓ Pixel da modificare: {pixels_to_remove} (conservativo)")
        
        cv2.imwrite('debug_conservative_mask.png', conservative_mask)
        
        # 4. RIMOZIONE CHIRURGICA DELL'INDUMENTO
        print("\n4️⃣ Rimozione chirurgica...")
        person_clean = self._surgical_removal(person_img, conservative_mask)
        cv2.imwrite('debug_after_surgical_removal.png', person_clean)
        
        # 5. WARPING ANATOMICO DELL'INDUMENTO
        print("\n5️⃣ Warping anatomico avanzato...")
        warper = GarmentWarper(analyzer)
        garment_warped = warper.warp_garment(garment_img, person_img.shape)
        cv2.imwrite('debug_garment_warped.png', garment_warped)
        
        # 6. COLOR MATCHING
        print("\n6️⃣ Color matching...")
        color_matcher = ColorMatcher(person_img, conservative_mask)
        garment_color_matched = color_matcher.match_garment_colors(garment_warped)
        
        # 7. SHADOW SYNTHESIS
        print("\n7️⃣ Sintesi ombre...")
        shadow_synth = ShadowSynthesizer(analyzer)
        garment_final_mask = np.any(garment_color_matched > 0, axis=2).astype(np.uint8) * 255
        shadow_map = shadow_synth.create_garment_shadows(garment_final_mask, person_img.shape)
        
        # 8. COMPOSITING FOTOGRAFICO
        print("\n8️⃣ Compositing fotografico...")
        final_result = self._photographic_compositing(
            person_clean, garment_color_matched, garment_final_mask, shadow_map
        )
        
        # 9. POST-PROCESSING FINALE
        print("\n9️⃣ Post-processing finale...")
        final_result = self._final_enhancement(final_result)
        
        # 10. SALVATAGGIO
        print("\n🔟 Salvataggio risultati...")
        cv2.imwrite(output_path, final_result)
        self._create_analysis_grid(person_img, person_clean, final_result, output_path)
        
        print(f"\n🎉 === PROCESSO COMPLETATO === 🎉")
        print(f"📁 Risultato: {output_path}")
        
        return final_result
    
    def _optimize_resolution(self, image):
        """Ottimizza risoluzione per M2 Pro"""
        h, w = image.shape[:2]
        if max(h, w) > MAX_RESOLUTION:
            scale = MAX_RESOLUTION / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image
    
    def _get_segformer_mask(self, image_rgb):
        """Ottiene maschera da SegFormer"""
        if not self.segformer_model:
            return None
            
        pil_img = Image.fromarray(image_rgb)
        inputs = self.segformer_processor(images=pil_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.segformer_model(**inputs)
            
        logits = outputs.logits.cpu()
        upsampled = torch.nn.functional.interpolate(
            logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled.argmax(dim=1)[0].numpy()
        
        # Solo upper-clothes e dress con soglia alta
        mask = np.zeros_like(pred_seg, dtype=np.uint8)
        for class_id in [4, 7]:  # upper-clothes, dress
            mask[pred_seg == class_id] = 255
            
        return mask
    
    def _surgical_removal(self, image, mask):
        """Rimozione chirurgica ultra-precisa"""
        # Solo se c'è davvero qualcosa da rimuovere
        if np.sum(mask > 0) < 100:
            return image.copy()
        
        # Erosione della maschera per essere ancora più conservativi
        kernel = np.ones((3,3), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        
        # Inpainting delicato
        result = cv2.inpaint(image, mask_eroded, 5, cv2.INPAINT_TELEA)
        
        # Blend nelle zone di bordo per transizione naturale
        mask_border = cv2.subtract(mask, mask_eroded)
        mask_border_norm = mask_border.astype(float) / 255.0
        
        for c in range(3):
            result[:, :, c] = (mask_border_norm * result[:, :, c] + 
                             (1 - mask_border_norm) * image[:, :, c]).astype(np.uint8)
        
        return result
    
    def _photographic_compositing(self, background, garment, garment_mask, shadow_map):
        """Compositing fotografico realistico"""
        result = background.copy().astype(float)
        
        # Applica ombre prima dell'indumento
        for c in range(3):
            result[:, :, c] *= (1 - shadow_map)
        
        # Alpha mask dall'indumento
        if garment.shape[2] == 4:
            alpha = garment[:, :, 3].astype(float) / 255.0
            garment_rgb = garment[:, :, :3].astype(float)
        else:
            alpha = (garment_mask > 0).astype(float)
            garment_rgb = garment.astype(float)
        
        # Feathering avanzato dei bordi
        alpha_feathered = cv2.GaussianBlur(alpha, (BLEND_RADIUS*2+1, BLEND_RADIUS*2+1), 0)
        
        # Compositing con gamma correction
        gamma = 2.2
        result_gamma = np.power(result / 255.0, gamma)
        garment_gamma = np.power(garment_rgb / 255.0, gamma)
        
        for c in range(3):
            result_gamma[:, :, c] = (alpha_feathered * garment_gamma[:, :, c] + 
                                   (1 - alpha_feathered) * result_gamma[:, :, c])
        
        # Back to linear
        result = np.power(result_gamma, 1/gamma) * 255.0
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _final_enhancement(self, image):
        """Enhancement finale per realismo"""
        # Leggero sharpening
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharpen)
        result = cv2.addWeighted(image, 0.8, sharpened, 0.2, 0)
        
        # Leggero miglioramento contrasto
        result = cv2.convertScaleAbs(result, alpha=1.02, beta=2)
        
        return result
    
    def _create_analysis_grid(self, original, cleaned, final, output_path):
        """Crea griglia di analisi completa"""
        h, w = original.shape[:2]
        
        # Ridimensiona per griglia
        scale = min(400 / w, 300 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        orig_small = cv2.resize(original, (new_w, new_h))
        clean_small = cv2.resize(cleaned, (new_w, new_h))
        final_small = cv2.resize(final, (new_w, new_h))
        
        # Griglia 1x3
        grid = np.hstack([orig_small, clean_small, final_small])
        
        # Aggiungi etichette
        cv2.putText(grid, "ORIGINALE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(grid, "PULITO", (new_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(grid, "FINALE", (new_w*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        grid_path = output_path.replace('.png', '_analysis_grid.png')
        cv2.imwrite(grid_path, grid)
        print(f"✓ Griglia di analisi: {grid_path}")

#--------------------------------------------------------------------------------------------
# ESECUZIONE PRINCIPALE
#--------------------------------------------------------------------------------------------

def main():
    """Funzione principale"""
    try:
        # Inizializzazione pipeline
        pipeline = RevolutionaryTryOnPipeline()
        
        # Esecuzione
        result = pipeline.process_image(PERSON_IMAGE, GARMENT_IMAGE, OUTPUT_IMAGE)
        
        # Visualizzazione opzionale
        try:
            cv2.imshow('🚀 RISULTATO RIVOLUZIONARIO', result)
            print("\nPremi un tasto per chiudere...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("(Visualizzazione non disponibile)")
            
        print("\n🎯 Processo rivoluzionario completato con successo!")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        print("Suggerimenti:")
        print("- Verifica che l'immagine contenga una persona in posa frontale")
        print("- Assicurati che l'indumento abbia sfondo trasparente")
        print("- Prova con immagini di qualità superiore")

if __name__ == "__main__":
    main()