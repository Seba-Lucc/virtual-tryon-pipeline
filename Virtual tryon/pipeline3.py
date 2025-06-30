#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Virtual Try-On MIGLIORATA - Risolve problemi di segmentazione e sizing
"""

import cv2
import mediapipe as mp
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import os

print("=== Pipeline Virtual Try-On MIGLIORATA ===")
print(f"PyTorch: {torch.__version__}")

#--------------------------------------------------------------------------------------------
# CONFIGURAZIONE
#--------------------------------------------------------------------------------------------

PERSON_IMAGE = "tst.jpg"
GARMENT_IMAGE = "Soggetto.png"
OUTPUT_IMAGE = "result_improved.png"

# Verifica files
for file_path in [PERSON_IMAGE, GARMENT_IMAGE]:
    if not os.path.exists(file_path):
        print(f"❌ ERRORE: File non trovato: {file_path}")
        exit(1)

print(f"✓ File persona: {PERSON_IMAGE}")
print(f"✓ File indumento: {GARMENT_IMAGE}")

#--------------------------------------------------------------------------------------------
# CARICAMENTO E PREPROCESSING MIGLIORATO
#--------------------------------------------------------------------------------------------

print("\n1. Caricamento e preprocessing...")
image = cv2.imread(PERSON_IMAGE)
if image is None:
    raise IOError(f"Impossibile aprire: {PERSON_IMAGE}")

# Manteniamo una risoluzione più alta per migliore segmentazione
max_width = 800  # Aumentato da 640
original_height, original_width = image.shape[:2]

if image.shape[1] > max_width:
    scale = max_width / image.shape[1]
    new_width = max_width
    new_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (new_width, new_height))
    print(f"✓ Ridimensionato a: {new_width}x{new_height}")
else:
    new_width, new_height = original_width, original_height
    print(f"✓ Dimensioni mantenute: {new_width}x{new_height}")

#--------------------------------------------------------------------------------------------
# CARICAMENTO MODELLI
#--------------------------------------------------------------------------------------------

print("\n2. Caricamento modelli...")

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# SegFormer
try:
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    print("✓ Modelli caricati!")
except Exception as e:
    print(f"❌ Errore: {e}")
    exit(1)

#--------------------------------------------------------------------------------------------
# RILEVAMENTO POSE MIGLIORATO
#--------------------------------------------------------------------------------------------

print("\n3. Rilevamento pose...")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

if not results.pose_landmarks:
    print("❌ Pose non rilevata")
    exit(1)

print("✓ Pose rilevata!")

# Debug pose
debug_image = image.copy()
mp.solutions.drawing_utils.draw_landmarks(
    debug_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
)
cv2.imwrite('debug_pose_improved.png', debug_image)

#--------------------------------------------------------------------------------------------
# ANALISI CORPOREA DETTAGLIATA
#--------------------------------------------------------------------------------------------

print("\n4. Analisi corporea dettagliata...")
height, width, _ = image.shape
landmarks = results.pose_landmarks.landmark

def landmark_to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

# Estrazione punti estesi per migliore definizione torso
x_left_shoulder, y_left_shoulder = landmark_to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], width, height)
x_right_shoulder, y_right_shoulder = landmark_to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], width, height)
x_left_hip, y_left_hip = landmark_to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_HIP], width, height)
x_right_hip, y_right_hip = landmark_to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], width, height)

# AGGIUNGIAMO PUNTI AGGIUNTIVI per migliore copertura
x_left_elbow, y_left_elbow = landmark_to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], width, height)
x_right_elbow, y_right_elbow = landmark_to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], width, height)

# Punto centrale del torso per riferimento
center_x = (x_left_shoulder + x_right_shoulder) // 2
center_y = (y_left_shoulder + y_right_shoulder + y_left_hip + y_right_hip) // 4

shoulder_width = abs(x_right_shoulder - x_left_shoulder)
torso_height = max(y_left_hip, y_right_hip) - min(y_left_shoulder, y_right_shoulder)

print(f"✓ Larghezza spalle: {shoulder_width}px")
print(f"✓ Altezza torso: {torso_height}px")
print(f"✓ Centro torso: ({center_x}, {center_y})")

#--------------------------------------------------------------------------------------------
# SEGMENTAZIONE MIGLIORATA CON MULTIPLE STRATEGIE
#--------------------------------------------------------------------------------------------

print("\n5. Segmentazione avanzata...")

# SegFormer segmentation
pil_image = Image.fromarray(image_rgb)
inputs = processor(images=pil_image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits.cpu()
upsampled_logits = torch.nn.functional.interpolate(
    logits, size=pil_image.size[::-1], mode="bilinear", align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

# STRATEGIA 1: Segmentazione SegFormer base
segformer_mask = np.zeros_like(pred_seg, dtype=np.uint8)
clothing_classes = [4, 7]  # Upper-clothes, Dress
for class_id in clothing_classes:
    segformer_mask[pred_seg == class_id] = 255

print(f"✓ SegFormer trova {np.sum(segformer_mask > 0)} pixel")

# STRATEGIA 2: Maschera basata su colore (per catturare la maglietta rosa)
# Convertiamo in HSV per migliore rilevamento colore
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Range per maglietta rosa (da calibrare in base al colore specifico)
# Pink/Rosa in HSV: H=300-350, S=30-255, V=100-255
lower_pink1 = np.array([140, 30, 100])  # Rosa-magenta
upper_pink1 = np.array([180, 255, 255])
lower_pink2 = np.array([0, 30, 100])    # Rosa-rosso
upper_pink2 = np.array([20, 255, 255])

color_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
color_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
color_mask = cv2.bitwise_or(color_mask1, color_mask2)

print(f"✓ Rilevamento colore trova {np.sum(color_mask > 0)} pixel")

# STRATEGIA 3: Maschera anatomica estesa
anatomical_mask = np.zeros((height, width), dtype=np.uint8)

# Margini molto più generosi
margin_x = int(shoulder_width * 0.45)     # Aumentato da 0.3
margin_y_top = int(torso_height * 0.2)    # Aumentato da 0.15  
margin_y_bottom = int(torso_height * 0.3) # Aumentato da 0.25

# Poligono anatomico più preciso che segue meglio la forma del torso
torso_points = np.array([
    # Parte superiore più larga per spalle e braccia
    [x_left_shoulder - margin_x, y_left_shoulder - margin_y_top],
    [center_x, y_left_shoulder - margin_y_top],  # Punto centrale alto
    [x_right_shoulder + margin_x, y_right_shoulder - margin_y_top],
    
    # Lati che seguono le braccia
    [x_right_elbow + margin_x//2, y_right_elbow],
    [x_right_hip + margin_x//2, y_right_hip + margin_y_bottom],
    
    # Base del torso
    [center_x, y_right_hip + margin_y_bottom],
    [x_left_hip - margin_x//2, y_left_hip + margin_y_bottom],
    
    # Risalita lato sinistro
    [x_left_elbow - margin_x//2, y_left_elbow],
], dtype=np.int32)

# Clip ai bordi
torso_points[:, 0] = np.clip(torso_points[:, 0], 0, width - 1)
torso_points[:, 1] = np.clip(torso_points[:, 1], 0, height - 1)

cv2.fillPoly(anatomical_mask, [torso_points], 255)

print(f"✓ Maschera anatomica: {np.sum(anatomical_mask > 0)} pixel")

# COMBINAZIONE INTELLIGENTE delle maschere
print("\n6. Combinazione maschere intelligente...")

# Combina tutte le strategie
combined_mask = np.zeros_like(anatomical_mask)

# Prima: usa SegFormer se trova abbastanza pixel
if np.sum(segformer_mask > 0) > 1000:
    combined_mask = cv2.bitwise_or(combined_mask, segformer_mask)
    print("✓ SegFormer incluso")

# Seconda: aggiungi rilevamento colore nell'area anatomica
color_in_torso = cv2.bitwise_and(color_mask, anatomical_mask)
if np.sum(color_in_torso > 0) > 500:
    combined_mask = cv2.bitwise_or(combined_mask, color_in_torso)
    print("✓ Rilevamento colore incluso")

# Terza: se le maschere sono ancora troppo piccole, usa quella anatomica
if np.sum(combined_mask > 0) < shoulder_width * torso_height * 0.1:
    print("⚠ Maschere troppo piccole, uso maschera anatomica")
    combined_mask = anatomical_mask

# Miglioramento morfologico aggressivo
kernel_large = np.ones((11,11), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

# Dilatazione per catturare bordi
combined_mask = cv2.dilate(combined_mask, np.ones((5,5), np.uint8), iterations=3)

# Blur finale per transizioni morbide
combined_mask = cv2.GaussianBlur(combined_mask, (11, 11), 0)

final_pixels = np.sum(combined_mask > 0)
print(f"✓ Maschera finale: {final_pixels} pixel")

# Salvataggio maschere debug
cv2.imwrite('debug_segformer_mask.png', segformer_mask)
cv2.imwrite('debug_color_mask.png', color_mask)
cv2.imwrite('debug_anatomical_mask.png', anatomical_mask)
cv2.imwrite('debug_combined_mask.png', combined_mask)

#--------------------------------------------------------------------------------------------
# RIMOZIONE AGGRESSIVA INDUMENTO
#--------------------------------------------------------------------------------------------

print("\n7. Rimozione indumento (multi-passaggio)...")

if final_pixels > 100:
    # Passaggio 1: TELEA (conserva strutture)
    image_temp = cv2.inpaint(image, combined_mask, 15, cv2.INPAINT_TELEA)
    
    # Passaggio 2: NS (riempie meglio)
    image_clean = cv2.inpaint(image_temp, combined_mask, 10, cv2.INPAINT_NS)
    
    # Passaggio 3: Inpainting aggiuntivo su aree residue
    # Rileva aree che potrebbero ancora contenere residui della maglietta originale
    diff = cv2.absdiff(image, image_clean)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, residue_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    residue_mask = cv2.bitwise_and(residue_mask, combined_mask)
    
    if np.sum(residue_mask > 0) > 100:
        image_clean = cv2.inpaint(image_clean, residue_mask, 8, cv2.INPAINT_TELEA)
    
    print("✓ Rimozione completata (3 passaggi)")
else:
    image_clean = image.copy()
    print("⚠ Rimozione saltata")

cv2.imwrite('debug_after_inpaint_improved.png', image_clean)

#--------------------------------------------------------------------------------------------
# RIDIMENSIONAMENTO INDUMENTO OTTIMIZZATO
#--------------------------------------------------------------------------------------------

print("\n8. Ridimensionamento indumento ottimizzato...")

shirt = cv2.imread(GARMENT_IMAGE, cv2.IMREAD_UNCHANGED)
if shirt is None:
    raise IOError(f"Impossibile caricare: {GARMENT_IMAGE}")

print(f"✓ Indumento originale: {shirt.shape}")

# ALGORITMO DI SIZING ANTROPOMETRICO
# Calcoliamo dimensioni realistiche basate sulle proporzioni corporee

# Fattori calibrati per look realistico
width_factor = 2.2   # Molto più generoso (era 1.6)
height_factor = 1.1  # Più alto per coprire meglio (era 0.85)

# Calcolo dimensioni target
target_shirt_width = int(shoulder_width * width_factor)
target_shirt_height = int(torso_height * height_factor)

# Assicuriamoci che non sia troppo grande per l'immagine
max_shirt_width = int(width * 0.9)
max_shirt_height = int(height * 0.8)

target_shirt_width = min(target_shirt_width, max_shirt_width)
target_shirt_height = min(target_shirt_height, max_shirt_height)

print(f"✓ Dimensioni target calcolate: {target_shirt_width}x{target_shirt_height}")

# Ridimensionamento mantenendo proporzioni
original_shirt_ratio = shirt.shape[1] / shirt.shape[0]
target_ratio = target_shirt_width / target_shirt_height

if original_shirt_ratio > target_ratio:
    # Indumento più largo del target - limitiamo dalla larghezza
    final_shirt_width = target_shirt_width
    final_shirt_height = int(target_shirt_width / original_shirt_ratio)
else:
    # Indumento più alto del target - limitiamo dall'altezza  
    final_shirt_height = target_shirt_height
    final_shirt_width = int(target_shirt_height * original_shirt_ratio)

# Controllo finale dimensioni
final_shirt_width = min(final_shirt_width, max_shirt_width)
final_shirt_height = min(final_shirt_height, max_shirt_height)

shirt_resized = cv2.resize(shirt, (final_shirt_width, final_shirt_height))
print(f"✓ Indumento ridimensionato a: {final_shirt_width}x{final_shirt_height}")

# Fattore di ingrandimento rispetto al precedente
scale_improvement = (final_shirt_width * final_shirt_height) / (shoulder_width * 1.4 * torso_height * 0.85)
print(f"✓ Fattore di ingrandimento: {scale_improvement:.2f}x")

#--------------------------------------------------------------------------------------------
# POSIZIONAMENTO INTELLIGENTE MIGLIORATO
#--------------------------------------------------------------------------------------------

print("\n9. Posizionamento intelligente...")

# Posizionamento orizzontale: centrato sulle spalle
x_start = center_x - final_shirt_width // 2

# Posizionamento verticale: calibrato per look naturale
# Inizia leggermente sopra le spalle per coprire bene il collo
vertical_offset = int(torso_height * 0.02)  # Molto meno offset (era 0.08)
y_start = min(y_left_shoulder, y_right_shoulder) - vertical_offset

# Controlli di sicurezza con bordi
x_start = max(5, min(x_start, width - final_shirt_width - 5))
y_start = max(5, min(y_start, height - final_shirt_height - 5))

print(f"✓ Posizione finale: ({x_start}, {y_start})")
print(f"✓ Copertura area: {final_shirt_width * final_shirt_height} pixel")

#--------------------------------------------------------------------------------------------
# APPLICAZIONE CON BLENDING MIGLIORATO
#--------------------------------------------------------------------------------------------

print("\n10. Applicazione indumento...")

if shirt_resized.shape[2] == 4:  # Ha canale alpha
    print("✓ Applicazione con trasparenza avanzata...")
    
    overlay_rgb = shirt_resized[..., :3].astype(float)
    overlay_alpha = shirt_resized[..., 3:].astype(float) / 255.0
    
    # ROI con controllo dimensioni
    roi_height = min(final_shirt_height, height - y_start)
    roi_width = min(final_shirt_width, width - x_start)
    
    roi = image_clean[y_start:y_start+roi_height, x_start:x_start+roi_width].astype(float)
    overlay_rgb_crop = overlay_rgb[:roi_height, :roi_width]
    overlay_alpha_crop = overlay_alpha[:roi_height, :roi_width]
    
    # Alpha blending migliorato con gamma correction
    gamma = 2.2
    overlay_gamma = np.power(overlay_rgb_crop / 255.0, gamma)
    roi_gamma = np.power(roi / 255.0, gamma)
    
    for c in range(3):
        roi_gamma[:, :, c] = (overlay_alpha_crop[:, :, 0] * overlay_gamma[:, :, c] + 
                             (1 - overlay_alpha_crop[:, :, 0]) * roi_gamma[:, :, c])
    
    roi_final = np.power(roi_gamma, 1/gamma) * 255.0
    image_clean[y_start:y_start+roi_height, x_start:x_start+roi_width] = roi_final.astype(np.uint8)
    
else:
    print("✓ Applicazione diretta...")
    roi_height = min(final_shirt_height, height - y_start)
    roi_width = min(final_shirt_width, width - x_start)
    shirt_crop = shirt_resized[:roi_height, :roi_width]
    image_clean[y_start:y_start+roi_height, x_start:x_start+roi_width] = shirt_crop

#--------------------------------------------------------------------------------------------
# POST-PROCESSING AVANZATO
#--------------------------------------------------------------------------------------------

print("\n11. Post-processing avanzato...")

# Smoothing selettivo solo nell'area del nuovo indumento
mask_new_garment = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(mask_new_garment, (x_start, y_start), 
              (x_start + final_shirt_width, y_start + final_shirt_height), 255, -1)

# Blur solo sui bordi per integrazione naturale
kernel_smooth = np.ones((3,3), np.uint8)
border_mask = cv2.morphologyEx(mask_new_garment, cv2.MORPH_GRADIENT, kernel_smooth)

# Applicazione smoothing selettivo
result = image_clean.copy()
smoothed = cv2.bilateralFilter(image_clean, 5, 80, 80)

# Blend solo sui bordi
border_mask_norm = border_mask.astype(float) / 255.0
for c in range(3):
    result[:, :, c] = (border_mask_norm * smoothed[:, :, c] + 
                      (1 - border_mask_norm) * result[:, :, c])

# Leggero miglioramento contrasto globale
result = cv2.convertScaleAbs(result, alpha=1.05, beta=3)

#--------------------------------------------------------------------------------------------
# SALVATAGGIO RISULTATI COMPLETO
#--------------------------------------------------------------------------------------------

print("\n12. Salvataggio risultati...")

# Risultato principale
cv2.imwrite(OUTPUT_IMAGE, result)
print(f"✓ Risultato principale: {OUTPUT_IMAGE}")

# Comparazione tripla: originale | pulito | finale
comparison_width = width // 3
comparison_height = height
comparison = np.hstack([
    cv2.resize(image, (comparison_width, comparison_height)),
    cv2.resize(image_clean, (comparison_width, comparison_height)),
    cv2.resize(result, (comparison_width, comparison_height))
])
comparison_file = OUTPUT_IMAGE.replace('.png', '_comparison_triple.png')
cv2.imwrite(comparison_file, comparison)
print(f"✓ Comparazione tripla: {comparison_file}")

# Grid debug completa
debug_grid = np.vstack([
    # Riga 1: pose + maschera finale
    np.hstack([
        cv2.resize(debug_image, (width//2, height//2)),
        cv2.resize(cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), (width//2, height//2))
    ]),
    # Riga 2: dopo inpaint + risultato finale
    np.hstack([
        cv2.resize(image_clean, (width//2, height//2)),
        cv2.resize(result, (width//2, height//2))
    ])
])
debug_file = OUTPUT_IMAGE.replace('.png', '_debug_improved.png')
cv2.imwrite(debug_file, debug_grid)
print(f"✓ Debug grid: {debug_file}")

#--------------------------------------------------------------------------------------------
# STATISTICHE FINALI DETTAGLIATE
#--------------------------------------------------------------------------------------------

print("\n" + "="*60)
print("🎉 VIRTUAL TRY-ON MIGLIORATO COMPLETATO!")
print("="*60)
print(f"📁 Risultato: {OUTPUT_IMAGE}")
print(f"📁 Comparazione: {comparison_file}")
print(f"📁 Debug: {debug_file}")
print()
print("📊 STATISTICHE:")
print(f"   • Dimensioni immagine: {width}x{height}")
print(f"   • Pixel indumento rimossi: {final_pixels}")
print(f"   • Dimensioni nuovo indumento: {final_shirt_width}x{final_shirt_height}")
print(f"   • Fattore ingrandimento: {scale_improvement:.2f}x")
print(f"   • Posizione: ({x_start}, {y_start})")
print()
print("🔧 MIGLIORAMENTI APPLICATI:")
print("   ✓ Segmentazione multipla (SegFormer + colore + anatomica)")
print("   ✓ Rimozione in 3 passaggi")
print("   ✓ Ridimensionamento antropometrico")
print("   ✓ Posizionamento ottimizzato")
print("   ✓ Blending con gamma correction")
print("="*60)

# Visualizzazione opzionale
try:
    print("Visualizzazione risultato...")
    cv2.imshow('Virtual Try-On MIGLIORATO', result)
    print("Premi un tasto per chiudere...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("(Visualizzazione non disponibile)")

print("🚀 Processo migliorato completato!")