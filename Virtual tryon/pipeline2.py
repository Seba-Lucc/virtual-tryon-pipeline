import cv2
import mediapipe as mp
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch

# Prima installiamo le dipendenze necessarie
# pip install transformers torch torchvision pillow

#--------------------------------------------------------------------------------------------
# CONFIGURAZIONE E CARICAMENTO MODELLI
#--------------------------------------------------------------------------------------------

# Caricamento dell'immagine di test
image_path = 'tst.jpg'
image = cv2.imread(image_path)
if image is None:
    raise IOError("Impossibile aprire l'immagine di test. Controlla il path.")

# Ridimensionamento per ottimizzare performance
max_width = 640
if image.shape[1] > max_width:
    scale = max_width / image.shape[1]
    image = cv2.resize(image, (max_width, int(image.shape[0] * scale)))
print("Dimensioni immagine:", image.shape)

# Inizializzazione del rilevatore di pose MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Inizializzazione SegFormer per segmentazione indumenti
print("Caricamento SegFormer B2 Clothes...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Mappa delle classi per SegFormer B2 Clothes
SEGFORMER_LABELS = {
    0: 'Background', 1: 'Hat', 2: 'Hair', 3: 'Sunglasses', 4: 'Upper-clothes', 
    5: 'Skirt', 6: 'Pants', 7: 'Dress', 8: 'Belt', 9: 'Left-shoe', 
    10: 'Right-shoe', 11: 'Face', 12: 'Left-leg', 13: 'Right-leg', 
    14: 'Left-arm', 15: 'Right-arm', 16: 'Bag', 17: 'Scarf'
}

# Classi che vogliamo rimuovere (indumenti torso)
CLOTHING_CLASSES_TO_REMOVE = [4, 7]  # Upper-clothes, Dress
print("Modelli caricati con successo!")

#--------------------------------------------------------------------------------------------
# RILEVAMENTO POSE
#--------------------------------------------------------------------------------------------

# Convertire l'immagine per MediaPipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

if not results.pose_landmarks:
    print("Pose non rilevata. Uscita dal programma.")
    exit()

print("Pose rilevata con successo!")

# Debug: salva immagine con pose landmarks
debug_image = image.copy()
mp.solutions.drawing_utils.draw_landmarks(
    debug_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
)
cv2.imwrite('pose_landmarks_debug.png', debug_image)

#--------------------------------------------------------------------------------------------
# ESTRAZIONE COORDINATE TORSO MIGLIORATA
#--------------------------------------------------------------------------------------------

height, width, _ = image.shape

# Estrazione landmarks chiave
landmarks = results.pose_landmarks.landmark
left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

# Aggiungiamo più punti per una definizione migliore del torso
left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

# Conversione in coordinate pixel
def landmark_to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

x_left, y_left = landmark_to_pixel(left_shoulder, width, height)
x_right, y_right = landmark_to_pixel(right_shoulder, width, height)
x_left_hip, y_left_hip = landmark_to_pixel(left_hip, width, height)
x_right_hip, y_right_hip = landmark_to_pixel(right_hip, width, height)
x_left_elbow, y_left_elbow = landmark_to_pixel(left_elbow, width, height)
x_right_elbow, y_right_elbow = landmark_to_pixel(right_elbow, width, height)

print(f"Spalle: ({x_left}, {y_left}) - ({x_right}, {y_right})")
print(f"Anche: ({x_left_hip}, {y_left_hip}) - ({x_right_hip}, {y_right_hip})")

# Calcolo dimensioni corporee
shoulder_width = abs(x_right - x_left)
torso_height = max(y_left_hip, y_right_hip) - min(y_left, y_right)
print(f"Larghezza spalle: {shoulder_width}px, Altezza torso: {torso_height}px")

#--------------------------------------------------------------------------------------------
# SEGMENTAZIONE AVANZATA CON SEGFORMER
#--------------------------------------------------------------------------------------------

print("Esecuzione segmentazione SegFormer...")

# Preparazione immagine per SegFormer
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
inputs = processor(images=pil_image, return_tensors="pt")

# Inferenza
with torch.no_grad():
    outputs = model(**inputs)
    
# Post-processing per ottenere le classi predette
logits = outputs.logits.cpu()
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=pil_image.size[::-1],  # (height, width)
    mode="bilinear",
    align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

print(f"Segmentazione completata. Shape: {pred_seg.shape}")
print(f"Classi trovate: {np.unique(pred_seg)}")

# Creazione maschera per gli indumenti da rimuovere
clothing_mask = np.zeros_like(pred_seg, dtype=np.uint8)
for class_id in CLOTHING_CLASSES_TO_REMOVE:
    clothing_mask[pred_seg == class_id] = 255

print(f"Pixel di indumenti trovati: {np.sum(clothing_mask > 0)}")

# Salva maschera per debug
cv2.imwrite('segformer_clothing_mask_debug.png', clothing_mask)

# Creazione maschera colorata per visualizzazione
colored_mask = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
          (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
          (128, 0, 128), (0, 128, 128), (64, 64, 64), (192, 192, 192), 
          (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128)]

for i, label in SEGFORMER_LABELS.items():
    if i < len(colors):
        colored_mask[pred_seg == i] = colors[i]

cv2.imwrite('segformer_full_segmentation_debug.png', colored_mask)

#--------------------------------------------------------------------------------------------
# DEFINIZIONE AREA TORSO COMBINATA
#--------------------------------------------------------------------------------------------

# Creiamo una maschera dell'area torso basata sui landmarks
torso_region_mask = np.zeros((height, width), dtype=np.uint8)

# Poligono del torso esteso con più punti per maggiore precisione
margin_x = int(shoulder_width * 0.25)  # Margine più generoso
margin_y_top = int(torso_height * 0.15)
margin_y_bottom = int(torso_height * 0.25)

# Definiamo un poligono più preciso che segue meglio la forma del corpo
torso_points = np.array([
    [x_left - margin_x, y_left - margin_y_top],          # spalla sinistra estesa
    [x_right + margin_x, y_right - margin_y_top],        # spalla destra estesa
    [x_right_elbow + margin_x//2, y_right_elbow],        # gomito destro
    [x_right_hip + margin_x//2, y_right_hip + margin_y_bottom],  # anca destra
    [x_left_hip - margin_x//2, y_left_hip + margin_y_bottom],   # anca sinistra
    [x_left_elbow - margin_x//2, y_left_elbow],         # gomito sinistro
], dtype=np.int32)

# Assicuriamoci che i punti siano dentro i limiti dell'immagine
torso_points[:, 0] = np.clip(torso_points[:, 0], 0, width - 1)
torso_points[:, 1] = np.clip(torso_points[:, 1], 0, height - 1)

cv2.fillPoly(torso_region_mask, [torso_points], 255)

# Combiniamo la segmentazione SegFormer con l'area del torso
final_clothing_mask = cv2.bitwise_and(clothing_mask, torso_region_mask)

# Miglioramento morfologico della maschera finale
kernel = np.ones((7,7), np.uint8)
final_clothing_mask = cv2.morphologyEx(final_clothing_mask, cv2.MORPH_CLOSE, kernel)
final_clothing_mask = cv2.morphologyEx(final_clothing_mask, cv2.MORPH_OPEN, kernel)

# Dilatazione leggera per catturare i bordi
kernel_dilate = np.ones((3,3), np.uint8)
final_clothing_mask = cv2.dilate(final_clothing_mask, kernel_dilate, iterations=2)

# Blur per bordi più morbidi
final_clothing_mask = cv2.GaussianBlur(final_clothing_mask, (7, 7), 0)

# Salva maschera finale per debug
cv2.imwrite('final_clothing_mask_debug.png', final_clothing_mask)

print(f"Pixel finali da rimuovere: {np.sum(final_clothing_mask > 0)}")

#--------------------------------------------------------------------------------------------
# RIMOZIONE INDUMENTO CON INPAINTING MIGLIORATO
#--------------------------------------------------------------------------------------------

# Inpainting più aggressivo per rimozione completa
image_clean = cv2.inpaint(image, final_clothing_mask, 10, cv2.INPAINT_TELEA)

# Secondo passaggio di inpainting per risultati ancora migliori
image_clean = cv2.inpaint(image_clean, final_clothing_mask, 7, cv2.INPAINT_NS)

cv2.imwrite('image_after_inpaint_debug.png', image_clean)

#--------------------------------------------------------------------------------------------
# CARICAMENTO E RIDIMENSIONAMENTO INDUMENTO OTTIMIZZATO
#--------------------------------------------------------------------------------------------

shirt = cv2.imread('Soggetto.png', cv2.IMREAD_UNCHANGED)
if shirt is None:
    raise IOError("Immagine della maglietta non trovata. Controlla il percorso.")

print(f"Dimensioni originali indumento: {shirt.shape}")

# ALGORITMO DI RIDIMENSIONAMENTO AVANZATO
# Calcoliamo le dimensioni target basate sulle proporzioni corporee

# Fattori di scala antropometrici più accurati
width_scale_factor = 1.6  # Copertura più ampia per realismo
height_scale_factor = 0.9  # Copertura più alta del torso

# Dimensioni target
target_width = int(shoulder_width * width_scale_factor)
target_height = int(torso_height * height_scale_factor)

# Manteniamo le proporzioni dell'indumento
original_ratio = shirt.shape[1] / shirt.shape[0]  # width/height
target_ratio = target_width / target_height

if original_ratio > target_ratio:
    # L'indumento è più largo del target, limitiamo dalla larghezza
    final_width = target_width
    final_height = int(target_width / original_ratio)
else:
    # L'indumento è più alto del target, limitiamo dall'altezza
    final_height = target_height
    final_width = int(target_height * original_ratio)

# Assicuriamoci che non sia troppo grande
final_width = min(final_width, int(width * 0.8))
final_height = min(final_height, int(height * 0.8))

shirt_resized = cv2.resize(shirt, (final_width, final_height))
print(f"Dimensioni indumento ridimensionato: {shirt_resized.shape}")

#--------------------------------------------------------------------------------------------
# POSIZIONAMENTO E APPLICAZIONE INTELLIGENTE
#--------------------------------------------------------------------------------------------

# Posizionamento ottimizzato
center_x = (x_left + x_right) // 2
x_start = center_x - final_width // 2

# Posizionamento verticale: leggermente sotto le spalle
vertical_offset = int(torso_height * 0.08)  # 8% sotto le spalle
y_start = min(y_left, y_right) + vertical_offset

# Controlli di sicurezza
x_start = max(0, min(x_start, width - final_width))
y_start = max(0, min(y_start, height - final_height))

print(f"Posizione finale indumento: ({x_start}, {y_start})")

# Applicazione con alpha blending migliorato
if shirt_resized.shape[2] == 4:  # Ha canale alpha
    overlay_rgb = shirt_resized[..., :3].astype(float)
    overlay_alpha = shirt_resized[..., 3:].astype(float) / 255.0
    
    # ROI sull'immagine pulita
    roi = image_clean[y_start:y_start+final_height, x_start:x_start+final_width].astype(float)
    
    # Blending con curve gamma per colori più naturali
    gamma = 2.2
    overlay_rgb_gamma = np.power(overlay_rgb / 255.0, gamma)
    roi_gamma = np.power(roi / 255.0, gamma)
    
    # Alpha blending nello spazio gamma
    for c in range(3):
        roi_gamma[:, :, c] = (overlay_alpha[:, :, 0] * overlay_rgb_gamma[:, :, c] + 
                             (1 - overlay_alpha[:, :, 0]) * roi_gamma[:, :, c])
    
    # Conversione back to linear space
    roi_final = np.power(roi_gamma, 1/gamma) * 255.0
    
    image_clean[y_start:y_start+final_height, x_start:x_start+final_width] = roi_final.astype(np.uint8)
else:
    # Senza canale alpha
    image_clean[y_start:y_start+final_height, x_start:x_start+final_width] = shirt_resized

#--------------------------------------------------------------------------------------------
# POST-PROCESSING E FINALIZZAZIONE
#--------------------------------------------------------------------------------------------

# Smoothing finale per integrazione naturale
final_result = cv2.bilateralFilter(image_clean, 5, 80, 80)

# Opzionale: leggero aumento del contrasto per migliorare l'aspetto
alpha = 1.1  # fattore contrasto
beta = 5     # fattore luminosità
final_result = cv2.convertScaleAbs(final_result, alpha=alpha, beta=beta)

#--------------------------------------------------------------------------------------------
# VISUALIZZAZIONE E SALVATAGGIO
#--------------------------------------------------------------------------------------------

# Visualizzazione
cv2.imshow('Virtual Try-On Result', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvataggio risultati
cv2.imwrite('output_tryon_segformer.png', final_result)
print("Risultato salvato come output_tryon_segformer.png")

# Salvataggio comparazione
comparison = np.hstack([
    cv2.resize(image, (width//2, height//2)),
    cv2.resize(final_result, (width//2, height//2))
])
cv2.imwrite('comparison_before_after_segformer.png', comparison)
print("Comparazione salvata come comparison_before_after_segformer.png")

# Salvataggio griglia debug completa
debug_grid = np.vstack([
    np.hstack([
        cv2.resize(debug_image, (width//2, height//2)),
        cv2.resize(cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR), (width//2, height//2))
    ]),
    np.hstack([
        cv2.resize(cv2.cvtColor(final_clothing_mask, cv2.COLOR_GRAY2BGR), (width//2, height//2)),
        cv2.resize(final_result, (width//2, height//2))
    ])
])
cv2.imwrite('debug_grid_complete.png', debug_grid)
print("Griglia debug completa salvata come debug_grid_complete.png")

print("\n=== ANALISI COMPLETATA ===")
print(f"- Pose rilevata: ✓")
print(f"- Segmentazione SegFormer: ✓")
print(f"- Indumento rimosso: ✓")
print(f"- Nuovo indumento applicato: ✓")
print(f"- Dimensioni finali: {final_width}x{final_height}")
print("===========================")