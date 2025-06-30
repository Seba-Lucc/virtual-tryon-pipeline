import cv2
import mediapipe as mp
import numpy as np

# Caricamento dell'immagine di test
image_path = 'tst.jpg'  # Sostituire con il percorso dell'immagine
image = cv2.imread(image_path)
if image is None:
    raise IOError("Impossibile aprire l'immagine di test. Controlla il path.")

# Opzionale: ridimensionamento se l'immagine è molto grande
max_width = 640
if image.shape[1] > max_width:
    scale = max_width / image.shape[1]
    image = cv2.resize(image, (max_width, int(image.shape[0] * scale)))
print("Dimensioni immagine:", image.shape)

#--------------------------------------------------------------------------------------------
# RILEVAMENTO POSE
#--------------------------------------------------------------------------------------------

# Inizializzazione del rilevatore di pose di MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Convertiamo l'immagine da BGR a RGB prima di passarla a MediaPipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# Verifichiamo se sono stati rilevati i pose landmarks
if results.pose_landmarks:
    print("Pose rilevata con successo!")
    # Disegniamo i punti e le connessioni sul corpo per verifica (opzionale)
    debug_image = image.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        debug_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
    )
    cv2.imwrite('pose_landmarks_debug.png', debug_image)
else:
    print("Pose non rilevata.")
    exit()

#--------------------------------------------------------------------------------------------
# ESTRAZIONE COORDINATE TORSO
#--------------------------------------------------------------------------------------------

height, width, _ = image.shape

# Estraiamo i landmarks principali del torso
left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

# Convertiamo le coordinate normalizzate in pixel
x_left = int(left_shoulder.x * width)
y_left = int(left_shoulder.y * height)
x_right = int(right_shoulder.x * width)
y_right = int(right_shoulder.y * height)
x_left_hip = int(left_hip.x * width)
y_left_hip = int(left_hip.y * height)
x_right_hip = int(right_hip.x * width)
y_right_hip = int(right_hip.y * height)

print("Coordinate spalla sinistra:", x_left, y_left)
print("Coordinate spalla destra:", x_right, y_right)
print("Coordinate anca sinistra:", x_left_hip, y_left_hip)
print("Coordinate anca destra:", x_right_hip, y_right_hip)

# Calcolo dimensioni del torso
shoulder_width = abs(x_right - x_left)
torso_height = max(y_left_hip, y_right_hip) - min(y_left, y_right)
print(f"Larghezza spalle: {shoulder_width}px, Altezza torso: {torso_height}px")

#--------------------------------------------------------------------------------------------
# RIMOZIONE INDUMENTO ORIGINALE (MIGLIORATA)
#--------------------------------------------------------------------------------------------

# Segmentazione persona con MediaPipe
mp_selfie = mp.solutions.selfie_segmentation
seg = mp_selfie.SelfieSegmentation(model_selection=1)
seg_res = seg.process(image_rgb)

# Maschera della persona (più precisa con soglia adattiva)
person_mask = (seg_res.segmentation_mask > 0.3).astype(np.uint8) * 255

# Definizione poligono torso esteso per catturare meglio l'indumento
# Estendiamo il poligono per essere sicuri di catturare tutto l'indumento
margin_x = int(shoulder_width * 0.15)  # margine laterale 15%
margin_y_top = int(torso_height * 0.1)  # margine superiore 10%
margin_y_bottom = int(torso_height * 0.2)  # margine inferiore 20%

pts = np.array([
    [x_left - margin_x, y_left - margin_y_top],           # spalla sinistra estesa
    [x_right + margin_x, y_right - margin_y_top],         # spalla destra estesa
    [x_right_hip + margin_x, y_right_hip + margin_y_bottom], # anca destra estesa
    [x_left_hip - margin_x, y_left_hip + margin_y_bottom]    # anca sinistra estesa
], dtype=np.int32)

# Assicuriamoci che i punti siano dentro i limiti dell'immagine
pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

# Maschera del torso estesa
torso_mask = np.zeros_like(person_mask)
cv2.fillPoly(torso_mask, [pts], 255)

# Maschera finale: intersezione persona + torso
final_mask = cv2.bitwise_and(person_mask, torso_mask)

# Miglioramento della maschera con operazioni morfologiche
kernel = np.ones((5,5), np.uint8)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

# Applicazione di un leggero blur per bordi più morbidi
final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

# Salva la maschera per debug
cv2.imwrite('torso_mask_debug.png', final_mask)

# Inpainting migliorato per rimuovere l'indumento originale
image_clean = cv2.inpaint(image, final_mask, 7, cv2.INPAINT_TELEA)
cv2.imwrite('image_after_inpaint_debug.png', image_clean)

#--------------------------------------------------------------------------------------------
# CARICAMENTO E RIDIMENSIONAMENTO INDUMENTO (MIGLIORATO)
#--------------------------------------------------------------------------------------------

# Carica l'immagine PNG dell'indumento con il canale alfa
shirt = cv2.imread('Soggetto.png', cv2.IMREAD_UNCHANGED)
if shirt is None:
    raise IOError("Immagine della maglietta non trovata. Controlla il percorso.")

print(f"Dimensioni originali indumento: {shirt.shape}")

# RIDIMENSIONAMENTO MIGLIORATO
# Calcoliamo un fattore di scala più appropriato considerando sia larghezza che altezza

# Fattore per la larghezza: l'indumento dovrebbe coprire circa 80-90% della larghezza del torso
width_scale_factor = 1.4  # più generoso del precedente 1.1
target_width = int(shoulder_width * width_scale_factor)

# Fattore per l'altezza: l'indumento dovrebbe coprire circa 70-80% dell'altezza del torso
height_scale_factor = 0.8
target_height = int(torso_height * height_scale_factor)

# Calcoliamo il rapporto di scala mantenendo le proporzioni
scale_width = target_width / shirt.shape[1]
scale_height = target_height / shirt.shape[0]

# Usiamo il fattore di scala maggiore per garantire una copertura adeguata
scale = max(scale_width, scale_height)

# Ridimensionamento finale
new_width = int(shirt.shape[1] * scale)
new_height = int(shirt.shape[0] * scale)
shirt_resized = cv2.resize(shirt, (new_width, new_height))

print(f"Dimensioni indumento ridimensionato: {shirt_resized.shape}")
print(f"Fattore di scala applicato: {scale:.2f}")

#--------------------------------------------------------------------------------------------
# POSIZIONAMENTO E APPLICAZIONE INDUMENTO
#--------------------------------------------------------------------------------------------

# Posizionamento migliorato
center_x = (x_left + x_right) // 2
x_start = center_x - new_width // 2

# Posizionamento verticale: leggermente sotto le spalle per un look più naturale
y_start = min(y_left, y_right) + int(torso_height * 0.05)  # 5% sotto le spalle

# Assicuriamoci che l'indumento non esca dai bordi dell'immagine
x_start = max(0, min(x_start, width - new_width))
y_start = max(0, min(y_start, height - new_height))

print(f"Posizione finale indumento: ({x_start}, {y_start})")

# Verifichiamo che ci sia abbastanza spazio per l'indumento
if x_start + new_width > width or y_start + new_height > height:
    print("ATTENZIONE: L'indumento potrebbe essere troppo grande per l'immagine")
    # Ridimensioniamo ulteriormente se necessario
    available_width = width - x_start
    available_height = height - y_start
    
    if new_width > available_width or new_height > available_height:
        scale_correction = min(available_width / new_width, available_height / new_height)
        new_width = int(new_width * scale_correction)
        new_height = int(new_height * scale_correction)
        shirt_resized = cv2.resize(shirt_resized, (new_width, new_height))
        print(f"Indumento ridimensionato per adattarsi: {new_width}x{new_height}")

# Sovrapposizione usando il canale alfa con blending migliorato
if shirt_resized.shape[2] == 4:  # se ha il canale alpha
    overlay_rgb = shirt_resized[..., :3].astype(float)
    overlay_alpha = shirt_resized[..., 3:].astype(float) / 255.0
    
    # ROI sull'immagine pulita
    roi = image_clean[y_start:y_start+new_height, x_start:x_start+new_width].astype(float)
    
    # Blending migliorato con gamma correction per colori più naturali
    for c in range(3):
        roi[:, :, c] = (overlay_alpha[:, :, 0] * overlay_rgb[:, :, c] + 
                       (1 - overlay_alpha[:, :, 0]) * roi[:, :, c])
    
    # Applicazione del risultato
    image_clean[y_start:y_start+new_height, x_start:x_start+new_width] = roi.astype(np.uint8)
else:
    # Se non c'è canale alpha, sovrapponi direttamente
    image_clean[y_start:y_start+new_height, x_start:x_start+new_width] = shirt_resized

#--------------------------------------------------------------------------------------------
# FINALIZZAZIONE E SALVATAGGIO
#--------------------------------------------------------------------------------------------

# Applicazione di un leggero smoothing per rendere l'integrazione più naturale
final_result = cv2.GaussianBlur(image_clean, (3, 3), 0)

# Visualizzazione e salvataggio
cv2.imshow('Virtual Try-On Result', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvataggio risultato finale
cv2.imwrite('output_tryon_improved.png', final_result)
print("Risultato migliorato salvato come output_tryon_improved.png")

# Salvataggio immagini di debug per analisi
cv2.imwrite('comparison_before_after.png', np.hstack([image, final_result]))
print("Comparazione prima/dopo salvata come comparison_before_after.png")