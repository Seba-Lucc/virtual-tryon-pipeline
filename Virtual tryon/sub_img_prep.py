import cv2
import mediapipe as mp
import numpy as np

# Caricamento dell'immagine di test
image_path = 'tst.jpg'  # Sostituire con il percorso dell’immagine
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
    mp.solutions.drawing_utils.draw_landmarks(
        image,  # disegniamo direttamente sull'immagine originale 
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
    )
    cv2.imwrite('pose_landmarks.png', image)  # salva un'immagine con la posa disegnata
else:
    print("Pose non rilevata.")

#--------------------------------------------------------------------------------------------

height, width, _ = image.shape
# Estraiamo il landmark della spalla sinistra (indice 11) e destra (12)
left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    # Convertiamo le coordinate normalizzate in pixel dell'immagine
x_left = int(left_shoulder.x * width)
y_left = int(left_shoulder.y * height)
x_right = int(right_shoulder.x * width)
y_right = int(right_shoulder.y * height)
print("Coordinate spalla sinistra:", x_left, y_left)
print("Coordinate spalla destra:", x_right, y_right)

#---------

# --- inizio: calcolo coordinate anche in pixel ---
# landmarks per le anche (LEFT_HIP = 23, RIGHT_HIP = 24)
left_hip  = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

# converto da normalizzato a pixel
x_left_hip  = int(left_hip.x  * width)
y_left_hip  = int(left_hip.y  * height)
x_right_hip = int(right_hip.x * width)
y_right_hip = int(right_hip.y * height)

print(f"[DEBUG] anche: ({x_left_hip},{y_left_hip})  ({x_right_hip},{y_right_hip})")
# --- fine: calcolo anche ---

# --- inizio: definisco il poligono del torso ---
pts = np.array([
    [x_left,      y_left],      # spalla sinistra
    [x_right,     y_right],     # spalla destra
    [x_right_hip, y_right_hip], # anca destra
    [x_left_hip,  y_left_hip]   # anca sinistra
], dtype=np.int32)
# --- fine poligono torso ---

# 1) Dopo il Pose.process, hai già image_rgb, results, width, height...
# 2) Estraggo spalle:
x_left,  y_left  = int(left_shoulder.x * width), int(left_shoulder.y * height)
x_right, y_right = int(right_shoulder.x * width), int(right_shoulder.y * height)

# 3) **Qui** inserisci il calcolo delle anche e di pts (come sopra)

# 4) Ora procedi con SelfieSegmentation
mp_selfie = mp.solutions.selfie_segmentation
seg = mp_selfie.SelfieSegmentation(model_selection=1)
seg_res = seg.process(image_rgb)

person_mask = (seg_res.segmentation_mask > 0.5).astype(np.uint8) * 255

# 5) Calcolo la mask del solo torso
torso_mask = np.zeros_like(person_mask)
cv2.fillPoly(torso_mask, [pts], 255)

final_mask = cv2.bitwise_and(person_mask, torso_mask)

# 6) Inpaint sulla mask finale
image = cv2.inpaint(image, final_mask, 3, cv2.INPAINT_TELEA)

# 7) A questo punto l'area del torso è pulita: prosegui con il caricamento e blending del PNG

#--------------------------------------------------------------------------------------------

# Carica l'immagine PNG dell'indumento con il canale alfa
shirt = cv2.imread('Soggetto.png', cv2.IMREAD_UNCHANGED)  # risultato: array HxWx4 (BGRA)
if shirt is None:
    raise IOError("Immagine della maglietta non trovata. Controlla il percorso.")

# Determina la larghezza tra le spalle in pixel usando i keypoint
shoulder_width = int(abs(x_right - x_left))
print("Larghezza spalle (px):", shoulder_width)

# Ridimensiona la maglietta alla larghezza delle spalle (con un piccolo margine)
new_width = int(shoulder_width * 1.1)  # es: 10% più larga delle spalle
scale = new_width / shirt.shape[1]
new_height = int(shirt.shape[0] * scale)
shirt_resized = cv2.resize(shirt, (new_width, new_height))
print("Dimensioni maglietta ridimensionata:", shirt_resized.shape)

# Determina dove posizionare la maglietta sull'immagine originale
# Allineiamo orizzontalmente al centro tra le due spalle
center_x = (x_left + x_right) // 2
x_start = center_x - new_width // 2
# Allineiamo verticalmente all'altezza delle spalle (o leggermente sopra)
y_start = min(y_left, y_right) - 10  # piccolo offset verso l'alto (es. 10 px)
y_start = max(0, y_start)  # assicuriamoci di non uscire sopra l'immagine
print("Posizione di overlay (px):", x_start, y_start)

# Sovrapposizione usando il canale alfa
# Separiamo i canali BGR e Alfa dell'immagine dell'indumento
overlay_rgb = shirt_resized[..., :3]    # canali BGR
overlay_alpha = shirt_resized[..., 3:]  # canale alpha
overlay_alpha = overlay_alpha.astype(float) / 255.0  # normalizziamo alpha 0.0–1.0

# Definiamo l'area di ROI (Region of Interest) sull'immagine originale dove mettere la maglietta
h, w, _ = overlay_rgb.shape
roi = image[y_start:y_start+h, x_start:x_start+w]

# Effettuiamo il blending pixel-per-pixel
roi = roi.astype(float)
overlay_rgb = overlay_rgb.astype(float)
for c in range(3):  # per ogni canale B, G, R
    roi[:, :, c] = overlay_alpha[:, :, 0] * overlay_rgb[:, :, c] + (1 - overlay_alpha[:, :, 0]) * roi[:, :, c]

# Sostituiamo l'area dell'immagine originale con il risultato
image[y_start:y_start+h, x_start:x_start+w] = roi.astype(np.uint8)

#--------------------------------------------------------------------------------------------

# Visualizza l'immagine risultante in una finestra (se eseguito in locale)
cv2.imshow('Virtual Try-On', image)
cv2.waitKey(0)  # attende la pressione di un tasto
cv2.destroyAllWindows()

# Salva l'immagine risultante su file
cv2.imwrite('output_tryon.png', image)
print("Risultato salvato come output_tryon.png")