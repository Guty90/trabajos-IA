import cv2
import mediapipe as mp
import math
import numpy as np
import csv
import os

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Función auxiliar para medir distancia entre puntos
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# Función para calcular la apertura del ojo (EAR - Eye Aspect Ratio)
def eye_aspect_ratio(eye_points):
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Función para extraer características faciales
def extract_features(landmarks, w, h):
    """Extrae características numéricas de la cara (similar al dataset Iris)"""
    
    # BOCA
    left_mouth_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
    right_mouth_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
    upper_lip_top = (int(landmarks[13].x * w), int(landmarks[13].y * h))
    lower_lip_bottom = (int(landmarks[14].x * w), int(landmarks[14].y * h))
    upper_lip_inner = (int(landmarks[12].x * w), int(landmarks[12].y * h))
    lower_lip_inner = (int(landmarks[15].x * w), int(landmarks[15].y * h))
    
    # CEJAS
    left_eyebrow_center = (int(landmarks[105].x * w), int(landmarks[105].y * h))
    right_eyebrow_center = (int(landmarks[334].x * w), int(landmarks[334].y * h))
    left_eyebrow_inner = (int(landmarks[70].x * w), int(landmarks[70].y * h))
    right_eyebrow_inner = (int(landmarks[300].x * w), int(landmarks[300].y * h))

    # OJOS
    left_eye_inner = (int(landmarks[133].x * w), int(landmarks[133].y * h))
    left_eye_outer = (int(landmarks[33].x * w), int(landmarks[33].y * h))
    left_eye_top = (int(landmarks[159].x * w), int(landmarks[159].y * h))
    left_eye_bottom = (int(landmarks[145].x * w), int(landmarks[145].y * h))
    
    right_eye_inner = (int(landmarks[362].x * w), int(landmarks[362].y * h))
    right_eye_outer = (int(landmarks[263].x * w), int(landmarks[263].y * h))
    right_eye_top = (int(landmarks[386].x * w), int(landmarks[386].y * h))
    right_eye_bottom = (int(landmarks[374].x * w), int(landmarks[374].y * h))

    # Puntos para EAR
    left_eye_points = [
        left_eye_inner, 
        (int(landmarks[160].x * w), int(landmarks[160].y * h)),
        left_eye_top, left_eye_outer,
        (int(landmarks[144].x * w), int(landmarks[144].y * h)),
        left_eye_bottom
    ]
    
    right_eye_points = [
        right_eye_inner,
        (int(landmarks[387].x * w), int(landmarks[387].y * h)),
        right_eye_top, right_eye_outer,
        (int(landmarks[373].x * w), int(landmarks[373].y * h)),
        right_eye_bottom
    ]

    # NARIZ
    nose_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))

    # ===== CARACTERÍSTICAS =====
    
    # Boca
    mouth_width = distance(left_mouth_corner, right_mouth_corner)
    mouth_height = distance(upper_lip_top, lower_lip_bottom)
    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
    
    mouth_center_y = (upper_lip_top[1] + lower_lip_bottom[1]) / 2
    mouth_corners_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2
    mouth_curve = mouth_center_y - mouth_corners_y
    
    mouth_inner_height = distance(upper_lip_inner, lower_lip_inner)
    inner_mouth_ratio = mouth_inner_height / mouth_width if mouth_width > 0 else 0
    
    # Cejas y ojos
    left_brow_eye_dist = distance(left_eyebrow_center, left_eye_top)
    right_brow_eye_dist = distance(right_eyebrow_center, right_eye_top)
    avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
    
    face_width = distance(left_eye_outer, right_eye_outer)
    brow_ratio = avg_brow_eye_dist / face_width if face_width > 0 else 0
    
    eyebrow_distance = distance(left_eyebrow_inner, right_eyebrow_inner)
    eyebrow_ratio = eyebrow_distance / face_width if face_width > 0 else 0
    
    # Ojos
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2
    
    # Nariz-boca
    nose_mouth_dist = distance(nose_tip, upper_lip_top)
    nose_mouth_ratio = nose_mouth_dist / face_width if face_width > 0 else 0
    
    # Retornar como lista (como iris.data)
    return [
        mouth_ratio,         # 0
        mouth_curve,         # 1
        inner_mouth_ratio,   # 2
        brow_ratio,          # 3
        eyebrow_ratio,       # 4
        avg_ear,             # 5
        nose_mouth_ratio,    # 6
        left_ear,            # 7
        right_ear            # 8
    ]

# Configuración
DATASET_FILE = 'emotion_dataset.csv'
EMOTIONS = ['Feliz', 'Triste', 'Enojado', 'Sorprendido', 'Neutral']
EMOTION_KEYS = ['1', '2', '3', '4', '5']

# Colores para visualización
COLORS = [(0, 255, 0), (255, 100, 0), (0, 0, 255), (0, 255, 255), (200, 200, 200)]

# Crear archivo CSV con encabezados
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [
            'mouth_ratio', 'mouth_curve', 'inner_mouth_ratio',
            'brow_ratio', 'eyebrow_ratio', 'avg_ear',
            'nose_mouth_ratio', 'left_ear', 'right_ear', 'emotion'
        ]
        writer.writerow(headers)

# Captura de video
cap = cv2.VideoCapture(0)

print("=" * 70)
print(" " * 15 + "RECOLECTOR DE DATASET - EMOCIONES")
print("=" * 70)
print("\nInstrucciones:")
print("  Presiona 1-5 para etiquetar tu expresión facial actual:")
for i, emotion in enumerate(EMOTIONS):
    print(f"    [{i+1}] {emotion}")
print("\n  [Q] Salir y guardar dataset")
print("\nRecomendaciones:")
print("  • Captura al menos 30-50 muestras por emoción")
print("  • Varía tu expresión ligeramente en cada captura")
print("  • Mantén buena iluminación y encuadre frontal")
print("=" * 70 + "\n")

samples_count = {emotion: 0 for emotion in EMOTIONS}
total_samples = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Extraer características
            features = extract_features(landmarks, w, h)
            
            # Panel de información
            cv2.rectangle(frame, (10, 10), (400, 260), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 260), (255, 255, 255), 2)
            
            cv2.putText(frame, "Presiona para etiquetar:", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_pos = 70
            for i, emotion in enumerate(EMOTIONS):
                count = samples_count[emotion]
                color = COLORS[i] if count > 0 else (100, 100, 100)
                text = f"[{i+1}] {emotion}: {count}"
                cv2.putText(frame, text, (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 30
            
            # Total
            cv2.line(frame, (20, y_pos-10), (390, y_pos-10), (255, 255, 255), 1)
            cv2.putText(frame, f"TOTAL: {total_samples}", (30, y_pos+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Métricas actuales
            cv2.rectangle(frame, (420, 10), (w-10, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (420, 10), (w-10, 150), (255, 255, 255), 2)
            
            cv2.putText(frame, "Metricas:", (430, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            metrics = [
                f"Boca: {features[0]:.2f}",
                f"Curva: {features[1]:.1f}",
                f"Cejas: {features[3]:.2f}",
                f"Ojos: {features[5]:.2f}"
            ]
            
            y_pos = 70
            for metric in metrics:
                cv2.putText(frame, metric, (430, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 25

    cv2.imshow("Recolector de Dataset - Presiona Q para salir", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Guardar muestra
    if chr(key) in EMOTION_KEYS and results.multi_face_landmarks:
        emotion_idx = EMOTION_KEYS.index(chr(key))
        emotion_label = EMOTIONS[emotion_idx]
        
        # Guardar en CSV
        with open(DATASET_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            row = features + [emotion_label]
            writer.writerow(row)
        
        samples_count[emotion_label] += 1
        total_samples += 1
        print(f"✓ [{total_samples}] {emotion_label} guardado (Total {emotion_label}: {samples_count[emotion_label]})")
    
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()

# Resumen final
print("\n" + "=" * 70)
print(" " * 25 + "RESUMEN FINAL")
print("=" * 70)
print(f"\nTotal de muestras recolectadas: {total_samples}")
print("\nDistribución por emoción:")
for emotion, count in samples_count.items():
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    bar = "█" * int(percentage / 5)
    print(f"  {emotion:15} {count:3} muestras  {bar} {percentage:.1f}%")

print(f"\n✓ Dataset guardado en: {DATASET_FILE}")
print("\nAhora puedes entrenar tu árbol de decisión con este archivo.")
print("=" * 70)