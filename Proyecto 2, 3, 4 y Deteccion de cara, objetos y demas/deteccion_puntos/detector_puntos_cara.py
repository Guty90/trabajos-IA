import cv2
import mediapipe as mp
import math
import numpy as np
import pickle
import os

# Verificar que existe el modelo entrenado
if not os.path.exists('emotion_tree.pkl'):
    print("=" * 60)
    print("ERROR: No se encontró el modelo 'emotion_tree.pkl'")
    print("=" * 60)
    print("\nDebes ejecutar primero el archivo de ENTRENAMIENTO para:")
    print("1. Capturar muestras de tus expresiones faciales")
    print("2. Entrenar el árbol de decisión")
    print("3. Generar el archivo 'emotion_tree.pkl'")
    print("\nUna vez hecho esto, podrás ejecutar este detector.")
    print("=" * 60)
    exit()

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Diccionario de colores para cada emoción (BGR)
EMOTION_COLORS = {
    "Feliz": (0, 255, 0),        # Verde
    "Triste": (255, 100, 0),     # Azul
    "Enojado": (0, 0, 255),      # Rojo
    "Sorprendido": (0, 255, 255), # Amarillo
    "Neutral": (200, 200, 200)   # Gris claro
}

# Mapeo de números a emociones
LABEL_EMOTIONS = {
    0: "Neutral",
    1: "Feliz",
    2: "Triste",
    3: "Enojado",
    4: "Sorprendido"
}

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(eye_points):
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def extract_features(landmarks, w, h):
    """Extrae las características faciales necesarias para el clasificador"""
    
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
        left_eye_top,
        left_eye_outer,
        (int(landmarks[144].x * w), int(landmarks[144].y * h)),
        left_eye_bottom
    ]
    
    right_eye_points = [
        right_eye_inner,
        (int(landmarks[387].x * w), int(landmarks[387].y * h)),
        right_eye_top,
        right_eye_outer,
        (int(landmarks[373].x * w), int(landmarks[373].y * h)),
        right_eye_bottom
    ]
    
    # NARIZ
    nose_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
    
    # CÁLCULOS DE MÉTRICAS
    mouth_width = distance(left_mouth_corner, right_mouth_corner)
    mouth_height = distance(upper_lip_top, lower_lip_bottom)
    mouth_inner_height = distance(upper_lip_inner, lower_lip_inner)
    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
    
    mouth_center_y = (upper_lip_top[1] + lower_lip_bottom[1]) / 2
    mouth_corners_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2
    mouth_curve = mouth_center_y - mouth_corners_y
    
    left_brow_eye_dist = distance(left_eyebrow_center, left_eye_top)
    right_brow_eye_dist = distance(right_eyebrow_center, right_eye_top)
    avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
    
    face_width = distance(left_eye_outer, right_eye_outer)
    brow_ratio = avg_brow_eye_dist / face_width if face_width > 0 else 0
    
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2
    
    nose_mouth_dist = distance(nose_tip, upper_lip_top)
    nose_mouth_ratio = nose_mouth_dist / face_width if face_width > 0 else 0
    
    # Retornar características como array
    features = [
        mouth_ratio,
        mouth_curve,
        brow_ratio,
        avg_ear,
        nose_mouth_ratio,
        mouth_inner_height / face_width if face_width > 0 else 0
    ]
    
    return np.array(features), {
        'left_mouth_corner': left_mouth_corner,
        'right_mouth_corner': right_mouth_corner,
        'upper_lip_top': upper_lip_top,
        'lower_lip_bottom': lower_lip_bottom,
        'left_eyebrow_center': left_eyebrow_center,
        'right_eyebrow_center': right_eyebrow_center,
        'left_eye_top': left_eye_top,
        'right_eye_top': right_eye_top,
        'nose_tip': nose_tip
    }

# ===== CARGAR MODELO ENTRENADO =====
print("=" * 60)
print("DETECTOR DE EMOCIONES EN TIEMPO REAL")
print("=" * 60)

try:
    with open('emotion_tree.pkl', 'rb') as f:
        clf = pickle.load(f)
    print("✓ Modelo cargado exitosamente!")
    print("✓ Árbol de decisión listo para predecir emociones")
    print("\nPresiona 'q' para salir")
    print("=" * 60)
except Exception as e:
    print(f"ERROR al cargar el modelo: {e}")
    exit()

# Variables para suavizado temporal
emotion_history = []
history_size = 5

# Captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: No se pudo acceder a la cámara")
    exit()

# Variable para mostrar/ocultar métricas de debug
show_debug = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer de la cámara")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    emotion = "Neutral"
    emotion_color = EMOTION_COLORS["Neutral"]
    confidence = 0.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extraer características y puntos clave
            features, key_points = extract_features(face_landmarks.landmark, w, h)
            
            # ===== PREDICCIÓN CON ÁRBOL DE DECISIÓN =====
            features_reshaped = features.reshape(1, -1)
            prediction = clf.predict(features_reshaped)[0]
            
            # Obtener probabilidades (confianza)
            if hasattr(clf, 'predict_proba'):
                probabilities = clf.predict_proba(features_reshaped)[0]
                confidence = probabilities[prediction]
            
            # Convertir predicción a emoción
            emotion = LABEL_EMOTIONS[prediction]
            
            # Suavizado temporal para estabilidad
            emotion_history.append(emotion)
            if len(emotion_history) > history_size:
                emotion_history.pop(0)
            
            # Emoción más frecuente en la historia
            emotion = max(set(emotion_history), key=emotion_history.count)
            emotion_color = EMOTION_COLORS[emotion]

            # ===== VISUALIZACIÓN =====
            
            # Dibujar puntos clave con el color de la emoción
            points_to_draw = [
                key_points['left_mouth_corner'],
                key_points['right_mouth_corner'],
                key_points['upper_lip_top'],
                key_points['lower_lip_bottom'],
                key_points['left_eyebrow_center'],
                key_points['right_eyebrow_center'],
                key_points['left_eye_top'],
                key_points['right_eye_top'],
                key_points['nose_tip']
            ]
            
            for p in points_to_draw:
                cv2.circle(frame, p, 3, emotion_color, -1)

            # Barra de fondo para el texto principal
            cv2.rectangle(frame, (20, 20), (w-20, 90), (0, 0, 0), -1)
            
            # Mostrar emoción detectada
            cv2.putText(frame, f"Emocion: {emotion}", (30, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, emotion_color, 3)
            
            # Mostrar métricas de debug (si está activado)
            if show_debug:
                debug_y = 120
                feature_names = [
                    'Boca (ratio)',
                    'Curva boca',
                    'Cejas',
                    'Ojos (EAR)',
                    'Nariz-boca',
                    'Labios interior'
                ]
                
                cv2.rectangle(frame, (20, 100), (300, 100 + len(features)*30 + 20), (0, 0, 0), -1)
                cv2.putText(frame, "METRICAS:", (30, debug_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for i, (name, value) in enumerate(zip(feature_names, features)):
                    cv2.putText(frame, f"{name}: {value:.3f}", (30, debug_y + 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Instrucciones
            cv2.putText(frame, "Presiona 'q' para salir | 'd' para debug", (30, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    else:
        # Si no se detecta rostro
        cv2.putText(frame, "NO SE DETECTA ROSTRO", (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Coloca tu cara frente a la camara", (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Detector de Emociones - Decision Tree", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\nCerrando detector...")
        break
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"Modo debug: {'ACTIVADO' if show_debug else 'DESACTIVADO'}")

cap.release()
cv2.destroyAllWindows()

print("=" * 60)
print("Detector finalizado correctamente")
print("=" * 60)