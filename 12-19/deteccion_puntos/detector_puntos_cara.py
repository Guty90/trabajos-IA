import cv2
import mediapipe as mp
import math
import numpy as np

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Función auxiliar para medir distancia entre puntos
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# Función para calcular la apertura del ojo (EAR - Eye Aspect Ratio)
def eye_aspect_ratio(eye_points):
    # Distancias verticales
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    # Distancia horizontal
    C = distance(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Captura de video
cap = cv2.VideoCapture(0)

# Variables para suavizado temporal
emotion_history = []
history_size = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    emotion = "Neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # ===== BOCA =====
            # Comisuras externas
            left_mouth_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
            right_mouth_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
            
            # Labios superior e inferior (centro)
            upper_lip_top = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            lower_lip_bottom = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            
            # Puntos adicionales de la boca
            upper_lip_inner = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            lower_lip_inner = (int(landmarks[15].x * w), int(landmarks[15].y * h))
            
            # ===== CEJAS =====
            # Ceja izquierda (del usuario)
            left_eyebrow_inner = (int(landmarks[70].x * w), int(landmarks[70].y * h))
            left_eyebrow_center = (int(landmarks[105].x * w), int(landmarks[105].y * h))
            left_eyebrow_outer = (int(landmarks[107].x * w), int(landmarks[107].y * h))
            
            # Ceja derecha
            right_eyebrow_inner = (int(landmarks[300].x * w), int(landmarks[300].y * h))
            right_eyebrow_center = (int(landmarks[334].x * w), int(landmarks[334].y * h))
            right_eyebrow_outer = (int(landmarks[337].x * w), int(landmarks[337].y * h))

            # ===== OJOS =====
            # Ojo izquierdo
            left_eye_inner = (int(landmarks[133].x * w), int(landmarks[133].y * h))
            left_eye_outer = (int(landmarks[33].x * w), int(landmarks[33].y * h))
            left_eye_top = (int(landmarks[159].x * w), int(landmarks[159].y * h))
            left_eye_bottom = (int(landmarks[145].x * w), int(landmarks[145].y * h))
            
            # Ojo derecho
            right_eye_inner = (int(landmarks[362].x * w), int(landmarks[362].y * h))
            right_eye_outer = (int(landmarks[263].x * w), int(landmarks[263].y * h))
            right_eye_top = (int(landmarks[386].x * w), int(landmarks[386].y * h))
            right_eye_bottom = (int(landmarks[374].x * w), int(landmarks[374].y * h))

            # Puntos para EAR del ojo izquierdo
            left_eye_points = [
                left_eye_inner, 
                (int(landmarks[160].x * w), int(landmarks[160].y * h)),
                left_eye_top,
                left_eye_outer,
                (int(landmarks[144].x * w), int(landmarks[144].y * h)),
                left_eye_bottom
            ]
            
            # Puntos para EAR del ojo derecho
            right_eye_points = [
                right_eye_inner,
                (int(landmarks[387].x * w), int(landmarks[387].y * h)),
                right_eye_top,
                right_eye_outer,
                (int(landmarks[373].x * w), int(landmarks[373].y * h)),
                right_eye_bottom
            ]

            # ===== NARIZ =====
            nose_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            nose_bridge = (int(landmarks[168].x * w), int(landmarks[168].y * h))

            # ===== CÁLCULO DE MÉTRICAS =====
            
            # Ancho de la boca
            mouth_width = distance(left_mouth_corner, right_mouth_corner)
            
            # Apertura vertical de la boca
            mouth_height = distance(upper_lip_top, lower_lip_bottom)
            
            # Apertura interna de los labios
            mouth_inner_height = distance(upper_lip_inner, lower_lip_inner)
            
            # Ratio de apertura de boca
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Posición de las comisuras (si están hacia arriba o abajo)
            mouth_center_y = (upper_lip_top[1] + lower_lip_bottom[1]) / 2
            mouth_corners_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2
            mouth_curve = mouth_center_y - mouth_corners_y  # Positivo = sonrisa, Negativo = ceño
            
            # Distancia entre cejas y ojos
            left_brow_eye_dist = distance(left_eyebrow_center, left_eye_top)
            right_brow_eye_dist = distance(right_eyebrow_center, right_eye_top)
            avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
            
            # Normalizar con respecto al ancho de la cara
            face_width = distance(left_eye_outer, right_eye_outer)
            brow_ratio = avg_brow_eye_dist / face_width if face_width > 0 else 0
            
            # EAR para detectar ojos abiertos/cerrados
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2
            
            # Distancia nariz-boca (para detectar tensión facial)
            nose_mouth_dist = distance(nose_tip, upper_lip_top)
            nose_mouth_ratio = nose_mouth_dist / face_width if face_width > 0 else 0

            # ===== DETECCIÓN DE EMOCIONES MEJORADA =====
            
            scores = {
                "Feliz 😄": 0,
                "Triste 😢": 0,
                "Enojado 😠": 0,
                "Sorprendido 😮": 0,
                "Neutral 😐": 0
            }
            
            # FELIZ: Boca sonriente, comisuras arriba, ojos normales
            if mouth_curve > 5 and mouth_ratio < 0.4:
                scores["Feliz 😄"] += 3
            if mouth_ratio > 0.15 and mouth_ratio < 0.35 and mouth_curve > 3:
                scores["Feliz 😄"] += 2
            if brow_ratio > 0.25 and brow_ratio < 0.35:
                scores["Feliz 😄"] += 1
                
            # TRISTE: Boca hacia abajo, cejas ligeramente juntas, ojos normales
            if mouth_curve < -3:
                scores["Triste 😢"] += 3
            if mouth_ratio < 0.2 and mouth_curve < -1:
                scores["Triste 😢"] += 2
            if brow_ratio > 0.28 and brow_ratio < 0.35:
                scores["Triste 😢"] += 1
            if avg_ear < 0.25:
                scores["Triste 😢"] += 1
                
            # ENOJADO: Cejas hacia abajo, boca tensa
            if brow_ratio < 0.25:
                scores["Enojado 😠"] += 3
            if mouth_ratio < 0.2 and abs(mouth_curve) < 3:
                scores["Enojado 😠"] += 2
            if nose_mouth_ratio < 0.22:
                scores["Enojado 😠"] += 1
                
            # SORPRENDIDO: Boca muy abierta, cejas arriba, ojos abiertos
            if mouth_ratio > 0.45:
                scores["Sorprendido 😮"] += 3
            if brow_ratio > 0.35:
                scores["Sorprendido 😮"] += 2
            if avg_ear > 0.28:
                scores["Sorprendido 😮"] += 2
                
            # NEUTRAL: valores intermedios
            if 0.15 <= mouth_ratio <= 0.25 and abs(mouth_curve) < 3:
                scores["Neutral 😐"] += 2
            if 0.25 <= brow_ratio <= 0.32:
                scores["Neutral 😐"] += 2
            if 0.22 <= avg_ear <= 0.27:
                scores["Neutral 😐"] += 1

            # Seleccionar la emoción con mayor puntaje
            emotion = max(scores, key=scores.get)
            
            # Suavizado temporal
            emotion_history.append(emotion)
            if len(emotion_history) > history_size:
                emotion_history.pop(0)
            
            # Emoción más frecuente en la historia
            emotion = max(set(emotion_history), key=emotion_history.count)

            # ===== VISUALIZACIÓN =====
            # Dibujar puntos clave
            key_points = [
                left_mouth_corner, right_mouth_corner, upper_lip_top, lower_lip_bottom,
                left_eyebrow_center, right_eyebrow_center, 
                left_eye_top, right_eye_top, nose_tip
            ]
            
            for p in key_points:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            # Mostrar información
            cv2.putText(frame, f"Emocion: {emotion}", (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Mostrar métricas de debug (opcional)
            debug_y = 90
            cv2.putText(frame, f"Boca: {mouth_ratio:.2f}", (30, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Curva: {mouth_curve:.1f}", (30, debug_y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Cejas: {brow_ratio:.2f}", (30, debug_y+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Ojos: {avg_ear:.2f}", (30, debug_y+75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Detector de emociones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()