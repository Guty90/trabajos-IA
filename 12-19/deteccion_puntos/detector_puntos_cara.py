import cv2
import mediapipe as mp
import math

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

# Captura de video
cap = cv2.VideoCapture(0)

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
            # Puntos relevantes
            landmarks = face_landmarks.landmark

            # Comisuras de la boca
            left_mouth = (int(landmarks[61].x * w), int(landmarks[61].y * h))
            right_mouth = (int(landmarks[291].x * w), int(landmarks[291].y * h))
            top_mouth = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            bottom_mouth = (int(landmarks[14].x * w), int(landmarks[14].y * h))

            # Cejas
            left_eyebrow = (int(landmarks[105].x * w), int(landmarks[105].y * h))
            right_eyebrow = (int(landmarks[334].x * w), int(landmarks[334].y * h))

            # Ojos (para referencia)
            left_eye = (int(landmarks[159].x * w), int(landmarks[159].y * h))
            right_eye = (int(landmarks[386].x * w), int(landmarks[386].y * h))

            # Distancias
            mouth_width = distance(left_mouth, right_mouth)
            mouth_open = distance(top_mouth, bottom_mouth)
            brow_to_eye = (distance(left_eyebrow, left_eye) + distance(right_eyebrow, right_eye)) / 2

            # Heurística simple
            ratio_mouth = mouth_open / mouth_width
            ratio_brow = brow_to_eye / mouth_width

            # Detección básica de emoción
            if ratio_mouth > 0.32 and ratio_brow > 0.09:
                emotion = "Feliz 😄"
            elif ratio_brow < 0.065:
                emotion = "Enojado 😠"
            elif ratio_mouth < 0.25 and ratio_brow > 0.08:
                emotion = "Triste 😢"
            else:
                emotion = "Neutral 😐"

            # Dibujar puntos y texto
            for p in [left_mouth, right_mouth, top_mouth, bottom_mouth, left_eyebrow, right_eyebrow]:
                cv2.circle(frame, p, 3, (0, 255, 0), -1)

            cv2.putText(frame, f"Emocion: {emotion}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Detector de emociones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
