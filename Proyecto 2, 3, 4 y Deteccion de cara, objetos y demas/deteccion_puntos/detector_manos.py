import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

rect_width = 200
rect_height = 100
angle = 0
scale = 1.0

# Para mantener escala previa
last_distance = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rect_center = (w // 2, h // 2)  # Fijo al centro

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    index_points = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[8]  # Punto del índice
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            index_points.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

    # Si hay dos manos detectadas
    if len(index_points) == 2:
        (x1, y1), (x2, y2) = index_points

        # Calcular distancia entre los índices
        distance = math.hypot(x2 - x1, y2 - y1)

        # Calcular ángulo de rotación
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Escalado suave con base en la distancia
        if last_distance is not None:
            diff = distance - last_distance
            scale += diff / 300.0  # Sensibilidad del zoom
            scale = max(0.3, min(scale, 3.0))  # Limitar tamaño

        last_distance = distance

    # Dibujar el rectángulo rotado y escalado
    scaled_w = int(rect_width * scale)
    scaled_h = int(rect_height * scale)

    box = cv2.boxPoints(((rect_center[0], rect_center[1]), (scaled_w, scaled_h), angle))
    box = np.intp(box)
    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

    cv2.putText(frame, f"Angulo: {angle:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Escala: {scale:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Rectangulo fijo controlado por dedos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
