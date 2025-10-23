import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

rect_center = None
rect_width = 200
rect_height = 100
angle = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    index_points = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[8]  # Índice
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            index_points.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

    # Si se detectan las dos manos
    if len(index_points) == 2:
        (x1, y1), (x2, y2) = index_points

        # Centro del rectángulo
        rect_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Distancia entre índices (para escalar)
        distance = math.hypot(x2 - x1, y2 - y1)

        # Calcular ángulo de rotación
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Escala con base a la distancia
        scale = distance / 200.0
        scaled_w = int(rect_width * scale)
        scaled_h = int(rect_height * scale)

        # Dibujar el rectángulo rotado
        box = cv2.boxPoints(((rect_center[0], rect_center[1]), (scaled_w, scaled_h), angle))
        box = np.intp(box)
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

    cv2.imshow("Rectángulo con dedos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
