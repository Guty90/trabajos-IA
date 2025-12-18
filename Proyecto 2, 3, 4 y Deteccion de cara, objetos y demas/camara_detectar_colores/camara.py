import cv2 as cv
import mediapipe as mp
import numpy as np

# Inicializar mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Abrir cámara
cap = cv.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,       # Puede detectar hasta 2 manos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear para que sea como espejo
        frame = cv.flip(frame, 1)

        # Convertir a RGB para mediapipe
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Procesar manos
        result = hands.process(rgb)

        # Fondo negro del mismo tamaño
        output = np.zeros_like(frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Dibujar los puntos y conexiones de la mano en el fondo negro
                mp_drawing.draw_landmarks(
                    output, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2)
                )

        # Mostrar solo las manos sobre camara
        combined = cv.add(frame, output)
        cv.imshow("Manos sobre camara", combined)

        # Tecla ESC para salir
        if cv.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()
