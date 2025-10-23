import cv2 as cv
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Obtener tamaño de pantalla
screen_width, screen_height = pyautogui.size()

cap = cv.VideoCapture(0)

# Variables para suavizar movimiento del cursor
prev_x, prev_y = 0, 0
smooth = 2  # Cuanto más alto, más suave

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Coordenadas normalizadas de los dedos
            index_finger = hand_landmarks.landmark[8]   # punta del índice
            thumb_finger = hand_landmarks.landmark[4]   # punta del pulgar

            # Convertir a coordenadas de la cámara
            h, w, _ = frame.shape
            x_index, y_index = int(index_finger.x * w), int(index_finger.y * h)
            x_thumb, y_thumb = int(thumb_finger.x * w), int(thumb_finger.y * h)

            # Dibujar puntos
            cv.circle(frame, (x_index, y_index), 8, (0, 255, 0), -1)
            cv.circle(frame, (x_thumb, y_thumb), 8, (0, 0, 255), -1)
            cv.line(frame, (x_index, y_index), (x_thumb, y_thumb), (255, 255, 255), 2)

            # Convertir a coordenadas de pantalla
            screen_x = np.interp(x_index, (0, w), (0, screen_width))
            screen_y = np.interp(y_index, (0, h), (0, screen_height))

            # Suavizar movimiento
            cur_x = prev_x + (screen_x - prev_x) / smooth
            cur_y = prev_y + (screen_y - prev_y) / smooth
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            # Calcular distancia entre pulgar e índice
            distance = math.hypot(x_thumb - x_index, y_thumb - y_index)

            # Si se acercan lo suficiente → clic
            if distance < 30:
                cv.putText(frame, "CLICK", (x_index, y_index - 20),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                pyautogui.click()
                pyautogui.sleep(0.2)  # pequeña pausa para evitar múltiples clics

    cv.imshow("Control por mano", frame)
    if cv.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv.destroyAllWindows()
