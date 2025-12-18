import cv2 as cv
import numpy as np

# Abrir cámara
cap = cv.VideoCapture(0)

# Fondo negro (canvas donde se dibuja)
canvas = None

# Rango de color (ajústalo al color del lápiz que uses)
# Ejemplo: azul
lower_color = np.array([35, 100, 100])
upper_color = np.array([85, 255, 255])

# Guardar posición anterior
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    # Inicializar el canvas (fondo negro)
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convertir a HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Crear máscara del color del lápiz
    mask = cv.inRange(hsv, lower_color, upper_color)

    # Filtrar ruido
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # Detectar contornos
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Tomar el contorno más grande
        c = max(contours, key=cv.contourArea)
        if cv.contourArea(c) > 500:
            x, y, w, h = cv.boundingRect(c)
            cx, cy = x + w//2, y + h//2

            # Dibujar línea continua según el movimiento
            if prev_x != 0 and prev_y != 0:
                cv.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 4)

            prev_x, prev_y = cx, cy
        else:
            prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Mostrar sobre camara
    combined = cv.add(frame, canvas)
    cv.imshow("Dibujo con color", combined)

    # Mostrar máscara para depuración (opcional)
    # cv.imshow("Mascara color", mask)

    # Teclas de control
    key = cv.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)  # Limpiar dibujo
    elif key == 27:  # ESC para salir
        break

cap.release()
cv.destroyAllWindows()
