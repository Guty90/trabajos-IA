import cv2 as cv
import numpy as np
import mss
import time

# Inicializa el reconocedor y el clasificador
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.read('Fisherface.xml')
faces = ['ElMariana', 'Ger', 'Guty', 'Jesus', 'Tom']
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Define el área de la pantalla que quieres capturar
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# Usa mss para capturar pantalla
sct = mss.mss()

# 🔥 CREA LA VENTANA UNA SOLA VEZ CON CONFIGURACIÓN ESPECÍFICA
window_name = 'Reconocimiento'
cv.namedWindow(window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
cv.resizeWindow(window_name, 960, 540)  # Tamaño más manejable

print("Iniciando reconocimiento... Presiona ESC para salir")

# 🚀 Bucle principal con control de FPS
fps_limit = 30
frame_delay = 1.0 / fps_limit

try:
    while True:
        start_time = time.time()
        
        # Captura una imagen de pantalla
        img = np.array(sct.grab(monitor))
        frame = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rostros = rostro.detectMultiScale(gray, 1.3, 3)
        
        # Procesa cada rostro detectado
        for (x, y, w, h) in rostros:
            rostro_gray = gray[y:y+h, x:x+w]
            rostro_gray = cv.resize(rostro_gray, (100, 100), interpolation=cv.INTER_CUBIC)
            result = faceRecognizer.predict(rostro_gray)
            
            # Dibuja en pantalla
            if result[1] < 2800:
                cv.putText(frame, f'{faces[result[0]]}', (x, y-25), 2, 1.1, (0,255,0), 1, cv.LINE_AA)
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            else:
                cv.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0,0,255), 1, cv.LINE_AA)
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        
        # Muestra el frame
        cv.imshow(window_name, frame)
        
        # Control de teclas - NO BLOQUEAR CON waitKey(1)
        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break
        
        # Verifica si la ventana fue cerrada
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break
        
        # Limita los FPS para reducir carga
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nInterrumpido por el usuario")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Libera recursos de forma segura
    cv.destroyAllWindows()
    sct.close()
    # Asegura que todas las ventanas se cierren
    for i in range(10):
        cv.waitKey(1)
    print("Programa finalizado")