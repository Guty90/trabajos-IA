import cv2 as cv
import numpy as np

# Elegir color # Valores HSV 
print("Elige un color:  1. Verde  2. Rojo  3. Azul 4. Amarillo")
opcion = input("Opción (1/2/3/4): ")
if opcion == '1':
    ubb = (35, 100, 100)
    uba = (85, 255, 255)
elif opcion == '2':
    ubb = (0, 100, 100)
    uba = (20, 255, 255)
    ubb1 = (170, 100, 100)
    uba1 = (190, 255, 255)
elif opcion == '3':
    ubb = (100, 150, 0)
    uba = (140, 255, 255)
elif opcion == '4':
    ubb = (20, 100, 100)
    uba = (30, 255, 255)

# Cargar imagen
img = cv.imread('figura.png')

# Conversión a HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Crear máscara con el rango del color elegido
if opcion == '2': # Rojo tiene dos rangos en HSV
    mascara1 = cv.inRange(hsv, ubb, uba)
    mascara2 = cv.inRange(hsv, ubb1, uba1)
    mascara = cv.bitwise_or(mascara1, mascara2)
else: # Otros colores
    mascara = cv.inRange(hsv, ubb, uba)

# Operaciones morfológicas para limpiar la máscara
num_labels, labels = cv.connectedComponents(mascara)

# Calcular y mostrar centros de las figuras detectadas
for label in range(1, num_labels):  # empezamos en 1 porque 0 es fondo
    ys, xs = np.where(labels == label) # obtener coordenadas de píxeles
    cx = int(xs.sum() / len(xs)) # centroide x
    cy = int(ys.sum() / len(ys)) # centroide y

    # Dibujar el centro en la imagen original
    cv.circle(img, (cx, cy), 5, (0, 0, 0), -1)

    print(f"Centro de figura {label}: ({cx}, {cy})")    

# Mostrar ventanas
cv.imshow('resultado', img)

cv.waitKey(0)
cv.destroyAllWindows()