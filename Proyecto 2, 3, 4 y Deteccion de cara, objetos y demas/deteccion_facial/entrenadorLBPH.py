import cv2 as cvv
import numpy as np
import os
import tqdm

# Ruta del dataset (ajústala si la tienes en otro lugar)
dataSet = 'C:\\deteccion_facial\\Datasets'

# Obtener las carpetas (una por persona)
faces = os.listdir(dataSet)
print("Personas encontradas:", faces)

# Listas para guardar datos e identificadores
labels = []
facesData = []
label = 0

print("Entrenando modelo LBPH...")

# Recorremos cada carpeta (una por persona)
for face in faces:
    facePath = os.path.join(dataSet, face)
    print(f"Procesando imágenes de: {face}")
    
    for faceName in os.listdir(facePath):
        # Leer imagen en escala de grises
        imgPath = os.path.join(facePath, faceName)
        image = cvv.imread(imgPath, 0)
        
        # Validar que la imagen se haya leído correctamente
        if image is None:
            print(f"[ADVERTENCIA] No se pudo leer {imgPath}")
            continue

        facesData.append(image)
        labels.append(label)
    
    label += 1

# Convertir a arreglos numpy
labels = np.array(labels)

print(f"Cantidad de imágenes: {len(facesData)}")
print(f"Cantidad de etiquetas: {len(labels)}")
print("Empezando el entrenamiento...")

# Crear el reconocedor con LBPH
faceRecognizer = cvv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=123.0)

# Entrenar el modelo con los rostros y etiquetas
faceRecognizer.train(facesData, labels)

# Guardar el modelo entrenado
faceRecognizer.write('LBPH.xml')
print("Entrenamiento completado con éxito ✅")
