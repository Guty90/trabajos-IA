import cv2 as cvv
import numpy as np
import os

# Ruta del dataset (ajústala si la tienes en otro lugar)
dataSet = 'C:\\deteccion_facial\\Datasets'

# Obtener las carpetas (una por persona)
faces = os.listdir(dataSet)
print("Personas encontradas:", faces)

# Listas para guardar datos e identificadores
labels = []
facesData = []
label = 0

print("Entrenando modelo Fisherfaces...")

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

# Crear el reconocedor con Fisherfaces
faceRecognizer = cvv.face.FisherFaceRecognizer_create()

# Entrenar el modelo con los rostros y etiquetas
faceRecognizer.train(facesData, labels)

# Guardar el modelo entrenado
faceRecognizer.write('Fisherface.xml')
print("Entrenamiento completado con éxito ✅")
