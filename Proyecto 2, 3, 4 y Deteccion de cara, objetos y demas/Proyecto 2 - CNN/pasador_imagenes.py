import os
import shutil

# 🔹 Carpeta donde están todas las subcarpetas
ORIGEN = r"C:\Users\alexy\Downloads\gy9cqzssm2evflisdj5ti\images.cv_gy9cqzssm2evflisdj5ti\data\test"

# 🔹 Carpeta destino (se crea si no existe)
DESTINO = r"C:\trabajos-IA\12-19\CNN\animales_original\turtles"
os.makedirs(DESTINO, exist_ok=True)

# 🔹 Extensiones de imágenes permitidas
EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

contador = 1

for root, dirs, files in os.walk(ORIGEN):
    for file in files:
        if file.lower().endswith(EXTENSIONES):
            ruta_origen = os.path.join(root, file)

            # Evita nombres duplicados
            nuevo_nombre = f"img_{contador}{os.path.splitext(file)[1]}"
            ruta_destino = os.path.join(DESTINO, nuevo_nombre)

            shutil.copy2(ruta_origen, ruta_destino)
            contador += 1

print("✅ Todas las imágenes fueron copiadas correctamente")
