from PIL import Image, ImageOps
import os

CARPETA = r"C:\trabajos-IA\12-19\CNN\animales_original\turtles"
GRADOS = 90
MAX_IMAGENES = 1000

EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp")

contador = 0

for archivo in os.listdir(CARPETA):
    if contador >= MAX_IMAGENES:
        break

    if archivo.lower().endswith(EXTENSIONES):
        ruta = os.path.join(CARPETA, archivo)
        try:
            with Image.open(ruta) as img:
                img = ImageOps.exif_transpose(img)
                img_rotada = img.rotate(GRADOS, expand=True)
                img_rotada.save(ruta)
                contador += 1
                print(f"✔ {contador}/1000 → {archivo}")
        except Exception as e:
            print(f"❌ Error con {archivo}: {e}")

print(f"✅ Listo. Se rotaron exactamente {contador} imágenes")
