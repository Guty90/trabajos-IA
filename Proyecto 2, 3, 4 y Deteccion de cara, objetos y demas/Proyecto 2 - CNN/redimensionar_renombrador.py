from PIL import Image
import os

# ==== CONFIGURACIÓN ====
carpeta = "animales_train/turtles"   # Carpeta con las imágenes
size = (224, 224)            # Tamaño deseado
prefijo = "turtle"            # Prefijo para los nombres

# ==== PROCESAR ====
archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]
contador = 1

for file_name in archivos:
    file_path = os.path.join(carpeta, file_name)
    
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Nuevo nombre: dog_0001.jpg, dog_0002.jpg, etc.
        nuevo_nombre = f"{prefijo}_{contador:04d}.jpg"
        nuevo_path = os.path.join(carpeta, nuevo_nombre)
        
        img.save(nuevo_path, 'JPEG', quality=95, optimize=True)
        
        # Eliminar archivo original si tiene nombre diferente
        if file_path != nuevo_path:
            os.remove(file_path)
        
        print(f"✓ {contador}/{len(archivos)}: {nuevo_nombre}")
        contador += 1
        
    except Exception as e:
        print(f"✗ Error con {file_name}: {e}")
        os.remove(file_path)

print(f"\n🎉 Proceso completado: {contador-1} imágenes procesadas a {size}")