from bing_image_downloader import downloader

# === Configura aquí tu clase y cuántas imágenes quieres ===
clase = "turtle"        # ejemplo: tortuga
cantidad = 1000          # cuántas imágenes descargar

# === Descarga automática ===
downloader.download(
    query=clase,
    limit=cantidad,
    output_dir='animales_nuevo',   # carpeta donde se guardará
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)

print(f"Descarga completada. Imágenes guardadas en sportimages/{clase}")
