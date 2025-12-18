import cv2
import os
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models

DATASET_DIR = "animales_nuevo/ladybugs"  # <-- cámbialo
OUTPUT_BAD = "malas"

os.makedirs(OUTPUT_BAD, exist_ok=True)

# -----------------------------
# 1. Modelo CLIP-light para detectar dibujos
# -----------------------------
model = models.resnet18(pretrained=True)
model.eval()
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Palabras de referencia
CATEGORIES = ["real photo", "cartoon drawing"]

# -----------------------------
# Funciones
# -----------------------------

def is_cartoon(img):
    """Detecta si una imagen es caricatura usando ResNet18 como extractor."""
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img_t)
    pred = torch.argmax(out)

    # Heurística: valores muy altos en bordes + saturación
    arr = np.array(img)
    edges = cv2.Canny(arr, 100, 200)
    edge_ratio = np.mean(edges > 0)

    if edge_ratio > 0.18:
        return True
    return False


def has_person(path):
    """Detecta personas usando Haarcascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0


def is_blurry(path, thresh=80):
    """Detecta si está borrosa."""
    img = cv2.imread(path)
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < thresh


def is_too_dark(path, thresh=50):
    img = cv2.imread(path)
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = np.mean(gray)
    return bright < thresh


# -----------------------------
# Proceso principal
# -----------------------------
def check_image(path):
    """Regresa True si está mala."""
    try:
        img = Image.open(path).convert("RGB")
    except:
        return True

    if is_cartoon(img):
        return True
    
    if has_person(path):
        return True

    if is_blurry(path):
        return True

    if is_too_dark(path):
        return True

    return False


def run_cleaning():
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if not file.lower().endswith(("jpg","jpeg","png")):
                continue

            full_path = os.path.join(root, file)
            print("Analizando:", full_path)

            if check_image(full_path):
                print(" >> MALA:", full_path)
                shutil.move(full_path, os.path.join(OUTPUT_BAD, file))


if __name__ == "__main__":
    print("Limpieza iniciada...")
    run_cleaning()