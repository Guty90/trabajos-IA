import cv2
import mediapipe as mp
import math
import numpy as np
from sklearn import tree
import pickle

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mapeo de emociones a números
EMOTION_LABELS = {
    "Neutral": 0,
    "Feliz": 1,
    "Triste": 2,
    "Enojado": 3,
    "Sorprendido": 4
}

LABEL_EMOTIONS = {v: k for k, v in EMOTION_LABELS.items()}

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(eye_points):
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def extract_features(landmarks, w, h):
    """Extrae las características faciales necesarias para el clasificador"""
    
    # BOCA
    left_mouth_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
    right_mouth_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
    upper_lip_top = (int(landmarks[13].x * w), int(landmarks[13].y * h))
    lower_lip_bottom = (int(landmarks[14].x * w), int(landmarks[14].y * h))
    upper_lip_inner = (int(landmarks[12].x * w), int(landmarks[12].y * h))
    lower_lip_inner = (int(landmarks[15].x * w), int(landmarks[15].y * h))
    
    # CEJAS
    left_eyebrow_center = (int(landmarks[105].x * w), int(landmarks[105].y * h))
    right_eyebrow_center = (int(landmarks[334].x * w), int(landmarks[334].y * h))
    
    # OJOS
    left_eye_inner = (int(landmarks[133].x * w), int(landmarks[133].y * h))
    left_eye_outer = (int(landmarks[33].x * w), int(landmarks[33].y * h))
    left_eye_top = (int(landmarks[159].x * w), int(landmarks[159].y * h))
    left_eye_bottom = (int(landmarks[145].x * w), int(landmarks[145].y * h))
    
    right_eye_inner = (int(landmarks[362].x * w), int(landmarks[362].y * h))
    right_eye_outer = (int(landmarks[263].x * w), int(landmarks[263].y * h))
    right_eye_top = (int(landmarks[386].x * w), int(landmarks[386].y * h))
    right_eye_bottom = (int(landmarks[374].x * w), int(landmarks[374].y * h))
    
    # Puntos para EAR
    left_eye_points = [
        left_eye_inner,
        (int(landmarks[160].x * w), int(landmarks[160].y * h)),
        left_eye_top,
        left_eye_outer,
        (int(landmarks[144].x * w), int(landmarks[144].y * h)),
        left_eye_bottom
    ]
    
    right_eye_points = [
        right_eye_inner,
        (int(landmarks[387].x * w), int(landmarks[387].y * h)),
        right_eye_top,
        right_eye_outer,
        (int(landmarks[373].x * w), int(landmarks[373].y * h)),
        right_eye_bottom
    ]
    
    # NARIZ
    nose_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
    
    # CÁLCULOS DE MÉTRICAS
    mouth_width = distance(left_mouth_corner, right_mouth_corner)
    mouth_height = distance(upper_lip_top, lower_lip_bottom)
    mouth_inner_height = distance(upper_lip_inner, lower_lip_inner)
    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
    
    mouth_center_y = (upper_lip_top[1] + lower_lip_bottom[1]) / 2
    mouth_corners_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2
    mouth_curve = mouth_center_y - mouth_corners_y
    
    left_brow_eye_dist = distance(left_eyebrow_center, left_eye_top)
    right_brow_eye_dist = distance(right_eyebrow_center, right_eye_top)
    avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
    
    face_width = distance(left_eye_outer, right_eye_outer)
    brow_ratio = avg_brow_eye_dist / face_width if face_width > 0 else 0
    
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2
    
    nose_mouth_dist = distance(nose_tip, upper_lip_top)
    nose_mouth_ratio = nose_mouth_dist / face_width if face_width > 0 else 0
    
    # Retornar características como array
    features = [
        mouth_ratio,      # 0: Proporción de apertura de boca
        mouth_curve,      # 1: Curvatura de la boca (sonrisa o ceño)
        brow_ratio,       # 2: Distancia cejas-ojos normalizada
        avg_ear,          # 3: Apertura de los ojos
        nose_mouth_ratio, # 4: Distancia nariz-boca
        mouth_inner_height / face_width if face_width > 0 else 0  # 5: Apertura interna de labios
    ]
    
    return np.array(features)

# ===== PROGRAMA PRINCIPAL DE ENTRENAMIENTO =====
print("=" * 60)
print("MODO ENTRENAMIENTO - DETECTOR DE EMOCIONES")
print("=" * 60)
print("\nINSTRUCCIONES:")
print("1. Haz una expresión facial clara")
print("2. Presiona la tecla correspondiente para etiquetarla:")
print("   [1] = Neutral")
print("   [2] = Feliz")
print("   [3] = Triste")
print("   [4] = Enojado")
print("   [5] = Sorprendido")
print("   [q] = Terminar y guardar modelo")
print("\nRECOMENDACION: Captura al menos 20-30 muestras de cada emoción")
print("=" * 60)
print("\nPresiona cualquier tecla para comenzar...")
input()

cap = cv2.VideoCapture(0)
training_data = []
training_labels = []

# Contador de muestras por emoción
emotion_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer de la cámara")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            features = extract_features(face_landmarks.landmark, w, h)
            
            # Mostrar instrucciones en pantalla
            cv2.rectangle(frame, (20, 20), (w-20, 200), (0, 0, 0), -1)
            cv2.putText(frame, "HAZ UNA EXPRESION Y PRESIONA:", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "1:Neutral 2:Feliz 3:Triste", (30, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "4:Enojado 5:Sorprendido  Q:Salir", (30, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Mostrar contador de muestras
            cv2.putText(frame, f"TOTAL: {len(training_data)} muestras", (30, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar desglose por emoción
            y_pos = 230
            for i, emotion_name in LABEL_EMOTIONS.items():
                count = emotion_counts[i]
                color = (0, 255, 0) if count >= 20 else (0, 165, 255)
                cv2.putText(frame, f"{emotion_name}: {count}", (30, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 25
    else:
        cv2.putText(frame, "NO SE DETECTA ROSTRO", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Entrenamiento - Detector de Emociones", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\nFinalizando entrenamiento...")
        break
    elif key >= ord('1') and key <= ord('5'):
        emotion_label = key - ord('1')  # Convierte tecla a número 0-4
        if results.multi_face_landmarks:
            training_data.append(features)
            training_labels.append(emotion_label)
            emotion_counts[emotion_label] += 1
            emotion_name = LABEL_EMOTIONS[emotion_label]
            print(f"✓ Muestra #{len(training_data)}: {emotion_name} - Total {emotion_name}: {emotion_counts[emotion_label]}")

cap.release()
cv2.destroyAllWindows()

# ===== ENTRENAR Y GUARDAR EL MODELO =====
print("\n" + "=" * 60)
if len(training_data) > 0:
    print(f"ENTRENANDO MODELO con {len(training_data)} muestras...")
    print("=" * 60)
    
    X = np.array(training_data)
    y = np.array(training_labels)
    
    # Crear y entrenar el árbol de decisión
    clf = tree.DecisionTreeClassifier(
        max_depth=5,              # Profundidad máxima del árbol
        min_samples_split=3,      # Mínimo de muestras para dividir un nodo
        random_state=42           # Semilla para reproducibilidad
    )
    clf = clf.fit(X, y)
    
    # Mostrar precisión del modelo
    accuracy = clf.score(X, y)
    print(f"\n✓ Modelo entrenado exitosamente!")
    print(f"✓ Precisión en datos de entrenamiento: {accuracy * 100:.2f}%")
    
    # Guardar el modelo
    with open('emotion_tree.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print(f"✓ Modelo guardado como 'emotion_tree.pkl'")
    
    # Mostrar estadísticas
    print("\n" + "-" * 60)
    print("ESTADÍSTICAS DE MUESTRAS:")
    print("-" * 60)
    for i, emotion_name in LABEL_EMOTIONS.items():
        count = emotion_counts[i]
        percentage = (count / len(training_data)) * 100
        print(f"{emotion_name:12s}: {count:3d} muestras ({percentage:.1f}%)")
    
    # Opcional: Visualizar el árbol de decisión
    print("\n" + "-" * 60)
    try:
        import graphviz
        feature_names = [
            'mouth_ratio',
            'mouth_curve', 
            'brow_ratio',
            'avg_ear',
            'nose_mouth_ratio',
            'mouth_inner'
        ]
        
        dot_data = tree.export_graphviz(
            clf, 
            out_file=None,
            feature_names=feature_names,
            class_names=list(LABEL_EMOTIONS.values()),
            filled=True, 
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        graph.render("emotion_decision_tree")
        print("✓ Árbol de decisión visualizado en 'emotion_decision_tree.pdf'")
    except ImportError:
        print("⚠ No se pudo generar la visualización (instala graphviz)")
    except Exception as e:
        print(f"⚠ Error al generar visualización: {e}")
    
    print("=" * 60)
    print("\n¡LISTO! Ahora puedes ejecutar el archivo de DETECCIÓN")
    print("=" * 60)
    
else:
    print("ERROR: No se recolectaron muestras")
    print("El entrenamiento fue cancelado")