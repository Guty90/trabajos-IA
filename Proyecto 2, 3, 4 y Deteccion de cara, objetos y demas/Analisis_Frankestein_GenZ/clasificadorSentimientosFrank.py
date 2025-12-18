# -----------------------------
# 1️⃣ Importar librerías
# -----------------------------
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# 2️⃣ Cargar dataset
# -----------------------------
df = pd.read_excel("datasetFrankestein.xlsx")  # Cambia al nombre de tu archivo
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]  # Normalizar columnas

# -----------------------------
# 3️⃣ Inicializar VADER
# -----------------------------
analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# 4️⃣ Función para obtener sentimiento
# -----------------------------
def obtener_sentimiento(texto):
    puntaje = analyzer.polarity_scores(str(texto))
    if puntaje['compound'] >= 0.05:
        return "positivo"
    elif puntaje['compound'] <= -0.05:
        return "negativo"
    else:
        return "neutral"

# -----------------------------
# 5️⃣ Aplicar a todo el dataset
# -----------------------------
df['sentimiento'] = df['comentario_reaccion'].apply(obtener_sentimiento)

# -----------------------------
# 6️⃣ Guardar resultado en Excel
# -----------------------------
df.to_excel("dataset_sentimiento.xlsx", index=False)
print("✅ Archivo 'dataset_sentimiento.xlsx' creado con comentarios y sentimiento.")

# -----------------------------
# 7️⃣ Contar y mostrar porcentaje de cada clase
# -----------------------------
conteo = df['sentimiento'].value_counts()
porcentaje = df['sentimiento'].value_counts(normalize=True) * 100

print("\n=== Conteo de cada sentimiento ===")
print(conteo)
print("\n=== Porcentaje de cada sentimiento ===")
print(porcentaje.round(2))
