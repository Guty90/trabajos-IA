# =============================================
# 📊 Análisis completo de corpus en español
# Incluye: sentimiento, temas, tipo y resumen
# =============================================

from pysentimiento import create_analyzer
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
import nltk
import json
import warnings

# Ignorar warnings innecesarios
warnings.filterwarnings("ignore")

# Descargar stopwords si no existen
nltk.download("stopwords")

# =========================
# 1️⃣ Cargar los comentarios
# =========================
with open("posts_paro_tec.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

comentarios = [item["text"] for item in data]

# ===================================
# 2️⃣ Análisis de sentimiento (pysentimiento)
# ===================================
print("Analizando sentimientos...")
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
sentimientos = [sentiment_analyzer.predict(text).output for text in comentarios]

# ===================================
# 3️⃣ Temas más mencionados (Bag of Words)
# ===================================
print("Extrayendo temas más frecuentes...")
stopwords_es = stopwords.words("spanish")
vectorizer = CountVectorizer(stop_words=stopwords_es, max_features=50)
X = vectorizer.fit_transform(comentarios)
temas = Counter(vectorizer.get_feature_names_out())

# ===================================
# 4️⃣ Clasificación del tipo de mensaje
# ===================================
def clasificar_mensaje(texto):
    texto = texto.lower()
    if "no" in texto or "mala" in texto or "odio" in texto:
        return "crítica"
    elif "jaj" in texto or "xd" in texto:
        return "burla"
    elif "?" in texto:
        return "pregunta"
    elif "gracias" in texto or "bien" in texto or "ánimo" in texto:
        return "apoyo"
    else:
        return "información"

tipos = [clasificar_mensaje(t) for t in comentarios]

# ===================================
# 5️⃣ Resumen general (modelo alternativo en español)
# ===================================
print("Generando resumen general...")
summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum"
)

texto_largo = " ".join(comentarios)[:4000]  # evitar que sea demasiado largo
resumen = summarizer(
    texto_largo,
    max_length=150,
    min_length=60,
    do_sample=False
)[0]['summary_text']

# ===================================
# 6️⃣ Resultados finales
# ===================================
print("\n--- 🧠 Sentimiento general ---")
print(Counter(sentimientos))

print("\n--- 🔥 Temas más mencionados ---")
for palabra, freq in temas.most_common(10):
    print(f"{palabra}: {freq}")

print("\n--- 💬 Tipos de mensajes ---")
print(Counter(tipos))

print("\n--- 📋 Resumen general ---")
print(resumen)

# ===================================
# 7️⃣ Guardar resultados
# ===================================
resultados = {
    "sentimientos": dict(Counter(sentimientos)),
    "temas_mas_mencionados": dict(temas.most_common(10)),
    "tipos_de_mensajes": dict(Counter(tipos)),
    "resumen_general": resumen
}

with open("resultados_analisis.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=4)

print("\n✅ Resultados guardados en 'resultados_analisis.json'")
