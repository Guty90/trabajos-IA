# -----------------------------
# 1️⃣ Importar librerías
# -----------------------------
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 2️⃣ Cargar dataset
# -----------------------------
df = pd.read_excel("datasetGenZ.xlsx")  # Cambia el nombre al archivo real
df = df[['Comentario_Reaccion']]  # Nos enfocamos en esta columna

# -----------------------------
# 3️⃣ Etiquetar sentimiento automáticamente
# -----------------------------
def obtener_sentimiento(texto):
    polarity = TextBlob(str(texto)).sentiment.polarity
    if polarity > 0:
        return "positivo"
    elif polarity < 0:
        return "negativo"
    else:
        return "neutral"

df['sentimiento'] = df['Comentario_Reaccion'].apply(obtener_sentimiento)

# -----------------------------
# 4️⃣ Preparar datos para ML
# -----------------------------
X = df['Comentario_Reaccion']
y = df['sentimiento']

vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# -----------------------------
# 5️⃣ Entrenar modelo Naive Bayes
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Evaluar modelo
# -----------------------------
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 7️⃣ Probar con comentarios nuevos
# -----------------------------
# comentario_nuevo = ["Esta película fue increíble, me encantó!"]
# comentario_vec = vectorizer.transform(comentario_nuevo)
# prediccion = model.predict(comentario_vec)
# print("Comentario:", comentario_nuevo[0])
# print("Sentimiento predicho:", prediccion[0])
