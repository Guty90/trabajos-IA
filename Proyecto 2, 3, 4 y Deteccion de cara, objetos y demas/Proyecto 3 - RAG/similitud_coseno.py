import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def remove_duplicates_by_similarity(csv_path, threshold=0.85, output_path='datos_sin_duplicados.csv'):
    # Leer CSV
    df = pd.read_csv(csv_path)
    print(f"Registros originales: {len(df)}")
    
    # Verificar que existe la columna 'texto'
    if 'texto' not in df.columns:
        raise ValueError("El CSV debe tener una columna 'texto'")
    
    # Lista de stopwords en español
    stopwords_es = [
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
        'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
        'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
        'la', 'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
        'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo',
        'yo', 'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero',
        'desde', 'grande', 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella',
        'sí', 'día', 'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
        'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde', 'ahora', 'parte',
        'después', 'vida', 'quedar', 'siempre', 'creer', 'hablar', 'llevar', 'dejar',
        'nada', 'cada', 'seguir', 'menos', 'nuevo', 'encontrar', 'algo', 'solo',
        'decir', 'ahi', 'aquel', 'te', 'cómo', 'les', 'has', 'del', 'al', 'los',
        'las', 'unos', 'unas', 'es', 'son', 'era', 'fue', 'sido', 'ha', 'han',
        'he', 'hemos', 'había', 'estaba', 'está', 'están', 'estoy', 'tuvo', 'tiene',
        'tienen', 'tengo', 'tenía', 'eres', 'soy', 'somos', 'era', 'fueron'
    ]
    
    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        stop_words=stopwords_es,
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    
    # Vectorizar los textos
    tfidf_matrix = vectorizer.fit_transform(df['texto'].fillna(''))
    
    # Calcular matriz de similitud coseno
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Identificar duplicados
    indices_to_remove = set()
    
    for i in range(len(similarity_matrix)):
        if i in indices_to_remove:
            continue
            
        for j in range(i + 1, len(similarity_matrix)):
            if j in indices_to_remove:
                continue
                
            # Si la similitud supera el umbral, marcar como duplicado
            if similarity_matrix[i][j] >= threshold:
                indices_to_remove.add(j)
                print(f"Duplicado encontrado (similitud: {similarity_matrix[i][j]:.3f}):")
                print(f"  Original [{i}]: {df.iloc[i]['texto'][:80]}...")
                print(f"  Duplicado [{j}]: {df.iloc[j]['texto'][:80]}...")
                print()
    
    # Crear DataFrame sin duplicados
    df_clean = df.drop(index=list(indices_to_remove)).reset_index(drop=True)
    
    print(f"\nDuplicados eliminados: {len(indices_to_remove)}")
    print(f"Registros finales: {len(df_clean)}")
    
    # Guardar resultado
    df_clean.to_csv(output_path, index=False)
    print(f"\nArchivo guardado en: {output_path}")
    
    return df_clean

# Uso del script
if __name__ == "__main__":
    # Ajusta estos parámetros según necesites
    CSV_INPUT = 'dataset_tweets.csv'  
    UMBRAL_SIMILITUD = 0.85  # Ajusta entre 0 y 1 (mayor = más estricto)
    CSV_OUTPUT = 'dataset_tweets_sin_duplicados.csv'
    
    try:
        df_limpio = remove_duplicates_by_similarity(
            csv_path=CSV_INPUT,
            threshold=UMBRAL_SIMILITUD,
            output_path=CSV_OUTPUT
        )
        
        print("\n¡Proceso completado exitosamente!")
        print(f"Puedes revisar el archivo: {CSV_OUTPUT}")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{CSV_INPUT}'")
    except Exception as e:
        print(f"Error: {e}")