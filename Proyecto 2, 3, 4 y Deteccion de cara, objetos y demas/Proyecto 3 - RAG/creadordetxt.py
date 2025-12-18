import csv

def convertir_csv_a_txt(archivo_csv, archivo_txt):
    """
    Lee un CSV de tweets y genera un TXT con formato específico
    """
    
    try:
        with open(archivo_csv, 'r', encoding='utf-8') as csv_file:
            lector = csv.DictReader(csv_file)
            
            with open(archivo_txt, 'w', encoding='utf-8') as txt_file:
                
                for i, fila in enumerate(lector, 1):
                    # Formato para cada tweet
                    txt_file.write(f"[TEMA]: {fila['tema']}\n")
                    txt_file.write(f"[TEXTO]: {fila['texto']}\n")
                    txt_file.write(f"[SENTIMIENTO]: {fila['sentimiento']}\n")
                    txt_file.write("\n")  # Línea en blanco entre tweets
                    
                    # Mostrar progreso cada 1000 registros
                    if i % 1000 == 0:
                        print(f"   Procesados: {i} registros...")
        
        print(f"\n✅ Conversión completada!")
        print(f"📊 Total de tweets procesados: {i}")
        print(f"📁 Archivo guardado como: {archivo_txt}")
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{archivo_csv}'")
    except Exception as e:
        print(f"❌ Error al procesar el archivo: {str(e)}")

# Configuración
archivo_entrada = 'dataset_tweets_sin_duplicados.csv'  # Nombre del CSV de entrada
archivo_salida = 'tweets_formateados.txt'  # Nombre del TXT de salida

print("🔄 Iniciando conversión de CSV a TXT...\n")
convertir_csv_a_txt(archivo_entrada, archivo_salida)