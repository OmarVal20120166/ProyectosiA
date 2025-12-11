import pandas as pd
import os

def procesar_dataset():
    print("--- PROCESANDO DATASET PARA RAG ---")
    
    # 1. Cargar el archivo CSV corregido
    # Asegúrate de que el archivo 'datos.csv' esté en la misma carpeta
    try:
        df = pd.read_csv('datos.csv')
        print(f"✅ Archivo cargado con {len(df)} filas.")
    except FileNotFoundError:
        print("❌ Error: No encuentro 'datos.csv'. Revisa el nombre.")
        return

    # 2. Filtrar solo lo que nos interesa (Generación Z)
    # Quitamos lo de 'Frankenstein' u otros temas
    df_genz = df[df['Categoria'] == 'Generacion Z'].copy()
    print(f"✅ Filas filtradas (Solo Gen Z): {len(df_genz)}")

    # 3. Crear la carpeta de datos si no existe
    os.makedirs("datos", exist_ok=True)

    # 4. Convertir cada fila en un formato legible para la IA y guardarlo
    # Vamos a unir el Título con el Comentario para darle más contexto
    contenido_texto = ""
    
    for index, row in df_genz.iterrows():
        # Limpiamos el texto de posibles errores
        titulo = str(row['Titulo']).strip()
        comentario = str(row['ComentarioReaccion']).strip()
        medio = str(row['Medio']).strip()
        
        # Formato: [FUENTE] TEMA: COMENTARIO
        fragmento = f"Fuente: {medio} | Tema: {titulo}\nOpinión: {comentario}\n\n"
        contenido_texto += fragmento

    # Guardamos todo en un solo archivo robusto para el RAG
    ruta_salida = "datos/GenZ_redes.txt"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(contenido_texto)

    print(f"🎉 ¡Listo! Se ha creado el archivo '{ruta_salida}' con todos los datos limpios.")
    print("   Ahora puedes ejecutar tu 'main_rag.py' y leerá estos datos reales.")

if __name__ == "__main__":
    procesar_dataset()