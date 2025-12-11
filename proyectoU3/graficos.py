import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_visualizaciones():
    print("--- GENERANDO VISUALIZACIONES DEL PROYECTO ---")
    
    # 1. Cargar datos
    if not os.path.exists('datos.csv'):
        print("❌ Error: No encuentro el archivo 'datos.csv'")
        return

    df = pd.read_csv('datos.csv')
    
    # 2. Filtrar (Quitamos a Frankenstein para ver solo Gen Z)
    df_genz = df[df['Categoria'] == 'Generacion Z'].copy()
    print(f"📊 Datos cargados: {len(df_genz)} registros de Generación Z.")

    # Crear carpeta para guardar las imágenes
    os.makedirs("graficos", exist_ok=True)
    
    # Configurar estilo visual
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # --- GRÁFICO 1: DISTRIBUCIÓN DE SENTIMIENTOS ---
    # Esto responde a: "¿Predomina la ansiedad o la esperanza?"
    sns.histplot(data=df_genz, x='TonoSentimiento', bins=10, kde=True, color='skyblue')
    plt.title('Distribución Emocional de la Gen Z (1=Negativo, 10=Positivo)')
    plt.xlabel('Nivel de Sentimiento')
    plt.ylabel('Cantidad de Opiniones')
    plt.savefig('graficos/1_distribucion_emocional.png')
    plt.clf() # Limpiar lienzo
    print("✅ Gráfico 1 guardado: Distribución Emocional")

    # --- GRÁFICO 2: FUENTES DE DISCURSO ---
    # Esto responde a: "¿Dónde se da la discusión? (Habermas/Espacio Público)"
    plt.figure(figsize=(8, 5))
    conteo_medios = df_genz['Medio'].value_counts()
    sns.barplot(x=conteo_medios.index, y=conteo_medios.values, palette="viridis")
    plt.title('Plataformas Digitales donde opina la Gen Z')
    plt.ylabel('Número de Interacciones')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('graficos/2_plataformas_digitales.png')
    plt.clf()
    print("✅ Gráfico 2 guardado: Plataformas Digitales")

    # --- GRÁFICO 3: EVOLUCIÓN TEMPORAL (Crisis de sentido) ---
    # Convertir fecha
    df_genz['Fecha'] = pd.to_datetime(df_genz['Fecha'], errors='coerce')
    df_por_fecha = df_genz.groupby(df_genz['Fecha'].dt.to_period('M'))['TonoSentimiento'].mean()
    
    plt.figure(figsize=(12, 6))
    df_por_fecha.plot(kind='line', marker='o', color='coral', linewidth=2)
    plt.title('Evolución del Estado de Ánimo Promedio en el Tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Sentimiento Promedio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graficos/3_evolucion_animo.png')
    
    print("✅ Gráfico 3 guardado: Evolución Temporal")
    print("\n🎉 ¡Listo! Revisa la carpeta 'graficos' para ver tus imágenes.")

if __name__ == "__main__":
    generar_visualizaciones()