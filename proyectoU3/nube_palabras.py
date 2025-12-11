import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def crear_nube():
    print("--- GENERANDO VISUALIZACIÓN SEMÁNTICA (NUBE DE PALABRAS) ---")
    
    # Cargar datos limpios
    if not os.path.exists('datos.csv'):
        print("❌ Faltan los datos.")
        return

    df = pd.read_csv('datos.csv')
    df = df[df['Categoria'] == 'Generacion Z']

    # Unir todos los comentarios en un solo texto gigante
    texto_completo = " ".join(comentario for comentario in df['ComentarioReaccion'])

    # Configurar la nube de palabras (quitando palabras comunes irrelevantes)
    stopwords_es = ["de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "porque", "muy", "sin", "sobre", "tambien", "me", "hasta", "donde", "quien", "desde", "nos", "durante", "uno", "ni", "contra", "ese", "eso", "mí", "mis", "tengo", "esta", "estamos"]

    wordcloud = WordCloud(
        width=1600, 
        height=800, 
        background_color='white', 
        stopwords=stopwords_es,
        colormap='magma', # Colores "intensos"
        min_font_size=10
    ).generate(texto_completo)

    # Mostrar y guardar
    plt.figure(figsize=(20, 10), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    os.makedirs("graficos", exist_ok=True)
    plt.savefig('graficos/4_nube_semantica.png')
    print("✅ Nube de palabras guardada en 'graficos/4_nube_semantica.png'")

if __name__ == "__main__":
    crear_nube()