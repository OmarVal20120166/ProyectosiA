# Proyecto U3 - Análisis Gen Z

## 🚀 Configuración del Entorno

### 1. Activar el entorno virtual

**En PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**En CMD:**
```cmd
venv\Scripts\activate.bat
```

Sabrás que está activado cuando veas `(venv)` al inicio de tu línea de comandos.

### 2. Desactivar el entorno virtual

Cuando termines de trabajar:
```powershell
deactivate
```

## 📊 Archivos del Proyecto

- **`datos.csv`** - Dataset principal con los datos de la Generación Z
- **`graficos.py`** - Genera visualizaciones estadísticas
- **`nube_palabras.py`** - Crea nube de palabras semántica
- **`marco_teorico.py`** - Genera el marco teórico
- **`rag.py`** - Sistema RAG (Retrieval-Augmented Generation)
- **`preparar_datos_csv.py`** - Prepara los datos para el RAG

## 🎯 Cómo ejecutar los scripts

Asegúrate de tener el entorno virtual activado primero, luego:

```powershell
# Generar gráficos estadísticos
python graficos.py

# Generar nube de palabras
python nube_palabras.py

# Preparar datos para RAG
python preparar_datos_csv.py

# Ejecutar el sistema RAG
python rag.py

# Generar marco teórico
python marco_teorico.py
```

## 📦 Dependencias Instaladas

- pandas - Análisis de datos
- matplotlib - Visualizaciones
- seaborn - Gráficos estadísticos
- wordcloud - Nubes de palabras
- langchain - Framework para RAG
- chromadb - Base de datos vectorial
- sentence-transformers - Embeddings
- Y más...

## 💡 Notas

- Los gráficos se guardan en la carpeta `graficos/`
- Los datos procesados para RAG se guardan en `datos/`
- Asegúrate de tener el archivo `datos.csv` en la carpeta principal
