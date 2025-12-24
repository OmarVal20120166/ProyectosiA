import os
import re
from PIL import Image
from tqdm import tqdm

# Configuración
dataset_origen = r"A:\repositorios github\IA-Proyectos\proyectos\Unidad2\dataset_limpio"
dataset_destino = r"A:\repositorios github\IA-Proyectos\proyectos\Unidad2\dataset_28x28"
nuevo_tamaño = (28, 28)

# Extensiones de imagen válidas
extensiones_imagen = re.compile(r"\.(jpg|jpeg|png|bmp|tiff|webp)$", re.IGNORECASE)

def redimensionar_dataset():
    """
    Redimensiona todas las imágenes del dataset a 28x28 píxeles
    manteniendo la estructura de carpetas por categorías.
    """

    # Crear directorio destino si no existe
    if not os.path.exists(dataset_destino):
        os.makedirs(dataset_destino)
        print(f"Directorio creado: {dataset_destino}")

    # Obtener todas las categorías (subcarpetas)
    categorias = [d for d in os.listdir(dataset_origen)
                  if os.path.isdir(os.path.join(dataset_origen, d))]

    print(f"\nCategorías encontradas: {categorias}")
    print(f"Redimensionando imágenes a {nuevo_tamaño}...\n")

    total_imagenes = 0
    imagenes_procesadas = 0
    imagenes_error = 0

    # Procesar cada categoría
    for categoria in categorias:
        carpeta_origen = os.path.join(dataset_origen, categoria)
        carpeta_destino = os.path.join(dataset_destino, categoria)

        # Crear carpeta de destino para esta categoría
        if not os.path.exists(carpeta_destino):
            os.makedirs(carpeta_destino)

        # Obtener lista de archivos de imagen
        archivos = [f for f in os.listdir(carpeta_origen)
                   if extensiones_imagen.search(f)]

        total_imagenes += len(archivos)

        print(f"Procesando categoría: {categoria} ({len(archivos)} imágenes)")

        # Procesar cada imagen con barra de progreso
        for archivo in tqdm(archivos, desc=f"  {categoria}", ncols=80):
            try:
                ruta_origen = os.path.join(carpeta_origen, archivo)
                ruta_destino = os.path.join(carpeta_destino, archivo)

                # Abrir imagen
                img = Image.open(ruta_origen)

                # Convertir a RGB si es necesario (para manejar imágenes con transparencia)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Redimensionar usando LANCZOS para mejor calidad
                img_redimensionada = img.resize(nuevo_tamaño, Image.Resampling.LANCZOS)

                # Guardar imagen redimensionada
                img_redimensionada.save(ruta_destino, quality=95)

                imagenes_procesadas += 1

            except Exception as e:
                print(f"\n  Error procesando {archivo}: {e}")
                imagenes_error += 1

        print()

    # Resumen final
    print("=" * 60)
    print("RESUMEN:")
    print(f"  Total de imágenes encontradas: {total_imagenes}")
    print(f"  Imágenes procesadas exitosamente: {imagenes_procesadas}")
    print(f"  Imágenes con error: {imagenes_error}")
    print(f"  Dataset redimensionado guardado en: {dataset_destino}")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("REDIMENSIONAR DATASET A 28x28")
    print("=" * 60)
    print(f"Origen: {dataset_origen}")
    print(f"Destino: {dataset_destino}")
    print(f"Nuevo tamaño: {nuevo_tamaño}")
    print("=" * 60)

    redimensionar_dataset()

    print("\n¡Proceso completado!")
