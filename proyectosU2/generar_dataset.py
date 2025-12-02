import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
import os
import shutil

# --- CONFIGURACIÓN ADAPTADA ---

# 1. Detectamos dónde está este archivo (Unidad2) para armar la ruta
base_dir = os.path.dirname(os.path.abspath(__file__))
output_final = os.path.join(base_dir, "dataset") 

temp_folder = os.path.join(base_dir, "temp_crops") 
target_size = (64, 64)
cantidad_por_clase = 10000

# Mapeo: Nombre de TUS carpetas -> Nombre en Open Images (Inglés)
clases_map = {
    "gatos": ["Cat"],
    "perros": ["Dog"],
    "tortugas": ["Turtle", "Sea turtle", "Tortoise"], 
    "hormigas": ["Ant"],
    "mariquitas": ["Ladybug"]
}

def procesar_dataset():
    print(f"📂 El dataset se guardará en: {output_final}")
    
    if not os.path.exists(output_final):
        print("❌ Error: No encuentro la carpeta 'dataset'.")
        return

    for carpeta_nombre, clases_oi in clases_map.items():
        print(f"\n==========================================")
        print(f"🚀 Procesando: {carpeta_nombre.upper()}")
        print(f"==========================================")

        dataset = None
        try:
            # 1. Cargar/Descargar dataset
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],
                classes=clases_oi,
                max_samples=cantidad_por_clase,
                only_matching=True,
                shuffle=True
            )
            
            # --- CORRECCIÓN DEL ERROR ---
            # Verificamos qué nombre usó FiftyOne para las cajas
            schema = dataset.get_field_schema()
            campo_detecciones = "detections" # Nombre por defecto
            
            if "ground_truth" in schema:
                campo_detecciones = "ground_truth"
            elif "detections" not in schema:
                # Si no es ninguno de los dos, buscamos el primer campo que sea de detecciones
                for field, type_ in schema.items():
                    if "Detections" in str(type_):
                        campo_detecciones = field
                        break
            
            print(f" -> Usando campo de etiquetas: '{campo_detecciones}'")
            # -----------------------------

            print(f" -> Recortando objetos (quitando fondo)...")
            
            # 2. Recortar usando el nombre de campo correcto
            patches = dataset.to_patches(campo_detecciones)
            
            # Ruta temporal específica para esta clase
            class_temp_dir = os.path.join(temp_folder, carpeta_nombre)
            
            # 3. Exportar recortes crudos
            patches.export(
                export_dir=class_temp_dir,
                dataset_type=fo.types.ImageDirectory,
            )

            # 4. Procesar (Redimensionar 64x64) y Mover
            print(f" -> Redimensionando y guardando...")
            
            class_final_dir = os.path.join(output_final, carpeta_nombre)
            if not os.path.exists(class_final_dir):
                os.makedirs(class_final_dir)

            count = 0
            archivos_temp = os.listdir(class_temp_dir)
            
            for filename in archivos_temp:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(class_temp_dir, filename)
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                            
                            new_name = f"{carpeta_nombre}_auto_{count}.jpg"
                            img_resized.save(os.path.join(class_final_dir, new_name), quality=95)
                            count += 1
                    except Exception:
                        continue
            
            print(f"✅ LISTO {carpeta_nombre}: {count} imágenes añadidas.")

        except Exception as e:
            print(f"⚠️ Error procesando {carpeta_nombre}: {e}")
            # Imprimimos el traceback para entender mejor si falla de nuevo
            import traceback
            traceback.print_exc()

        finally:
            # Limpieza siempre, incluso si falla
            if dataset:
                dataset.delete()
            
            if os.path.exists(temp_folder):
                try:
                    shutil.rmtree(temp_folder)
                except:
                    pass # A veces Windows bloquea el borrado inmediato

if __name__ == "__main__":
    procesar_dataset()
    print("\n🎉 ¡PROCESO TERMINADO! Revisa tu carpeta 'dataset'.")