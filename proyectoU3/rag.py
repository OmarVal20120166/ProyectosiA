import os
import time

# --- IMPORTACIONES DEL MODO DIRECTO (PLAN B) ---
# Estas son las que te funcionaron bien
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    # Limpiar pantalla inicial
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("==================================================")
    print("   🤖 SISTEMA RAG - PROYECTO 3 (GEN Z & FILOSOFÍA)")
    print("==================================================")
    
    # 1. CARGAR DATOS
    print("📂 1. Cargando base de conocimientos...")
    if not os.path.exists('./datos'):
        print("❌ ERROR: No existe la carpeta 'datos'. Ejecuta primero 'preparar_datos_csv.py'")
        return

    loader = DirectoryLoader('./datos', glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"   ✅ {len(documents)} documentos cargados.")

    # 2. PREPARAR TEXTO
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 3. CREAR BASE DE DATOS (EMBEDDINGS)
    print("🧠 2. Inicializando cerebro vectorial (esto es rápido si ya descargó)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4}) 

    # 4. CONECTAR LLM
    print("🦙 3. Conectando con Ollama (Llama 3.2)...")
    llm = ChatOllama(model="llama3.2", temperature=0.3, keep_alive="1h")

    # 5. CREAR LA TUBERÍA (CHAIN)
    template = """Eres un experto investigador en sociología digital y filosofía contemporánea.
    Responde a la pregunta basándote EXCLUSIVAMENTE en el siguiente contexto extraído de redes sociales.
    
    Si la respuesta no está en el contexto, di "No tengo información suficiente en la base de datos".
    Cita ejemplos si el contexto los tiene.
    
    Contexto:
    {context}
    
    Pregunta del usuario: {question}
    
    Respuesta útil y fundamentada:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ ¡SISTEMA LISTO! Ya puedes chatear.")
    print("   (Escribe 'salir' o 'exit' para terminar el programa)\n")

    # --- BUCLE DE CHAT INFINITO ---
    while True:
        try:
            # Input del usuario
            pregunta = input("\n👤 Tú: ")
            
            # Condición de salida
            if pregunta.lower() in ['salir', 'exit', 'adios', 'bye']:
                print("\n👋 ¡Hasta luego! Cerrando sistema.")
                break
            
            if not pregunta.strip():
                continue

            print("🤖 IA: Pensando...", end="\r") # Efecto visual simple
            
            # Generar respuesta
            inicio = time.time()
            respuesta = rag_chain.invoke(pregunta)
            tiempo = time.time() - inicio
            
            # Borrar el "Pensando..." y mostrar respuesta
            print(f"🤖 IA ({tiempo:.1f}s): {respuesta}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Salida forzada.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()