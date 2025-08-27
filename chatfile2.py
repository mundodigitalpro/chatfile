import sqlite3
import re
import nltk
from sentence_transformers import SentenceTransformer, util
import torch

# Descargar ambos recursos
nltk.download('punkt')
nltk.download('punkt_tab')


# Cargar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_text_to_db(filename):
    """
    Carga el contenido de un archivo de texto en la base de datos SQLite, 
    dividiéndolo en oraciones y almacenando cada oración como una entrada en la tabla 'knowledge'.
    """
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()

    # Crear la tabla de conocimiento si no existe
    cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge (
                      id INTEGER PRIMARY KEY,
                      content TEXT)''')

    # Leer el archivo de texto y dividirlo en oraciones
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        sentences = nltk.sent_tokenize(text)
    
    # Insertar cada oración en la base de datos
    cursor.executemany("INSERT INTO knowledge (content) VALUES (?)", [(sentence,) for sentence in sentences])
    
    conn.commit()
    conn.close()
    print(f"Contenido de {filename} cargado en la base de datos.")


def process_question(question):
    """
    Procesa una pregunta en lenguaje natural buscando en la base de datos la respuesta más relevante
    mediante la comparación de embeddings de oraciones.
    """
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()

    # Obtener todas las oraciones de la base de datos
    cursor.execute("SELECT content FROM knowledge")
    knowledge = [row[0] for row in cursor.fetchall()]
    
    # Calcular el embedding de la pregunta y de cada oración
    question_embedding = model.encode(question, convert_to_tensor=True)
    knowledge_embeddings = model.encode(knowledge, convert_to_tensor=True)
    
    # Calcular las similitudes de coseno entre la pregunta y las oraciones
    similarities = util.pytorch_cos_sim(question_embedding, knowledge_embeddings)
    most_similar_index = torch.argmax(similarities).item()
    
    # Obtener la oración más similar
    answer = knowledge[most_similar_index]
    
    conn.close()
    return answer

def chatbot():
    """
    Función principal del chatbot que interactúa con el usuario en la terminal.
    """
    print("Chatbot: Hola, puedes hacerme preguntas sobre el archivo que cargaste. Escribe 'salir' para terminar.")
    while True:
        question = input("Tú: ")
        if question.lower() in ["salir", "adiós"]:
            print("Chatbot: ¡Hasta luego!")
            break
        response = process_question(question)
        print(f"Chatbot: {response}")

# Solicitar al usuario que cargue el archivo al iniciar el programa
if __name__ == "__main__":
    filename = input("Introduce el nombre del archivo de texto (ejemplo: documento.txt): ")
    load_text_to_db(filename)
    chatbot()
