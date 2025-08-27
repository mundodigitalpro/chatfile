import sqlite3
import re
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Cargar modelo de embeddings
model = SentenceTransformer('all-mpnet-base-v2')

def load_text_to_db(filename):
    """
    Carga el contenido de un archivo de texto en la base de datos SQLite,
    dividiéndolo en párrafos y almacenando los embeddings de cada párrafo para respuestas rápidas.
    """
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()

    # Eliminar la tabla si ya existe para crearla nuevamente con la columna 'embedding'
    cursor.execute('DROP TABLE IF EXISTS knowledge')

    # Crear la tabla de conocimiento con una columna para almacenar embeddings
    cursor.execute('''CREATE TABLE knowledge (
                      id INTEGER PRIMARY KEY,
                      content TEXT,
                      embedding BLOB)''')

    # Leer el archivo de texto y dividirlo en párrafos
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        paragraphs = text.split('\n\n')  # Dividir por párrafos (asumiendo doble salto de línea)
    
    # Insertar cada párrafo y su embedding en la base de datos
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:  # Evitar párrafos vacíos
            # Calcular embedding y convertirlo a un formato binario para almacenamiento
            embedding = model.encode(paragraph)
            embedding_blob = embedding.tobytes()
            cursor.execute("INSERT INTO knowledge (content, embedding) VALUES (?, ?)", (paragraph, embedding_blob))
    
    conn.commit()
    conn.close()
    print(f"Contenido de {filename} cargado en la base de datos con embeddings precomputados.")


def process_question(question, similarity_threshold=0.7):
    """
    Procesa una pregunta en lenguaje natural buscando en la base de datos la respuesta más relevante
    mediante la comparación de embeddings de párrafos ya almacenados.
    """
    conn = sqlite3.connect('knowledge_base.db')
    cursor = conn.cursor()

    # Obtener todos los párrafos y embeddings de la base de datos
    cursor.execute("SELECT content, embedding FROM knowledge")
    knowledge_data = cursor.fetchall()

    # Separar el contenido y los embeddings
    knowledge = [row[0] for row in knowledge_data]
    knowledge_embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in knowledge_data])

    # Calcular el embedding de la pregunta
    question_embedding = model.encode(question)

    # Convertir los embeddings a tensores de Torch
    question_embedding_tensor = torch.tensor(question_embedding)
    knowledge_embeddings_tensor = torch.tensor(knowledge_embeddings)

    # Calcular las similitudes de coseno entre la pregunta y los párrafos
    similarities = util.pytorch_cos_sim(question_embedding_tensor, knowledge_embeddings_tensor)
    most_similar_index = torch.argmax(similarities).item()
    max_similarity_score = similarities[0][most_similar_index].item()
    
    # Evaluar si la similitud alcanza el umbral establecido
    if max_similarity_score < similarity_threshold:
        answer = "Lo siento, no encontré una respuesta precisa para esa pregunta."
    else:
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
