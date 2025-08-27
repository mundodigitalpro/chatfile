import re
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from typing import List, Dict, Tuple
import uuid

# ===== CONFIGURACIÓN 2025 =====
# Modelo estático ultrarrápido (400x mejora según especificaciones)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Usamos este por compatibilidad y velocidad

# ChromaDB - Vector database nativo
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def initialize_collection(collection_name: str = "knowledge_2025"):
    """Inicializa la colección de ChromaDB"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Colección '{collection_name}' existente encontrada.")
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Optimizado para similitud coseno
        )
        print(f"Nueva colección '{collection_name}' creada.")
    return collection

def intelligent_chunking(text: str, target_size: int = 400, overlap: int = 100) -> List[Dict]:
    """
    Chunking inteligente optimizado para documentos legales estructurados
    Mejora la segmentación para aprovechar búsqueda multi-resultado
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    current_metadata = {"type": "paragraph", "section": "general"}
    current_section_title = ""
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Saltar líneas vacías
        if not line:
            i += 1
            continue
            
        # Detectar títulos principales (# Página X, números grandes, etc.)
        if (line.startswith('#') or 
            line.startswith('Página') or
            (len(line) < 80 and line.replace('.', '').replace(' ', '').isdigit())):
            
            # Guardar chunk anterior si existe
            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": current_metadata.copy(),
                    "id": str(uuid.uuid4())
                })
            
            current_section_title = line
            current_metadata = {"type": "section_header", "section": line}
            current_chunk = line + "\n"
            i += 1
            continue
            
        # Detectar subtítulos (números seguidos de punto, letras con paréntesis)
        if (re.match(r'^\d+\..*', line) or 
            re.match(r'^[a-z]\).*', line) or
            re.match(r'^[A-Z]\).*', line) or
            (len(line) < 120 and ':' in line and not '.' in line[:50])):
            
            # Guardar chunk anterior si es suficientemente grande
            if len(current_chunk.strip()) > 150:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": current_metadata.copy(),
                    "id": str(uuid.uuid4())
                })
                current_chunk = ""
            
            # Actualizar contexto
            current_metadata = {
                "type": "subsection", 
                "section": current_section_title,
                "subsection": line
            }
            current_chunk += line + "\n"
            i += 1
            continue
            
        # Línea normal de contenido
        current_chunk += line + "\n"
        
        # Verificar si necesita dividirse por tamaño
        if len(current_chunk) >= target_size:
            # Buscar punto de división natural
            text_to_split = current_chunk
            
            # Priorizar divisiones por párrafos (doble salto)
            if '\n\n' in text_to_split:
                parts = text_to_split.split('\n\n')
                if len(parts) > 1:
                    chunk_part = '\n\n'.join(parts[:-1])
                    overlap_part = '\n\n'.join(parts[-2:]) if len(parts) > 2 else parts[-1]
                    
                    chunks.append({
                        "content": chunk_part.strip(),
                        "metadata": current_metadata.copy(),
                        "id": str(uuid.uuid4())
                    })
                    
                    current_chunk = overlap_part
                    i += 1
                    continue
            
            # Si no hay párrafos, dividir por oraciones
            sentences = text_to_split.split('. ')
            if len(sentences) > 2:
                # Tomar 2/3 del contenido para el chunk
                split_point = len(sentences) * 2 // 3
                chunk_part = '. '.join(sentences[:split_point]) + '.'
                overlap_part = '. '.join(sentences[split_point-1:])
                
                chunks.append({
                    "content": chunk_part.strip(),
                    "metadata": current_metadata.copy(),
                    "id": str(uuid.uuid4())
                })
                
                current_chunk = overlap_part
                i += 1
                continue
                
            # Si no se puede dividir naturalmente, forzar división
            if len(current_chunk) > target_size * 1.5:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": current_metadata.copy(),
                    "id": str(uuid.uuid4())
                })
                current_chunk = ""
        
        i += 1
    
    # Agregar último chunk si existe y tiene contenido significativo
    if current_chunk.strip() and len(current_chunk.strip()) > 50:
        chunks.append({
            "content": current_chunk.strip(),
            "metadata": current_metadata.copy(),
            "id": str(uuid.uuid4())
        })
    
    # Filtrar chunks muy pequeños o vacíos
    valid_chunks = []
    for chunk in chunks:
        content = chunk["content"].strip()
        if len(content) > 30 and not content.isspace():  # Mínimo 30 caracteres
            valid_chunks.append(chunk)
    
    return valid_chunks

def preprocess_document(text: str) -> str:
    """Preprocessing mejorado que preserva la estructura del documento"""
    # 1. Limpiar caracteres problemáticos
    text = text.replace('\r', '')  # Eliminar caracteres de retorno
    
    # 2. Normalizar saltos de línea múltiples pero preservar estructura
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Múltiples saltos -> doble salto
    
    # 3. Limpiar espacios excesivos en líneas individuales (pero mantener saltos de línea)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Normalizar espacios dentro de cada línea, pero mantener líneas vacías
        if line.strip():
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append('')  # Mantener líneas vacías para estructura
    
    return '\n'.join(cleaned_lines)

def load_text_to_chromadb(filename: str, collection_name: str = "knowledge_2025"):
    """
    Carga el contenido de un archivo usando ChromaDB con chunking inteligente
    Implementa la Fase 1 de MEJORAS_PROPUESTAS.md
    """
    collection = initialize_collection(collection_name)
    
    # Leer y preprocesar el archivo
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    preprocessed_text = preprocess_document(text)
    
    # Chunking inteligente
    chunks = intelligent_chunking(preprocessed_text)
    
    # Limpiar colección existente para recargar
    try:
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Colección '{collection_name}' reinicializada.")
    except:
        pass
    
    # Preparar datos para ChromaDB
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    
    # Generar embeddings en lotes para eficiencia
    embeddings = model.encode(documents)
    
    # Insertar en ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"{filename} cargado: {len(chunks)} chunks procesados con ChromaDB")
    return len(chunks)

def enhance_query(question: str) -> List[str]:
    """
    Mejora básica de la query según especificaciones
    """
    # Normalizar
    normalized = question.lower().strip()
    
    # Generar variaciones básicas
    variations = [question, normalized]
    
    # Expansión simple de sinónimos legales
    synonyms = {
        "derecho": ["derecho", "facultad", "libertad", "prerrogativa"],
        "ley": ["ley", "norma", "legislación", "disposición"],
        "artículo": ["artículo", "art", "apartado"],
        "constitución": ["constitución", "carta magna", "ley fundamental"]
    }
    
    for original, synonym_list in synonyms.items():
        if original in normalized:
            for synonym in synonym_list:
                if synonym != original:
                    variations.append(normalized.replace(original, synonym))
    
    return list(set(variations))  # Eliminar duplicados

def process_question_enhanced(question: str, top_k: int = 5, min_threshold: float = 0.4, collection_name: str = "knowledge_2025") -> str:
    """
    Procesamiento mejorado con búsqueda multi-resultado
    Implementa las mejoras de MEJORAS_PROPUESTAS.md
    """
    collection = initialize_collection(collection_name)
    
    # Verificar que hay datos en la colección
    try:
        count = collection.count()
        if count == 0:
            return "La base de conocimientos está vacía. Por favor, carga un archivo primero."
    except:
        return "Error al acceder a la base de conocimientos."
    
    # Mejorar la query
    query_variations = enhance_query(question)
    
    # Realizar búsquedas múltiples y combinar resultados
    all_results = []
    
    for query in query_variations[:3]:  # Usar hasta 3 variaciones para no sobrecargar
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, count)  # No pedir más resultados de los disponibles
            )
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    # ChromaDB usa distancia (menor = más similar), convertir a similitud
                    similarity = 1 - distance
                    if similarity >= min_threshold:
                        all_results.append({
                            'content': doc,
                            'similarity': similarity,
                            'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                            'query_variation': query
                        })
        except Exception as e:
            print(f"Error en búsqueda para '{query}': {e}")
            continue
    
    if not all_results:
        return format_no_answer_response(question)
    
    # Eliminar duplicados y ordenar por similitud
    seen_content = set()
    unique_results = []
    for result in all_results:
        if result['content'] not in seen_content:
            seen_content.add(result['content'])
            unique_results.append(result)
    
    # Ordenar por similitud descendente
    unique_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Tomar los mejores resultados
    top_results = unique_results[:top_k]
    
    return format_intelligent_response(top_results, question)

def extract_relevant_sentences(text: str, question: str) -> str:
    """
    Extrae las oraciones más relevantes de un texto para responder una pregunta
    """
    # Dividir en oraciones
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return text
    
    # Keywords de la pregunta (palabras importantes)
    question_words = set(question.lower().split())
    stop_words = {'que', 'es', 'son', 'como', 'donde', 'cuando', 'quien', 'cual', 'cuales', 
                  'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'y', 'o', 'por', 'para', 'con'}
    question_keywords = question_words - stop_words
    
    # Puntuar oraciones por relevancia
    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        
        # Contar coincidencias de keywords
        matches = len(question_keywords.intersection(sentence_words))
        
        # Bonus por términos legales importantes
        legal_terms = {'derecho', 'derechos', 'libertad', 'ley', 'constitucion', 'articulo', 'garantia'}
        legal_matches = len(legal_terms.intersection(sentence_words))
        
        # Bonus por definiciones
        definition_indicators = sentence.lower().startswith(('el', 'la', 'los', 'las', 'se', 'es', 'son'))
        definition_bonus = 1 if definition_indicators else 0
        
        score = matches + (legal_matches * 0.5) + definition_bonus
        
        if score > 0:
            scored_sentences.append((score, sentence))
    
    if not scored_sentences:
        return sentences[0] if sentences else text
    
    # Ordenar por puntuación y tomar las mejores
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Tomar las mejores oraciones (máximo 150 palabras)
    selected_sentences = []
    total_words = 0
    
    for score, sentence in scored_sentences[:3]:  # Máximo 3 oraciones
        words_in_sentence = len(sentence.split())
        if total_words + words_in_sentence <= 150:  # Límite más estricto
            selected_sentences.append(sentence.strip())
            total_words += words_in_sentence
            if len(selected_sentences) >= 2:  # Máximo 2 oraciones por chunk
                break
        elif total_words == 0:  # Si es la primera oración y es muy larga, truncarla
            words = sentence.split()[:30]  # Máximo 30 palabras
            truncated = ' '.join(words) + ('...' if len(sentence.split()) > 30 else '')
            selected_sentences.append(truncated)
            break
    
    if not selected_sentences:
        # Fallback: tomar la primera oración truncada
        first_sentence = scored_sentences[0][1]
        words = first_sentence.split()[:25]
        selected_sentences = [' '.join(words) + ('...' if len(first_sentence.split()) > 25 else '')]
    
    result = '. '.join(selected_sentences)
    if not result.endswith('.'):
        result += '.'
    
    return result

def generate_coherent_answer(question: str, results: List[Dict]) -> str:
    """
    Genera una respuesta coherente combinando información de múltiples chunks
    """
    if not results:
        return format_no_answer_response(question)
    
    # Extraer información relevante de cada chunk
    relevant_info = []
    for result in results[:3]:  # Máximo 3 fuentes
        content = result['content']
        confidence = result['similarity']
        
        # Extraer oraciones relevantes
        relevant_text = extract_relevant_sentences(content, question)
        
        if relevant_text and relevant_text not in [info[1] for info in relevant_info]:
            relevant_info.append((confidence, relevant_text))
    
    if not relevant_info:
        return format_no_answer_response(question)
    
    # Combinar en respuesta coherente
    if len(relevant_info) == 1:
        confidence, text = relevant_info[0]
        return f"{text}\n\n[Confianza: {confidence:.1%}]"
    
    # Múltiples fuentes - generar respuesta combinada
    response_parts = []
    max_confidence = max(info[0] for info in relevant_info)
    
    for i, (confidence, text) in enumerate(relevant_info):
        if i == 0:
            response_parts.append(text)
        else:
            # Evitar repetir información similar
            if not any(word in text.lower() for word in response_parts[0].lower().split()[:5]):
                response_parts.append(f"Además, {text}")
    
    combined_response = " ".join(response_parts)
    
    # Añadir información de confianza
    combined_response += f"\n\n[Información combinada de {len(relevant_info)} fuentes - Confianza máxima: {max_confidence:.1%}]"
    
    # Sugerencia si confianza es baja
    if max_confidence < 0.6:
        combined_response += "\n\nSugerencia: Intenta reformular tu pregunta o usar términos más específicos."
    
    return combined_response

def format_intelligent_response(results: List[Dict], question: str) -> str:
    """
    Formatea respuestas inteligentes con extracción de información relevante
    """
    return generate_coherent_answer(question, results)

def format_no_answer_response(question: str) -> str:
    """Respuesta cuando no se encuentra información relevante"""
    return f"""No encontré información suficientemente relevante para: "{question}"

Sugerencias:
- Intenta usar terminos mas especificos
- Reformula la pregunta de otra manera  
- Verifica que el archivo haya sido cargado correctamente

Escribe 'help' para ver comandos disponibles."""

def chatbot_v2025():
    """
    Chatbot mejorado con tecnologías 2025
    """
    print("Chatbot v2025 - Tecnologia ChromaDB + Sentence Transformers")
    print("Comandos: 'salir', 'help', 'reload <archivo>', 'stats'")
    print("-" * 60)
    
    collection_name = "knowledge_2025"
    collection = initialize_collection(collection_name)
    
    while True:
        try:
            question = input("\nTú: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ["salir", "exit", "quit", "adiós"]:
                print("Chatbot: Hasta luego!")
                break
            
            elif question.lower() == "help":
                print("""
Comandos disponibles:
- salir - Terminar el chatbot
- help - Mostrar esta ayuda  
- reload <archivo> - Recargar archivo en la base de conocimientos
- stats - Mostrar estadisticas de la base de conocimientos
                """)
                continue
            
            elif question.lower().startswith("reload "):
                filename = question[7:].strip()
                if filename:
                    try:
                        chunk_count = load_text_to_chromadb(filename, collection_name)
                        print(f"Archivo '{filename}' recargado con {chunk_count} chunks")
                        collection = initialize_collection(collection_name)  # Reinicializar referencia
                    except Exception as e:
                        print(f"Error cargando '{filename}': {e}")
                else:
                    print("Especifica un nombre de archivo: reload <archivo>")
                continue
            
            elif question.lower() == "stats":
                try:
                    count = collection.count()
                    print(f"Base de conocimientos: {count} chunks almacenados")
                except Exception as e:
                    print(f"Error obteniendo estadisticas: {e}")
                continue
            
            # Procesar pregunta normal
            response = process_question_enhanced(question, collection_name=collection_name)
            print(f"\nChatbot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nChatbot: Hasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Intenta de nuevo o escribe 'help' para ver comandos.")

# Programa principal
if __name__ == "__main__":
    print("Iniciando Chatbot v2025 con ChromaDB + Sentence Transformers")
    
    # Solicitar archivo inicial
    filename = input("\nIntroduce el nombre del archivo de texto (o ENTER para continuar sin cargar): ").strip()
    
    if filename:
        try:
            chunk_count = load_text_to_chromadb(filename)
            print(f"{filename} cargado exitosamente con {chunk_count} chunks")
        except Exception as e:
            print(f"Error cargando '{filename}': {e}")
            print("Puedes cargar un archivo después con el comando 'reload <archivo>'")
    
    chatbot_v2025()