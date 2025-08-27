# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based chatbot application that uses semantic search to answer questions based on loaded text documents. The system uses sentence transformers for embedding generation and SQLite for knowledge storage.

## Architecture

### Core Components

1. **Text Processing & Storage** (`chatfile.py`, `chatfile2.py`)
   - Loads text files and processes them into searchable knowledge bases
   - Two versions exist with different processing approaches:
     - `chatfile.py`: Uses paragraph-based chunking with precomputed embeddings stored as BLOB
     - `chatfile2.py`: Uses sentence-based chunking with on-the-fly embedding computation
   
2. **Database Layer** (`knowledge_base.db`)
   - SQLite database storing processed text content
   - Schema varies between implementations (with/without embedding storage)

3. **Semantic Search Engine**
   - Uses SentenceTransformer models for text embedding
   - Implements cosine similarity for relevance matching
   - Configurable similarity threshold for answer quality control

### Key Technical Differences Between Versions

- **chatfile.py**: Uses `all-mpnet-base-v2` model, stores embeddings in database, paragraph chunking
- **chatfile2.py**: Uses `all-MiniLM-L6-v2` model, computes embeddings on-the-fly, sentence chunking

## Development Commands

### Environment Setup
```powershell
# Create virtual environment (if not exists)
python -m venv env

# Activate virtual environment (PowerShell)
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If the above fails, install packages individually:
pip install sentence-transformers
pip install nltk
pip install scikit-learn
```

**Note**: If you encounter path issues with the virtual environment, delete the `env` folder and recreate it from scratch.

### Running the Application
```bash
# Chatbot principal (mejor calidad, embeddings precomputados)
python chatfile.py

# Chatbot alternativo (más rápido, menor memoria)
python chatfile2.py
```

### Database Management
The SQLite database (`knowledge_base.db`) is automatically managed by the application. To inspect:
```bash
sqlite3 knowledge_base.db
.schema knowledge
.quit
```

## Dependencies

Core libraries (from `requirements.txt`):
- `sentence-transformers`: For text embedding generation
- `nltk`: Natural language processing utilities
- `scikit-learn`: Machine learning utilities
- Standard libraries: `sqlite3`, `torch`, `numpy`

## Development Notes

### Model Selection
- Consider the trade-off between model performance and storage efficiency when choosing between the two implementations
- `all-mpnet-base-v2` offers better performance but requires more storage
- `all-MiniLM-L6-v2` is faster and more lightweight

### Text Processing
- Both versions handle UTF-8 encoded text files
- Paragraph chunking may preserve more context but create fewer search units
- Sentence chunking provides more granular search but may lose contextual relationships

### Performance Considerations
- Precomputed embeddings (`chatfile.py`) offer faster query response but require more storage
- On-the-fly computation (`chatfile2.py`) uses less storage but has higher query latency
- Consider batch processing for large document sets

## Data Files

The repository includes `temario.md` as sample content - a Spanish legal document about fundamental rights and constitutional law. This demonstrates the system's capability with non-English content and structured legal text.