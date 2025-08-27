# Chatfile - Semantic Search Chatbot

A Python-based chatbot application that uses semantic search to answer questions based on loaded text documents. The system uses sentence transformers for embedding generation and SQLite for knowledge storage.

## Features

- **Semantic Search**: Uses advanced sentence transformer models for intelligent text matching
- **Two Implementation Options**:
  - `chatfile.py`: High-performance version with precomputed embeddings
  - `chatfile2.py`: Lightweight version with on-the-fly computation
- **SQLite Storage**: Efficient knowledge base management
- **Configurable Similarity**: Adjustable relevance thresholds
- **Multi-language Support**: Works with various text encodings including Spanish content

## Quick Start

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mundodigitalpro/chatfile.git
cd chatfile
```

2. Create and activate virtual environment:
```powershell
# Windows PowerShell
python -m venv env
.\env\Scripts\activate

# Linux/Mac
python -m venv env
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the main chatbot (recommended):
```bash
python chatfile.py
```

Or run the lightweight version:
```bash
python chatfile2.py
```

## Architecture

### Implementation Comparison

| Feature | chatfile.py | chatfile2.py |
|---------|-------------|--------------|
| Model | all-mpnet-base-v2 | all-MiniLM-L6-v2 |
| Storage | Precomputed embeddings | On-the-fly computation |
| Performance | Higher accuracy | Faster startup |
| Memory Usage | Higher | Lower |
| Text Processing | Paragraph-based | Sentence-based |

### Core Components

- **Text Processing**: Automatic chunking and preprocessing of input documents
- **Embedding Generation**: Uses state-of-the-art sentence transformer models
- **Database Layer**: SQLite-based knowledge storage with efficient querying
- **Similarity Search**: Cosine similarity matching with configurable thresholds

## Dependencies

- `sentence-transformers`: Text embedding generation
- `nltk`: Natural language processing utilities
- `scikit-learn`: Machine learning utilities
- `torch`: PyTorch backend for transformers
- `numpy`: Numerical computing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## License

This project is open source and available under the MIT License.