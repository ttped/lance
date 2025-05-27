### Prerequisites

- **Python 3.8+**
- **Node.js**
- **Ollama** installed and running

### 1. Install Ollama

Install Ollama Client: (visit https://ollama.ai for platform-specific instructions)

### 2. Clone and Setup

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=gemma3:4b
OLLAMA_EMBED_MODEL_NAME=nomic-embed-text

# Database Configuration
SQLITE_DB_FILE_PATH=engineering_docs.sqlite
SQLITE_TABLE_NAME=processed_documents
VECTOR_STORE_PATH=./ollama_chroma_db

# API Configuration
API_HOST=localhost
API_PORT=8000
```

### 4. Run the Application

```bash
# Start the FastAPI server
python main.py

# Open your browser to:
# http://localhost:8000
```
### Model Selection

You can use different Ollama models by updating the `.env` file: