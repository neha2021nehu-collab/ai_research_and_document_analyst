# AI Research Assistant

A production-ready AI Research Assistant with document ingestion, RAG-based Q&A with citations, summarization, and agentic multi-step research capabilities.

## Features

- **Document Ingestion**: Upload PDFs, TXT, Markdown, and DOCX files
- **Smart Chunking & Embeddings**: Automatic text segmentation with Ollama embeddings
- **RAG with Citations**: Query documents with source attribution (document ID + chunk index)
- **Summarization**: Generate concise summaries of uploaded documents
- **Agentic Research**: Autonomous multi-step research on topics using iterative querying
- **Modern Web UI**: Streamlit-based interface for all features
- **REST API**: FastAPI backend with full OpenAPI documentation

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Backend   │────▶│  ChromaDB   │
│  (Streamlit)│     │  (FastAPI)  │     │ (Vector DB) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Ollama    │
                    │   (LLM)     │
                    └─────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU (optional, for Ollama acceleration)

### Using Docker Compose (Recommended)

```bash
# Clone and navigate
cd ai_research_and_document_analyst

# Copy environment template
cp .env.example .env

# Start all services
docker compose up -d

# Pull required Ollama models (first run)
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

Access:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Start services (ChromaDB & Ollama via Docker)
docker compose up -d chromadb ollama

# Pull models
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text

# Run backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend (in another terminal)
streamlit run frontend/app.py --server.port 8501
```

## Configuration

All configuration is managed via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Backend bind address |
| `API_PORT` | `8000` | Backend port |
| `FRONTEND_PORT` | `8501` | Frontend port |
| `CHROMA_HOST` | `chromadb` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `OLLAMA_HOST` | `ollama` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `OLLAMA_MODEL` | `llama3.2` | LLM model for generation |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K` | `5` | Default retrieval count |

See `.env.example` for all options.

## API Endpoints

### Documents
- `POST /api/v1/documents/upload` - Upload and process a document
- `GET /api/v1/documents/{document_id}` - Get document metadata
- `DELETE /api/v1/documents/{document_id}` - Delete a document

### Query & Research
- `POST /api/v1/query` - RAG query with citations
- `POST /api/v1/summarize` - Summarize documents
- `POST /api/v1/research` - Agentic multi-step research

### Health
- `GET /api/v1/health` - Service health check

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push/PR:

1. **Lint** - Ruff code quality checks
2. **Type Check** - MyPy static type analysis
3. **Tests** - pytest with coverage reporting
4. **Docker Build** - Validate Docker images build correctly
5. **Security** - Trivy vulnerability scanning

## Project Structure

```
ai_research_and_document_analyst/
├── backend/
│   └── app/
│       ├── api/          # FastAPI routes
│       ├── core/         # Logging, exceptions
│       ├── models/       # Pydantic models
│       ├── services/     # Business logic (Chroma, Ollama, Document, Query)
│       └── main.py       # FastAPI application
├── frontend/
│   └── app.py            # Streamlit application
├── config/
│   └── settings.py       # Pydantic settings
├── tests/                # Unit tests
├── Dockerfile.backend    # Backend container
├── Dockerfile.frontend   # Frontend container
├── docker-compose.yml    # Service orchestration
├── pyproject.toml        # Dependencies & tool config
└── .env.example          # Environment template
```

## Development

### Code Quality

```bash
# Lint
ruff check .
ruff format .

# Type check
mypy backend frontend config

# Tests
pytest -v --cov=backend --cov=frontend --cov=config
```

### Adding New Models

Edit `backend/app/models/__init__.py` and corresponding API routes in `backend/app/api/routes.py`.

### Adding New Services

1. Create service in `backend/app/services/`
2. Export from `backend/app/services/__init__.py`
3. Use in API routes

## Security

- No secrets in repository (use `.env` file)
- Non-root containers
- CORS configured for specific origins
- Input validation on all endpoints
- Rate limiting ready (configure via env)

## License

MIT License