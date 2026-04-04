## Stack

- **n8n** — workflow orchestration and webhook handling
- **FastAPI** — Python backend with /ingest and /query endpoints
- **PyMuPDF** — PDF text extraction
- **OpenAI** — text-embedding-3-small + GPT-4o-mini
- **Qdrant** — vector database for semantic search
- **Docker** — containerized n8n and Qdrant

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Upload PDF, chunk, embed, store in Qdrant |
| POST | `/query` | Ask question, retrieve context, get GPT answer |

## Setup
```bash
# Install dependencies
pip install fastapi uvicorn pymupdf openai qdrant-client python-dotenv python-multipart

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start FastAPI
uvicorn main:app --reload

# Import n8n workflow
# Open n8n → Import workflow JSON
```

## Related Projects

- [TaxBot](https://github.com/shaheersalal/Tax-Bot) — Production RAG chatbot for UK tax firm
