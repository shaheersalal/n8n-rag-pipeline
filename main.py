from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import tempfile
import os
from ingest import PDFIngester

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post('/ingest')
def ingest(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    ingester = PDFIngester(tmp_path, "legal_docs")
    text = ingester.load_text()
    chunks = ingester.chunk_text(text)
    ingester.embed_and_store(chunks)
    
    os.unlink(tmp_path)
    
    return {
        "message": "ingested",
        "filename": file.filename,
        "chunks": len(chunks)
    }

@app.post('/query')
def query(request: QueryRequest):
    ingester = PDFIngester("", "legal_docs")
    answer = ingester.query_collection(request.question)
    return {"answer": answer}