from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from pydantic import BaseModel

from document_processor import (
    extract_text_from_txt,
    extract_text_from_pdf,
    split_text_into_chunks
)
from retriever import find_relevant_chunks
from answer_generator import generate_basic_answer

app = FastAPI(title="Admissions RAG Assistant")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DOCUMENT_CHUNKS = []


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Admissions RAG Assistant is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    content = await file.read()

    with open(file_path, "wb") as saved_file:
        saved_file.write(content)

    extracted_text = None
    chunks = []

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        
        extracted_text = extract_text_from_txt(file_path)

    elif suffix == ".pdf":
        extracted_text = extract_text_from_pdf(file_path)

    if extracted_text:
        chunks = split_text_into_chunks(extracted_text)

        DOCUMENT_CHUNKS.clear()

        for index, chunk in enumerate(chunks):
            DOCUMENT_CHUNKS.append({
                "chunk_id": index,
                "filename": file.filename,
                "text": chunk
            })

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "saved_to": str(file_path),
        "text_length": len(extracted_text) if extracted_text else 0,
        "chunks_count": len(chunks),
        "first_chunk": chunks[0] if chunks else None
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    relevant_chunks = find_relevant_chunks(
        question=request.question,
        chunks=DOCUMENT_CHUNKS
    )

    answer = generate_basic_answer(
        question=request.question,
        relevant_chunks=relevant_chunks
    )

    sources = []

    for chunk in relevant_chunks:
        sources.append({
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "score": chunk["score"],
            "preview": chunk["text"][:200] + "..."
        })

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }