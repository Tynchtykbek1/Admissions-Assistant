import logging
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from document_processor import (
    extract_text_from_txt,
    extract_text_from_pdf,
    parse_faq_entries,
    split_text_into_chunks
)
from retriever import find_relevant_chunks
from answer_generator import generate_basic_answer
from embedding_model import get_embedding_model, get_embedding_model_name
from embedding_retriever import find_relevant_chunks_semantic, normalize_retrieval_query
from llm_answer_generator import PROVIDER_UNAVAILABLE, generate_llm_answer
from logging_config import configure_logging, safe_log_text
from database import (
    initialize_database,
    insert_document,
    insert_chunk,
    load_latest_document
)
from retrieval_settings import (
    SEMANTIC_TOP_K,
    SEMANTIC_SCORE_THRESHOLD,
    LLM_MIN_CONTEXT_CHUNKS
)


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Admissions RAG Assistant")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

initialize_database()
DOCUMENT_CHUNKS = load_latest_document()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


def build_sources(relevant_chunks: list[dict]) -> list[dict]:
    sources = []

    for chunk in relevant_chunks:
        source = {
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "score": chunk["score"],
            "preview": chunk["text"][:200]
        }

        for diagnostic in ("faq_match_type", "faq_match_boost", "final_score"):
            if diagnostic in chunk:
                source[diagnostic] = chunk[diagnostic]

        if "faq_id" in chunk:
            source["faq_id"] = chunk["faq_id"]

        sources.append(source)

    return sources


def build_answer_response(question: str, relevant_chunks: list[dict]) -> dict:
    answer = generate_basic_answer(
        question=question,
        relevant_chunks=relevant_chunks
    )

    return {
        "question": question,
        "answer": answer,
        "sources": build_sources(relevant_chunks)
    }


def build_llm_answer_response(question: str, relevant_chunks: list[dict]) -> dict:
    result = generate_llm_answer(
        question=question,
        relevant_chunks=relevant_chunks
    )

    return {
        "question": question,
        "answer": result.answer,
        "sources": build_sources(relevant_chunks),
        "status": result.status,
        "provider": result.provider,
        "provider_duration_ms": round(result.provider_duration_ms, 2),
    }


@app.get("/")
def root():
    return {"message": "Admissions RAG Assistant is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui")
def user_interface():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="The uploaded file must have a filename.")

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    suffix = file_path.suffix.lower()

    allowed_extensions = {".txt", ".pdf"}

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload only .txt or .pdf files."
        )

    content = await file.read()

    with open(file_path, "wb") as saved_file:
        saved_file.write(content)

    try:
        if suffix == ".txt":
            extracted_text = extract_text_from_txt(file_path)
        else:
            extracted_text = extract_text_from_pdf(file_path)
    except Exception as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read."
        ) from error

    if not extracted_text:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the document. The file may be scanned or empty."
        )

    faq_entries = parse_faq_entries(extracted_text)

    if faq_entries:
        document_type = "faq"
        document_chunks = [
            {
                "chunk_id": entry["faq_id"],
                "faq_id": entry["faq_id"],
                "question": entry["question"],
                "answer": entry["answer"],
                "text_for_retrieval": entry["text_for_retrieval"],
                "text": entry["text"]
            }
            for entry in faq_entries
        ]
    else:
        document_type = "standard"
        chunks = split_text_into_chunks(extracted_text)
        document_chunks = [
            {
                "chunk_id": index,
                "text_for_retrieval": chunk,
                "text": chunk
            }
            for index, chunk in enumerate(chunks)
        ]

    model = get_embedding_model()
    chunk_embeddings = model.encode(
        [chunk["text_for_retrieval"] for chunk in document_chunks],
        normalize_embeddings=True
    )

    DOCUMENT_CHUNKS.clear()

    for chunk, embedding in zip(document_chunks, chunk_embeddings):
        chunk["filename"] = safe_filename
        chunk["embedding"] = embedding
        DOCUMENT_CHUNKS.append(chunk)

    document_id = insert_document(
        safe_filename,
        document_type,
        get_embedding_model_name()
    )

    for chunk in DOCUMENT_CHUNKS:
        insert_chunk(document_id, chunk)

    response = {
        "filename": safe_filename,
        "document_type": document_type,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "saved_to": str(file_path),
        "text_length": len(extracted_text),
        "chunks_count": len(document_chunks),
        "first_chunk": document_chunks[0]["text"] if document_chunks else None
    }

    if document_type == "faq":
        response["entries_count"] = len(document_chunks)

    return response


@app.post("/ask")
def ask_question(request: QuestionRequest):
    relevant_chunks = find_relevant_chunks(
        question=request.question,
        chunks=DOCUMENT_CHUNKS
    )

    return build_answer_response(request.question, relevant_chunks)

@app.post("/ask-semantic")
def ask_question_semantic(request: QuestionRequest):
    relevant_chunks = find_relevant_chunks_semantic(
        question=request.question,
        chunks=DOCUMENT_CHUNKS,
        top_k=SEMANTIC_TOP_K,
        min_score=max(SEMANTIC_SCORE_THRESHOLD, 0.30)
    )

    return build_answer_response(request.question, relevant_chunks)


@app.post("/ask-llm")
def ask_question_llm(request: QuestionRequest):
    request_id = uuid.uuid4().hex
    request_started_at = time.perf_counter()
    retrieval_started_at = time.perf_counter()
    normalized_query = normalize_retrieval_query(request.question)
    relevant_chunks = find_relevant_chunks_semantic(
        question=request.question,
        chunks=DOCUMENT_CHUNKS,
        top_k=SEMANTIC_TOP_K,
        min_score=SEMANTIC_SCORE_THRESHOLD,
        min_context_chunks=LLM_MIN_CONTEXT_CHUNKS
    )
    retrieval_duration_ms = (time.perf_counter() - retrieval_started_at) * 1000
    response = build_llm_answer_response(request.question, relevant_chunks)
    total_duration_ms = (time.perf_counter() - request_started_at) * 1000

    logger.info(
        "ask_llm request_id=%s query=%r normalized_query=%r context_count=%d "
        "chunk_ids=%s faq_ids=%s top_scores=%s provider=%s result=%s "
        "retrieval_ms=%.2f provider_ms=%.2f total_ms=%.2f",
        request_id,
        safe_log_text(request.question),
        safe_log_text(normalized_query),
        len(relevant_chunks),
        [chunk["chunk_id"] for chunk in relevant_chunks],
        [chunk.get("faq_id") for chunk in relevant_chunks if chunk.get("faq_id") is not None],
        [round(chunk["score"], 3) for chunk in relevant_chunks],
        response["provider"],
        response["status"],
        retrieval_duration_ms,
        response["provider_duration_ms"],
        total_duration_ms,
    )

    if response["status"] == PROVIDER_UNAVAILABLE:
        return JSONResponse(status_code=503, content=response)
    return response
