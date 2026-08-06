import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import app_settings

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from answer_generator import generate_basic_answer
from api_models import QuestionRequest, ResetRequest
from app_settings import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_READ_CHUNK_SIZE,
)
from conversation_service import (
    DocumentSelectionConflict,
    SystemDocumentUnavailable,
    TelegramIdentityRequired,
    get_system_document_state,
    is_system_document_configured,
    reset_conversation,
    resolve_conversation,
    synchronize_system_document_conversations,
)
from database import (
    ConversationIdentityMismatch,
    database_is_ready,
    get_document,
    initialize_database,
    insert_document_with_chunks,
    load_document_chunks,
    record_unanswered_question,
)
from document_processor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    parse_faq_entries,
    split_text_into_chunks,
)
from embedding_model import get_embedding_model, get_embedding_model_name
from embedding_retriever import find_relevant_chunks_semantic
from llm_answer_generator import PROVIDER_UNAVAILABLE
from logging_config import configure_logging
from rag_service import (
    SYSTEM_DOCUMENT_UNAVAILABLE,
    SYSTEM_DOCUMENT_UNAVAILABLE_ANSWER,
    answer_conversation_question,
    build_sources,
    invalidate_document_cache,
    safe_conversation_label,
)
from retriever import find_relevant_chunks
from retrieval_settings import (
    SEMANTIC_FALLBACK_SCORE_THRESHOLD,
    SEMANTIC_SCORE_THRESHOLD,
    SEMANTIC_TOP_K,
)


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    if not application.state.embedding_ready:
        try:
            model = get_embedding_model()
            model.encode(["embedding warmup"], normalize_embeddings=True)
            application.state.embedding_ready = True
            logger.info("Embedding model warmup completed.")
        except Exception as error:
            logger.error(
                "Embedding model warmup failed category=%s.",
                type(error).__name__,
            )
    yield


app = FastAPI(title="Admissions RAG Assistant", lifespan=lifespan)
app.state.embedding_ready = False
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
initialize_database()
synchronize_system_document_conversations()

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def _provider_configuration_ready() -> bool:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))
    if provider == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_MODEL"))
    return False


def _resolve_request_document(request: QuestionRequest) -> tuple[dict, list[dict]]:
    conversation = resolve_conversation(
        conversation_id=request.conversation_id,
        external_chat_id=request.external_chat_id,
        external_user_id=request.external_user_id,
        requested_document_id=request.document_id,
    )
    document_id = conversation["active_document_id"]
    if document_id is None:
        return conversation, []
    return conversation, load_document_chunks(document_id)


def _build_document_chunks(
    extracted_text: str,
    filename: str,
) -> tuple[str, list[dict]]:
    faq_entries = parse_faq_entries(extracted_text)
    if faq_entries:
        document_type = "faq"
        chunks = [
            {
                "chunk_id": entry["faq_id"],
                "faq_id": entry["faq_id"],
                "question": entry["question"],
                "answer": entry["answer"],
                "text_for_retrieval": entry["text_for_retrieval"],
                "text": entry["text"],
            }
            for entry in faq_entries
        ]
    else:
        document_type = "standard"
        chunks = [
            {
                "chunk_id": index,
                "text_for_retrieval": text,
                "text": text,
            }
            for index, text in enumerate(split_text_into_chunks(extracted_text))
        ]
    embeddings = get_embedding_model().encode(
        [chunk["text_for_retrieval"] for chunk in chunks],
        normalize_embeddings=True,
    )
    for chunk, embedding in zip(chunks, embeddings):
        chunk["filename"] = filename
        chunk["embedding"] = embedding
    return document_type, chunks


def _record_legacy_unanswered(question: str) -> None:
    try:
        record_unanswered_question(
            question=question,
            standalone_question=question,
            reason="no_relevant_chunks",
        )
    except Exception:
        logger.error(
            "Failed to record an unanswered question without logging its content."
        )


@app.get("/")
def root():
    return {"message": "Admissions RAG Assistant is running"}


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": app_settings.read_bool("DEMO_MODE", False)}


@app.get("/ready")
def ready():
    database_ready = database_is_ready()
    embedding_ready = app.state.embedding_ready
    provider_configured = _provider_configuration_ready()
    system_document_configured = is_system_document_configured()
    system_document_available = False
    if database_ready:
        system_document_available = get_system_document_state().document is not None
    status_code = (
        200
        if (
            database_ready
            and embedding_ready
            and provider_configured
            and system_document_configured
            and system_document_available
        )
        else 503
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if status_code == 200 else "not_ready",
            "database": "ok" if database_ready else "unavailable",
            "embedding": "ok" if embedding_ready else "unavailable",
            "provider_configured": provider_configured,
            "system_document_configured": system_document_configured,
            "system_document_available": system_document_available,
            "demo_mode": app_settings.read_bool("DEMO_MODE", False),
        },
    )


@app.exception_handler(DocumentSelectionConflict)
def document_selection_conflict_handler(_request, error):
    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.exception_handler(TelegramIdentityRequired)
def telegram_identity_required_handler(_request, _error):
    return JSONResponse(
        status_code=400,
        content={"detail": "Telegram external_chat_id is required."},
    )


@app.exception_handler(ConversationIdentityMismatch)
def conversation_identity_mismatch_handler(_request, _error):
    return JSONResponse(
        status_code=403,
        content={"detail": "Conversation identity does not match."},
    )


@app.exception_handler(SystemDocumentUnavailable)
def system_document_unavailable_handler(_request, error):
    content = {
        "status": SYSTEM_DOCUMENT_UNAVAILABLE,
        "answer": SYSTEM_DOCUMENT_UNAVAILABLE_ANSWER,
        "sources": [],
    }
    if error.conversation is not None:
        content["conversation_id"] = error.conversation["id"]
    return JSONResponse(status_code=503, content=content)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    external_chat_id: str | None = Form(default=None),
    external_user_id: str | None = Form(default=None),
):
    if not file.filename:
        raise HTTPException(400, "The uploaded file must have a filename.")
    original_filename = Path(file.filename).name
    suffix = Path(original_filename).suffix.casefold()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, "Unsupported file type. Please upload only TXT or PDF files."
        )

    stored_filename = f"{uuid.uuid4().hex}{suffix}"
    file_path = UPLOAD_DIR / stored_filename
    size = 0
    header = b""
    try:
        with open(file_path, "xb") as output:
            while chunk := await file.read(UPLOAD_READ_CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        413,
                        f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit.",
                    )
                if len(header) < 8:
                    header += chunk[: 8 - len(header)]
                output.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        if suffix == ".pdf" and not header.startswith(b"%PDF-"):
            raise HTTPException(400, "The uploaded file is not a valid PDF.")
        if suffix == ".txt" and b"\x00" in header:
            raise HTTPException(400, "The uploaded TXT file is not valid text.")

        try:
            extracted_text = (
                extract_text_from_txt(file_path)
                if suffix == ".txt"
                else extract_text_from_pdf(file_path)
            )
        except Exception as error:
            raise HTTPException(400, "The uploaded file could not be read.") from error
        if not extracted_text.strip():
            raise HTTPException(
                400,
                "No readable text was found. The document may be scanned or empty.",
            )

        document_type, chunks = _build_document_chunks(
            extracted_text, original_filename
        )
        system_document_configured = get_system_document_state().configured
        conversation = None
        if not system_document_configured and external_chat_id:
            conversation = resolve_conversation(
                conversation_id=conversation_id,
                external_chat_id=external_chat_id,
                external_user_id=external_user_id,
            )
        document_id = insert_document_with_chunks(
            original_filename,
            stored_filename,
            document_type,
            get_embedding_model_name(),
            chunks,
            activate_conversation_id=(
                conversation["id"] if conversation is not None else None
            ),
        )
        invalidate_document_cache(document_id)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        file_path.unlink(missing_ok=True)
        logger.exception("Document processing failed without exposing file contents.")
        raise HTTPException(500, "The document could not be processed.") from error
    finally:
        await file.close()

    return {
        "document_id": document_id,
        "conversation_id": conversation["id"] if conversation else None,
        "active_document_id": (
            conversation["active_document_id"] if conversation else None
        ),
        "filename": original_filename,
        "document_type": document_type,
        "size_bytes": size,
        "text_length": len(extracted_text),
        "chunks_count": len(chunks),
        **({"entries_count": len(chunks)} if document_type == "faq" else {}),
    }


@app.post("/chat")
def chat(request: QuestionRequest):
    request_started_at = time.perf_counter()
    try:
        response = answer_conversation_question(
            question=request.question.strip(),
            conversation_id=request.conversation_id,
            external_chat_id=request.external_chat_id,
            external_user_id=request.external_user_id,
            document_id=request.document_id,
        )
    except TelegramIdentityRequired as error:
        raise HTTPException(400, str(error)) from error
    except ConversationIdentityMismatch as error:
        raise HTTPException(403, "Conversation identity does not match.") from error
    except DocumentSelectionConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    logger.info(
        "chat_request request_id=%s conversation=%s status=%s total_ms=%.2f",
        uuid.uuid4().hex,
        safe_conversation_label(response["conversation_id"]),
        response["status"],
        (time.perf_counter() - request_started_at) * 1000,
    )
    if response["status"] in {PROVIDER_UNAVAILABLE, SYSTEM_DOCUMENT_UNAVAILABLE}:
        return JSONResponse(status_code=503, content=response)
    return response


@app.post("/ask-llm")
def ask_question_llm(request: QuestionRequest):
    return chat(request)


@app.post("/conversation/reset")
def reset(request: ResetRequest):
    return reset_conversation(
        conversation_id=request.conversation_id,
        external_chat_id=request.external_chat_id,
        external_user_id=request.external_user_id,
    )


@app.get("/conversation/status")
def conversation_status(
    conversation_id: str | None = None,
    external_chat_id: str | None = None,
    external_user_id: str | None = None,
):
    conversation = resolve_conversation(
        conversation_id=conversation_id,
        external_chat_id=external_chat_id,
        external_user_id=external_user_id,
    )
    document = (
        get_document(conversation["active_document_id"])
        if conversation["active_document_id"]
        else None
    )
    return {
        "status": "ok",
        "conversation_id": conversation["id"],
        "active_document_id": document["id"] if document else None,
        "active_document_filename": document["filename"] if document else None,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    _, chunks = _resolve_request_document(request)
    relevant = find_relevant_chunks(request.question, chunks)
    if not relevant:
        _record_legacy_unanswered(request.question)
    return {
        "question": request.question,
        "answer": generate_basic_answer(request.question, relevant),
        "sources": build_sources(relevant),
    }


@app.post("/ask-semantic")
def ask_question_semantic(request: QuestionRequest):
    _, chunks = _resolve_request_document(request)
    relevant = find_relevant_chunks_semantic(
        request.question,
        chunks,
        top_k=SEMANTIC_TOP_K,
        min_score=SEMANTIC_SCORE_THRESHOLD,
        fallback_score_threshold=SEMANTIC_FALLBACK_SCORE_THRESHOLD,
    )
    if not relevant:
        _record_legacy_unanswered(request.question)
    return {
        "question": request.question,
        "answer": generate_basic_answer(request.question, relevant),
        "sources": build_sources(relevant),
    }
