import hashlib
import logging
import time
from collections import OrderedDict
from threading import Lock

from app_settings import CHAT_HISTORY_LIMIT, DOCUMENT_CACHE_SIZE
from conversation_service import resolve_conversation
from database import (
    add_message,
    get_document,
    get_recent_messages,
    load_document_chunks,
    update_active_document,
)
from embedding_retriever import find_relevant_chunks_semantic
from llm_answer_generator import (
    INSUFFICIENT_DOCUMENT_INFORMATION,
    INSUFFICIENT_INFORMATION_ANSWER,
    PARTIAL_INFORMATION,
    PROVIDER_UNAVAILABLE,
    SUCCESS,
    generate_llm_answer,
)
from question_rewriter import rewrite_question
from retrieval_settings import (
    SEMANTIC_FALLBACK_SCORE_THRESHOLD,
    SEMANTIC_SCORE_THRESHOLD,
    SEMANTIC_TOP_K,
)


logger = logging.getLogger(__name__)

NO_ACTIVE_DOCUMENT_ANSWER = (
    "No document is active for this conversation. Upload or select a document first."
)

_document_cache: OrderedDict[int, list[dict]] = OrderedDict()
_document_cache_lock = Lock()


def invalidate_document_cache(document_id: int | None = None) -> None:
    with _document_cache_lock:
        if document_id is None:
            _document_cache.clear()
        else:
            _document_cache.pop(document_id, None)


def get_cached_document_chunks(document_id: int) -> list[dict]:
    with _document_cache_lock:
        cached = _document_cache.get(document_id)
        if cached is not None:
            _document_cache.move_to_end(document_id)
            return cached
    chunks = load_document_chunks(document_id)
    with _document_cache_lock:
        _document_cache[document_id] = chunks
        _document_cache.move_to_end(document_id)
        while len(_document_cache) > DOCUMENT_CACHE_SIZE:
            _document_cache.popitem(last=False)
    return chunks


def build_sources(relevant_chunks: list[dict]) -> list[dict]:
    sources = []
    seen = set()
    for chunk in relevant_chunks:
        identity = (
            chunk["filename"],
            chunk.get("faq_id"),
            chunk["chunk_id"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        source = {
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "score": round(float(chunk["score"]), 6),
            "preview": chunk["text"][:200],
        }
        if "faq_id" in chunk:
            source["faq_id"] = chunk["faq_id"]
        sources.append(source)
    return sources


def safe_conversation_label(conversation_id: str) -> str:
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:12]


def _base_response(
    *,
    question: str,
    standalone_question: str,
    answer: str,
    status: str,
    conversation_id: str,
    sources: list[dict],
    provider: str,
    provider_duration_ms: float,
    retrieval_duration_ms: float,
    document: dict | None,
) -> dict:
    return {
        "question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "status": status,
        "sources": sources,
        "conversation_id": conversation_id,
        "document_id": document["id"] if document else None,
        "document_filename": document["filename"] if document else None,
        "provider": provider,
        "provider_duration_ms": round(provider_duration_ms, 2),
        "retrieval_duration_ms": round(retrieval_duration_ms, 2),
    }


def answer_conversation_question(
    *,
    question: str,
    conversation_id: str | None = None,
    external_chat_id: str | None = None,
    external_user_id: str | None = None,
    document_id: int | None = None,
    allow_latest_document_default: bool = False,
) -> dict:
    conversation = resolve_conversation(
        conversation_id=conversation_id,
        external_chat_id=external_chat_id,
        external_user_id=external_user_id,
        allow_latest_document_default=allow_latest_document_default,
    )
    if document_id is not None:
        if get_document(document_id) is None:
            raise ValueError("The requested document does not exist.")
        update_active_document(conversation["id"], document_id)
        conversation["active_document_id"] = document_id

    history = get_recent_messages(conversation["id"], CHAT_HISTORY_LIMIT)
    add_message(conversation["id"], "user", question)
    rewrite = rewrite_question(question, history)

    active_document_id = conversation["active_document_id"]
    document = get_document(active_document_id) if active_document_id else None
    if document is None:
        add_message(conversation["id"], "assistant", NO_ACTIVE_DOCUMENT_ANSWER)
        return _base_response(
            question=question,
            standalone_question=rewrite.standalone_question,
            answer=NO_ACTIVE_DOCUMENT_ANSWER,
            status=INSUFFICIENT_DOCUMENT_INFORMATION,
            conversation_id=conversation["id"],
            sources=[],
            provider="unconfigured",
            provider_duration_ms=0.0,
            retrieval_duration_ms=0.0,
            document=None,
        )

    retrieval_started_at = time.perf_counter()
    chunks = get_cached_document_chunks(document["id"])
    relevant_chunks = find_relevant_chunks_semantic(
        question=rewrite.standalone_question,
        chunks=chunks,
        top_k=SEMANTIC_TOP_K,
        min_score=SEMANTIC_SCORE_THRESHOLD,
        fallback_score_threshold=SEMANTIC_FALLBACK_SCORE_THRESHOLD,
    )
    retrieval_duration_ms = (time.perf_counter() - retrieval_started_at) * 1000

    if not relevant_chunks:
        add_message(
            conversation["id"], "assistant", INSUFFICIENT_INFORMATION_ANSWER
        )
        response = _base_response(
            question=question,
            standalone_question=rewrite.standalone_question,
            answer=INSUFFICIENT_INFORMATION_ANSWER,
            status=INSUFFICIENT_DOCUMENT_INFORMATION,
            conversation_id=conversation["id"],
            sources=[],
            provider="unconfigured",
            provider_duration_ms=0.0,
            retrieval_duration_ms=retrieval_duration_ms,
            document=document,
        )
    else:
        result = generate_llm_answer(
            question,
            relevant_chunks,
            standalone_question=rewrite.standalone_question,
            history=history,
        )
        sources = (
            build_sources(relevant_chunks)
            if result.status in {SUCCESS, PARTIAL_INFORMATION}
            else []
        )
        if result.status != PROVIDER_UNAVAILABLE:
            add_message(conversation["id"], "assistant", result.answer)
        response = _base_response(
            question=question,
            standalone_question=rewrite.standalone_question,
            answer=result.answer,
            status=result.status,
            conversation_id=conversation["id"],
            sources=sources,
            provider=result.provider,
            provider_duration_ms=result.provider_duration_ms,
            retrieval_duration_ms=retrieval_duration_ms,
            document=document,
        )

    logger.info(
        "chat conversation=%s document_id=%s context_count=%d chunk_ids=%s "
        "faq_ids=%s scores=%s rewrite_used=%s provider=%s status=%s "
        "retrieval_ms=%.2f provider_ms=%.2f",
        safe_conversation_label(conversation["id"]),
        document["id"],
        len(relevant_chunks),
        [chunk["chunk_id"] for chunk in relevant_chunks],
        [chunk["faq_id"] for chunk in relevant_chunks if "faq_id" in chunk],
        [round(chunk["score"], 3) for chunk in relevant_chunks],
        rewrite.rewrite_used,
        response["provider"],
        response["status"],
        response["retrieval_duration_ms"],
        response["provider_duration_ms"],
    )
    return response
