import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import replace
from threading import Lock

from .app_settings import CHAT_HISTORY_CHARACTER_LIMIT, CHAT_HISTORY_LIMIT, DOCUMENT_CACHE_SIZE
from .conversation_service import resolve_conversation
from .database import add_message, get_document, get_recent_messages, load_document_chunks, record_unanswered_question
from .embedding_retriever import find_relevant_chunks_semantic
from .llm_answer_generator import INSUFFICIENT_DOCUMENT_INFORMATION, INSUFFICIENT_INFORMATION_ANSWER, PROVIDER_UNAVAILABLE, SUCCESS, generate_conversation_answer
from .retrieval_settings import CONTEXT_SCORE_MARGIN, SEMANTIC_FALLBACK_SCORE_THRESHOLD, SEMANTIC_SCORE_THRESHOLD, SEMANTIC_TOP_K


logger = logging.getLogger(__name__)
SYSTEM_DOCUMENT_UNAVAILABLE = "system_document_unavailable"
SYSTEM_DOCUMENT_UNAVAILABLE_ANSWER = "The knowledge base is temporarily unavailable. Please try again later."

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
    sources, seen = [], set()
    for chunk in relevant_chunks:
        identity = (chunk["filename"], chunk.get("faq_id"), chunk["chunk_id"])
        if identity in seen:
            continue
        seen.add(identity)
        source = {
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "score": round(float(chunk["score"]), 6),
            "preview": chunk["text"][:200],
        }
        if chunk.get("faq_id") is not None:
            source["faq_id"] = chunk["faq_id"]
        sources.append(source)
    return sources


def safe_conversation_label(conversation_id: str) -> str:
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:12]


def _bounded_history(history: list[dict]) -> list[dict]:
    selected = [
        {"role": item["role"], "content": item["content"].strip()}
        for item in history
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ][-CHAT_HISTORY_LIMIT:]
    while selected and sum(len(item["content"]) for item in selected) > CHAT_HISTORY_CHARACTER_LIMIT:
        selected.pop(0)
    return selected


def _chunk_content(chunk: dict) -> str:
    if chunk.get("question") or chunk.get("answer"):
        return "\n".join(part for part in (chunk.get("question"), chunk.get("answer")) if part)
    return str(chunk.get("text") or "")


def _tool_result(chunks: list[dict]) -> dict:
    results = []
    for chunk in chunks:
        results.append({
            "id": str(chunk.get("faq_id") or chunk["chunk_id"]),
            "content": _chunk_content(chunk),
            "source": chunk["filename"],
            "scope": chunk.get("knowledge_scope") or chunk.get("scope") or "document",
            "score": round(float(chunk.get("final_score", chunk["score"])), 6),
        })
    return {"results": results, "has_relevant_context": bool(results)}


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _factual_guard(answer: str, tool_output: dict | None) -> tuple[str, bool]:
    """Drop only high-risk concrete sentences unsupported by tool output."""
    context = " ".join(str(item.get("content") or "") for item in (tool_output or {}).get("results", []))
    context_folded = context.casefold()
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    kept, triggered = [], False
    for sentence in sentences:
        folded = sentence.casefold()
        numbers = re.findall(r"(?<!\w)\d[\d\s.,]*(?:%|\s*(?:€|eur|евро|руб(?:лей|ля)?|доллар\w*))?", sentence, re.I)
        contacts = re.findall(r"(?<!\w)@[A-Za-z0-9_]{3,}", sentence)
        guarantee_claim = bool(re.search(r"\b(?:гарантир\w*|guarantee(?:d|s)?)\b", folded))
        deadline_claim = bool(re.search(r"\b(?:дедлайн|срок\w*|deadline)\b", folded) and numbers)
        unsupported_number = any(_digits(token) and _digits(token) not in _digits(context) for token in numbers)
        unsupported_contact = any(contact.casefold() not in context_folded for contact in contacts)
        unsupported_guarantee = guarantee_claim and not any(word in context_folded for word in ("гарант", "guarantee"))
        if unsupported_number or unsupported_contact or unsupported_guarantee or (deadline_claim and not context):
            triggered = True
            continue
        if sentence.strip():
            kept.append(sentence.strip())
    guarded = " ".join(kept).strip()
    if triggered and not guarded:
        guarded = (
            "Точная информация по этому вопросу пока не подтверждена."
            if re.search(r"[А-Яа-яЁё]", answer)
            else "The exact information is not currently confirmed."
        )
    return guarded, triggered


def _compat_response(*, question, answer, conversation, document, result, chunks, retrieval_ms, guard_triggered):
    tool_called = result.tool_called
    return {
        "question": question,
        "standalone_question": result.tool_query or question,
        "answer": answer,
        "status": result.status,
        "sources": build_sources(chunks) if tool_called else [],
        "conversation_id": conversation["id"],
        "document_id": document["id"] if document else None,
        "document_filename": document["filename"] if document else None,
        "provider": result.provider,
        "provider_duration_ms": round(result.provider_duration_ms, 2),
        "retrieval_duration_ms": round(retrieval_ms, 2),
        "intent": "llm_managed",
        "response_mode": "verified_rag" if tool_called else "conversational",
        "risk_level": "model_managed",
        "is_follow_up": False,
        "rewrite_used": False,
        "retrieval_used": tool_called,
        "tool_called": tool_called,
        "tool_name": result.tool_name,
        "retrieval_result_count": len(chunks),
        "verified_context_used": bool(tool_called and chunks),
        "final_guard_triggered": guard_triggered,
        "final_response_source": "llm_tool" if tool_called else "llm_direct",
        "active_topic": None,
        "clarification_question": None,
        "decision_confidence": 0.0,
        "reason_code": "llm_tool_decision" if tool_called else "llm_direct_decision",
        "entities": {},
        "needs_retrieval": tool_called,
        "controller_used": False,
        "provider_error_category": result.error_category,
    }


def answer_conversation_question(*, question: str, conversation_id: str | None = None, external_chat_id: str | None = None, external_user_id: str | None = None, document_id: int | None = None) -> dict:
    conversation = resolve_conversation(
        conversation_id=conversation_id,
        external_chat_id=external_chat_id,
        external_user_id=external_user_id,
        requested_document_id=document_id,
    )

    # Read before persisting current_message, so it appears exactly once to the model.
    history = _bounded_history(get_recent_messages(conversation["id"], CHAT_HISTORY_LIMIT))
    add_message(conversation["id"], "user", question)
    active_document_id = conversation.get("active_document_id")
    document = get_document(active_document_id) if active_document_id else None
    selected_chunks: list[dict] = []
    retrieval_ms = 0.0

    def search_knowledge(query: str) -> dict:
        nonlocal selected_chunks, retrieval_ms
        started = time.perf_counter()
        if document is not None:
            selected_chunks = find_relevant_chunks_semantic(
                question=query,
                chunks=get_cached_document_chunks(document["id"]),
                top_k=SEMANTIC_TOP_K,
                min_score=SEMANTIC_SCORE_THRESHOLD,
                fallback_score_threshold=SEMANTIC_FALLBACK_SCORE_THRESHOLD,
                context_score_margin=CONTEXT_SCORE_MARGIN,
                intent="unknown",
                risk_level="high",
            )
        retrieval_ms = (time.perf_counter() - started) * 1000
        if not selected_chunks:
            try:
                record_unanswered_question(
                    question=question, standalone_question=query,
                    reason="no_relevant_chunks", max_similarity_score=None,
                    retrieved_faq_ids=[],
                )
            except Exception:
                logger.error("Failed to record an unanswered question without logging its content.")
        return _tool_result(selected_chunks)

    result = generate_conversation_answer(question, history, search_knowledge)
    if result.tool_called and not selected_chunks:
        result = replace(
            result,
            status=INSUFFICIENT_DOCUMENT_INFORMATION,
            answer=INSUFFICIENT_INFORMATION_ANSWER,
        )
    answer, guard_triggered = _factual_guard(result.answer, result.tool_output)
    if result.status != PROVIDER_UNAVAILABLE:
        add_message(conversation["id"], "assistant", answer)
    response = _compat_response(
        question=question, answer=answer, conversation=conversation, document=document,
        result=result, chunks=selected_chunks, retrieval_ms=retrieval_ms,
        guard_triggered=guard_triggered,
    )
    logger.info(
        "conversation_complete conversation=%s llm_called=true tool_called=%s tool_name=%s "
        "retrieval_result_count=%d verified_context_used=%s final_guard_triggered=%s provider=%s latency_ms=%.2f",
        safe_conversation_label(conversation["id"]), result.tool_called,
        result.tool_name or "none", len(selected_chunks), bool(result.tool_called and selected_chunks),
        guard_triggered, result.provider, result.provider_duration_ms + retrieval_ms,
    )
    return response
