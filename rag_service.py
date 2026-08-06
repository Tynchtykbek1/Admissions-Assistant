import hashlib
import logging
import time
from collections import OrderedDict
from threading import Lock

from app_settings import CHAT_HISTORY_CHARACTER_LIMIT, CHAT_HISTORY_LIMIT, DOCUMENT_CACHE_SIZE
from conversation_router import route_conversation
from conversation_service import SystemDocumentUnavailable, resolve_conversation
from database import (
    add_message,
    get_document,
    get_recent_messages,
    load_document_chunks,
    record_unanswered_question,
)
from embedding_retriever import find_relevant_chunks_semantic
from llm_answer_generator import (
    INSUFFICIENT_DOCUMENT_INFORMATION,
    INSUFFICIENT_INFORMATION_ANSWER,
    PARTIAL_INFORMATION,
    PROVIDER_UNAVAILABLE,
    SUCCESS,
    generate_llm_answer,
    generate_conversational_answer,
    generate_safe_general_answer,
    _build_history,
)
from question_rewriter import rewrite_question
from retrieval_settings import (
    SEMANTIC_FALLBACK_SCORE_THRESHOLD,
    SEMANTIC_SCORE_THRESHOLD,
    SEMANTIC_TOP_K,
    CONTEXT_SCORE_MARGIN,
)


logger = logging.getLogger(__name__)

NO_ACTIVE_DOCUMENT_ANSWER = (
    "No document is active for this conversation. Upload or select a document first."
)
SYSTEM_DOCUMENT_UNAVAILABLE = "system_document_unavailable"
SYSTEM_DOCUMENT_UNAVAILABLE_ANSWER = (
    "The knowledge base is temporarily unavailable. Please try again later."
)
SAFE_INSUFFICIENT_ANSWERS = {
    "ru": "В подтверждённой базе пока нет информации, которая отвечает на этот вопрос.",
    "en": "The verified knowledge base does not currently contain information that answers this question.",
}

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


def _record_unanswered_safely(
    *,
    question: str,
    standalone_question: str,
    reason: str,
    relevant_chunks: list[dict],
) -> None:
    scores = [
        float(chunk["score"])
        for chunk in relevant_chunks
        if chunk.get("score") is not None
    ]
    diagnostics = getattr(relevant_chunks, "diagnostics", {})
    max_similarity_score = (
        max(scores) if scores else diagnostics.get("max_candidate_semantic_score")
    )
    faq_ids = sorted(
        {
            int(chunk["faq_id"])
            for chunk in relevant_chunks
            if chunk.get("faq_id") is not None
        }
    )
    try:
        record_unanswered_question(
            question=question,
            standalone_question=standalone_question,
            reason=reason,
            max_similarity_score=max_similarity_score,
            retrieved_faq_ids=faq_ids,
        )
    except Exception:
        logger.error(
            "Failed to record an unanswered question without logging its content."
        )


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
    intent: str = "unknown",
    response_mode: str = "verified_rag",
    risk_level: str = "medium",
    is_follow_up: bool = False,
    rewrite_used: bool = False,
    retrieval_used: bool = False,
    final_response_source: str = "deterministic_fallback",
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
        "intent": intent,
        "response_mode": response_mode,
        "risk_level": risk_level,
        "is_follow_up": is_follow_up,
        "rewrite_used": rewrite_used,
        "retrieval_used": retrieval_used,
        "final_response_source": final_response_source,
    }


def _bounded_history(history: list[dict]) -> list[dict]:
    selected = [
        message for message in history
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    while selected and len(_build_history(selected)) > CHAT_HISTORY_CHARACTER_LIMIT:
        selected.pop(0)
    return selected


def _deterministic_conversational(intent: str, question: str) -> str | None:
    russian = any("а" <= char.casefold() <= "я" or char in "ёЁ" for char in question)
    if intent == "greeting":
        return "Здравствуйте! Чем могу помочь?" if russian else "Hello! How can I help?"
    if intent == "gratitude":
        return "Пожалуйста!" if russian else "You're welcome!"
    return None


def _deterministic_general(question: str) -> str | None:
    normalized = question.casefold()
    definitions = (
        (("бакалавр",), "Бакалавриат — первый уровень высшего образования.", "A bachelor's degree is the first level of higher education."),
        (("магистрат",), "Магистратура — уровень высшего образования после бакалавриата.", "A master's degree is a level of higher education after a bachelor's degree."),
        (("транскрипт", "transcript"), "Транскрипт — документ с перечнем изученных предметов и полученных оценок.", "A transcript is a record of courses studied and grades received."),
        (("мотивацион", "motivation letter", "motivational letter"), "Мотивационное письмо объясняет цели кандидата и причины выбора программы.", "A motivation letter explains an applicant's goals and reasons for choosing a programme."),
    )
    russian = any("а" <= char.casefold() <= "я" or char in "ёЁ" for char in question)
    for markers, ru_answer, en_answer in definitions:
        if any(marker in normalized for marker in markers):
            return ru_answer if russian else en_answer
    return None


def _safe_insufficient_answer(question: str) -> str:
    language = "ru" if any(
        "а" <= character.casefold() <= "я" or character in "ёЁ"
        for character in question
    ) else "en"
    return SAFE_INSUFFICIENT_ANSWERS[language]


_MISSING_ASPECT_LABELS = {
    "ru": {
        "company_pricing": "цене услуг компании",
        "company_guarantees": "гарантиях компании",
        "company_services": "составе услуг компании",
        "tuition": "стоимости обучения",
        "visa_fee": "визовых расходах",
        "admission_guarantee": "гарантии поступления",
        "scholarship_guarantee": "гарантии стипендии",
        "visa_guarantee": "гарантии визы",
    },
    "en": {
        "company_pricing": "the company's service price",
        "company_guarantees": "company guarantees",
        "company_services": "the scope of company services",
        "tuition": "tuition costs",
        "visa_fee": "visa expenses",
        "admission_guarantee": "an admission guarantee",
        "scholarship_guarantee": "a scholarship guarantee",
        "visa_guarantee": "a visa guarantee",
    },
}


def _append_missing_aspects(answer: str, categories: list[str], question: str) -> str:
    """Make deterministic partial coverage explicit without adding factual claims."""
    language = "ru" if any(
        "а" <= character.casefold() <= "я" or character in "ёЁ"
        for character in question
    ) else "en"
    labels = [
        _MISSING_ASPECT_LABELS[language].get(category, category.replace("_", " "))
        for category in categories
    ]
    if language == "ru":
        disclosure = (
            "В подтверждённой базе нет информации о " + " и ".join(labels) + "."
        )
    else:
        disclosure = (
            "The verified knowledge base does not contain information about "
            + " and ".join(labels) + "."
        )
    return f"{answer.rstrip()}\n\n{disclosure}"


def _log_response(conversation: dict, document: dict | None, route, response: dict, history: list[dict], chunks: list[dict]) -> None:
    diagnostics = getattr(chunks, "diagnostics", {})
    logger.info(
        "chat conversation=%s document_id=%s intent=%s response_mode=%s risk_level=%s "
        "is_follow_up=%s rewrite_used=%s history_message_count=%d retrieval_used=%s "
        "context_count=%d faq_ids=%s scores=%s provider=%s status=%s "
        "final_response_source=%s retrieval_ms=%.2f provider_ms=%.2f "
        "candidate_count=%d semantic_candidate_count=%d lexical_candidate_count=%d "
        "selected_count=%d selected_faq_ids=%s semantic_scores=%s lexical_scores=%s "
        "final_scores=%s inferred_categories=%s applied_penalties=%s "
        "retrieval_strategy=%s retrieval_confidence=%.4f knowledge_scopes=%s",
        safe_conversation_label(conversation["id"]), document["id"] if document else None,
        route.intent, route.response_mode, route.risk_level, route.is_follow_up,
        route.rewrite_used, len(history), response["retrieval_used"], len(chunks),
        [chunk["faq_id"] for chunk in chunks if "faq_id" in chunk],
        [round(chunk["score"], 3) for chunk in chunks], response["provider"],
        response["status"], response["final_response_source"],
        response["retrieval_duration_ms"], response["provider_duration_ms"],
        diagnostics.get("candidate_count", len(chunks)),
        diagnostics.get("semantic_candidate_count", len(chunks)),
        diagnostics.get("lexical_candidate_count", 0),
        diagnostics.get("selected_count", len(chunks)),
        diagnostics.get("selected_faq_ids", [chunk.get("faq_id") for chunk in chunks if chunk.get("faq_id") is not None]),
        diagnostics.get("semantic_scores", [round(chunk["score"], 3) for chunk in chunks]),
        diagnostics.get("lexical_scores", [round(chunk.get("lexical_score", 0.0), 3) for chunk in chunks]),
        diagnostics.get("final_scores", [round(chunk.get("final_score", chunk["score"]), 3) for chunk in chunks]),
        diagnostics.get("inferred_categories", [chunk.get("inferred_categories", []) for chunk in chunks]),
        diagnostics.get("applied_penalties", []),
        diagnostics.get("retrieval_strategy", "none" if not response["retrieval_used"] else "semantic_v1"),
        float(diagnostics.get("retrieval_confidence", 0.0)),
        diagnostics.get("knowledge_scopes", []),
    )


def answer_conversation_question(
    *,
    question: str,
    conversation_id: str | None = None,
    external_chat_id: str | None = None,
    external_user_id: str | None = None,
    document_id: int | None = None,
) -> dict:
    system_unavailable_category = None
    try:
        conversation = resolve_conversation(
            conversation_id=conversation_id,
            external_chat_id=external_chat_id,
            external_user_id=external_user_id,
            requested_document_id=document_id,
        )
    except SystemDocumentUnavailable as error:
        conversation = error.conversation
        if conversation is None:
            raise
        system_unavailable_category = error.category

    history = _bounded_history(get_recent_messages(conversation["id"], CHAT_HISTORY_LIMIT))
    add_message(conversation["id"], "user", question)
    route = route_conversation(question, history, rewrite_function=rewrite_question)

    active_document_id = conversation["active_document_id"]
    document = get_document(active_document_id) if active_document_id else None
    common = {
        "question": question,
        "standalone_question": route.standalone_question,
        "conversation_id": conversation["id"],
        "document": document,
        "intent": route.intent,
        "response_mode": route.response_mode,
        "risk_level": route.risk_level,
        "is_follow_up": route.is_follow_up,
        "rewrite_used": route.rewrite_used,
    }

    if route.response_mode in {"conversational", "safe_general"}:
        deterministic = (
            _deterministic_conversational(route.intent, question)
            if route.response_mode == "conversational"
            else _deterministic_general(question)
        )
        if deterministic is not None:
            answer, status, provider, provider_ms = deterministic, SUCCESS, "local", 0.0
            final_source = "deterministic_fallback"
        else:
            generator = generate_conversational_answer if route.response_mode == "conversational" else generate_safe_general_answer
            result = generator(question, history=history)
            answer, status, provider, provider_ms = result.answer, result.status, result.provider, result.provider_duration_ms
            final_source = (
                f"llm_{route.response_mode}" if result.status != PROVIDER_UNAVAILABLE
                else "provider_unavailable"
            )
        if status != PROVIDER_UNAVAILABLE:
            add_message(conversation["id"], "assistant", answer)
        response = _base_response(
            answer=answer, status=status, sources=[], provider=provider,
            provider_duration_ms=provider_ms, retrieval_duration_ms=0.0,
            retrieval_used=False, final_response_source=final_source, **common,
        )
        _log_response(conversation, document, route, response, history, [])
        return response

    if system_unavailable_category is not None:
        response = _base_response(
            answer=SYSTEM_DOCUMENT_UNAVAILABLE_ANSWER,
            status=SYSTEM_DOCUMENT_UNAVAILABLE,
            sources=[],
            provider="unconfigured",
            provider_duration_ms=0.0,
            retrieval_duration_ms=0.0,
            retrieval_used=False,
            final_response_source="provider_unavailable",
            **common,
        )
        _log_response(conversation, document, route, response, history, [])
        return response

    if document is None:
        add_message(conversation["id"], "assistant", NO_ACTIVE_DOCUMENT_ANSWER)
        response = _base_response(
            answer=NO_ACTIVE_DOCUMENT_ANSWER,
            status=INSUFFICIENT_DOCUMENT_INFORMATION,
            sources=[],
            provider="unconfigured",
            provider_duration_ms=0.0,
            retrieval_duration_ms=0.0,
            retrieval_used=False,
            final_response_source="deterministic_fallback",
            **common,
        )
        _log_response(conversation, document, route, response, history, [])
        return response

    retrieval_started_at = time.perf_counter()
    chunks = get_cached_document_chunks(document["id"])
    relevant_chunks = find_relevant_chunks_semantic(
        question=route.standalone_question,
        chunks=chunks,
        top_k=SEMANTIC_TOP_K,
        min_score=SEMANTIC_SCORE_THRESHOLD,
        fallback_score_threshold=SEMANTIC_FALLBACK_SCORE_THRESHOLD,
        context_score_margin=CONTEXT_SCORE_MARGIN,
        intent=route.intent,
        risk_level=route.risk_level,
    )
    retrieval_duration_ms = (time.perf_counter() - retrieval_started_at) * 1000

    if not relevant_chunks:
        _record_unanswered_safely(
            question=question,
            standalone_question=route.standalone_question,
            reason="no_relevant_chunks",
            relevant_chunks=relevant_chunks,
        )
        safe_answer = _safe_insufficient_answer(route.standalone_question)
        add_message(conversation["id"], "assistant", safe_answer)
        response = _base_response(
            answer=safe_answer,
            status=INSUFFICIENT_DOCUMENT_INFORMATION,
            sources=[],
            provider="unconfigured",
            provider_duration_ms=0.0,
            retrieval_duration_ms=retrieval_duration_ms,
            retrieval_used=True,
            final_response_source="deterministic_fallback",
            **common,
        )
    else:
        result = generate_llm_answer(
            route.standalone_question,
            relevant_chunks,
            standalone_question=route.standalone_question,
            history=history,
        )
        if result.status == INSUFFICIENT_DOCUMENT_INFORMATION:
            _record_unanswered_safely(
                question=question,
                standalone_question=route.standalone_question,
                reason="llm_insufficient_document_information",
                relevant_chunks=relevant_chunks,
            )
        answer = (
            _safe_insufficient_answer(route.standalone_question)
            if result.status == INSUFFICIENT_DOCUMENT_INFORMATION
            else result.answer
        )
        diagnostics = getattr(relevant_chunks, "diagnostics", {})
        missing_categories = diagnostics.get("missing_query_categories", [])
        result_status = result.status
        if missing_categories and result_status in {SUCCESS, PARTIAL_INFORMATION}:
            result_status = PARTIAL_INFORMATION
            answer = _append_missing_aspects(
                answer, missing_categories, route.standalone_question
            )
        sources = (
            build_sources(relevant_chunks)
            if result_status in {SUCCESS, PARTIAL_INFORMATION}
            else []
        )
        if result.status != PROVIDER_UNAVAILABLE:
            add_message(conversation["id"], "assistant", answer)
        response = _base_response(
            answer=answer,
            status=result_status,
            sources=sources,
            provider=result.provider,
            provider_duration_ms=result.provider_duration_ms,
            retrieval_duration_ms=retrieval_duration_ms,
            retrieval_used=True,
            final_response_source=(
                "llm_verified_rag" if result.status != PROVIDER_UNAVAILABLE
                else "provider_unavailable"
            ),
            **common,
        )

    _log_response(conversation, document, route, response, history, relevant_chunks)
    return response
