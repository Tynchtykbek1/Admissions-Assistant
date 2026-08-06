import hashlib
import logging
import re
import time
from collections import OrderedDict
from threading import Lock

from app_settings import CHAT_HISTORY_CHARACTER_LIMIT, CHAT_HISTORY_LIMIT, DOCUMENT_CACHE_SIZE
from dialogue_controller import build_conversation_state, decide_dialogue
from conversation_service import SystemDocumentUnavailable, resolve_conversation
from database import (
    add_message,
    clear_conversation_messages,
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

LOCAL_DIALOGUE_RESPONSES = {
    "incomplete_message": {
        "ru": "Похоже, сообщение не закончено. Напишите, пожалуйста, вопрос ещё раз.",
        "en": "It looks like the message is incomplete. Please send your question again.",
    },
    "capability": {
        "ru": (
            "Я могу рассказать об услугах компании, странах поступления, стоимости "
            "сопровождения, пакетах, языковой подготовке, документах, визах, сроках, "
            "стипендиях и контактах менеджеров. Для точных условий я использую "
            "подтверждённую базу. Если информации нет, я уточню вопрос или предложу "
            "связаться с менеджером."
        ),
        "en": (
            "I can help with company services, study destinations, service pricing, "
            "packages, language preparation, documents, visas, deadlines, scholarships, "
            "and manager contacts. I use verified information for exact conditions. "
            "If something is unclear or unavailable, I will clarify the question or "
            "suggest contacting a manager."
        ),
    },
    "acknowledgement": {"ru": "Хорошо.", "en": "Okay."},
    "restart": {"ru": "Хорошо, начнём заново. Чем могу помочь?", "en": "Okay, let's start over. How can I help?"},
    "manager_contact": {
        "ru": "Связаться с компанией можно через Telegram: @hellhg, @TheLuckiestPersonEver или @maksatuniguide.",
        "en": "You can contact the company on Telegram: @hellhg, @TheLuckiestPersonEver or @maksatuniguide.",
    },
}

INTENT_FALLBACKS = {
    "company_package_contents": {
        "ru": "Точный состав пакета пока не подтверждён.",
        "en": "The exact package contents have not yet been confirmed.",
    },
    "company_guarantees": {
        "ru": "Подтверждённой информации о гарантиях компании пока нет.",
        "en": "There is currently no confirmed information about company guarantees.",
    },
    "refund": {
        "ru": "Условия возврата денег пока не подтверждены.",
        "en": "The refund conditions have not yet been confirmed.",
    },
    "visa_documents": {
        "ru": "В базе пока нет полного подтверждённого перечня документов для визы.",
        "en": "A complete verified list of visa documents is not currently available.",
    },
    "manager_contact": {
        "ru": "Распределение ролей между менеджерами пока не подтверждено.",
        "en": "The division of roles between managers has not yet been confirmed.",
    },
    "language_support": {
        "ru": "Какие именно языковые экзамены поддерживаются, пока не подтверждено.",
        "en": "The specific supported language examinations have not yet been confirmed.",
    },
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
    active_topic: str | None = None,
    clarification_question: str | None = None,
    decision_confidence: float = 0.0,
    reason_code: str = "legacy_route",
    entities: dict | None = None,
    needs_retrieval: bool = False,
    controller_used: bool = False,
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
        "active_topic": active_topic,
        "clarification_question": clarification_question,
        "decision_confidence": round(float(decision_confidence), 3),
        "reason_code": reason_code,
        "entities": entities or {},
        "needs_retrieval": needs_retrieval,
        "controller_used": controller_used,
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


def _language(question: str) -> str:
    return "ru" if any("а" <= character.casefold() <= "я" or character in "ёЁ" for character in question) else "en"


def _intent_fallback(route, question: str) -> str:
    language = _language(question)
    topic = route.active_topic or route.intent
    return INTENT_FALLBACKS.get(topic, {}).get(language) or (
        "Подтверждённой информации по этому вопросу пока нет."
        if language == "ru"
        else "There is currently no confirmed information for this question."
    )


def _provider_failure_answer(question: str) -> str:
    return (
        "Сейчас не удалось получить ответ. Попробуйте ещё раз немного позже."
        if _language(question) == "ru"
        else "I couldn't get an answer right now. Please try again a little later."
    )


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


def _sanitize_grounded_answer(answer: str, question: str, chunks) -> str:
    """Remove user-only numbers and internal implementation wording."""
    context = " ".join(
        str(chunk.get("text") or "")
        for chunk in chunks
        if isinstance(chunk, dict)
    )
    context_digits = re.sub(r"\D", "", context)
    unsupported_numbers = {
        re.sub(r"\D", "", token)
        for token in re.findall(r"(?<!\w)\d[\d\s.,]*", question)
        if re.sub(r"\D", "", token)
        and re.sub(r"\D", "", token) not in context_digits
    }
    internal_markers = (
        "в загруженных документах",
        "в предоставленных документах",
        "в предоставленном контексте",
        "retrieval",
        "chunks",
        " rag ",
    )
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", answer.strip()):
        folded = f" {sentence.casefold()} "
        if any(marker in folded for marker in internal_markers):
            continue
        sentence_numbers = {
            re.sub(r"\D", "", token)
            for token in re.findall(r"(?<!\w)\d[\d\s.,]*", sentence)
        }
        if unsupported_numbers.intersection(sentence_numbers):
            continue
        if sentence.strip():
            kept.append(sentence.strip())
    return " ".join(kept)


def _log_response(conversation: dict, document: dict | None, route, response: dict, history: list[dict], chunks: list[dict]) -> None:
    diagnostics = getattr(chunks, "diagnostics", {})
    logger.info(
        "chat conversation=%s document_id=%s intent=%s response_mode=%s risk_level=%s "
        "active_topic=%s confidence=%.3f needs_retrieval=%s clarification_used=%s "
        "is_follow_up=%s controller_used=%s rewrite_used=%s history_message_count=%d retrieval_used=%s "
        "context_count=%d faq_ids=%s scores=%s provider=%s status=%s "
        "final_response_source=%s retrieval_ms=%.2f provider_ms=%.2f "
        "candidate_count=%d semantic_candidate_count=%d lexical_candidate_count=%d "
        "selected_count=%d selected_faq_ids=%s semantic_scores=%s lexical_scores=%s "
        "final_scores=%s inferred_categories=%s applied_penalties=%s "
        "retrieval_strategy=%s retrieval_confidence=%.4f knowledge_scopes=%s",
        safe_conversation_label(conversation["id"]), document["id"] if document else None,
        route.intent, route.response_mode, route.risk_level, route.active_topic,
        route.confidence, route.needs_retrieval,
        route.response_mode == "clarification", route.is_follow_up,
        route.controller_used, route.rewrite_used, len(history),
        response["retrieval_used"], len(chunks),
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
    conversation_state = build_conversation_state(history)
    route = decide_dialogue(
        question,
        history,
        state=conversation_state,
        language=_language(question),
    )

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
        "active_topic": route.active_topic,
        "clarification_question": route.clarification_question,
        "decision_confidence": route.confidence,
        "reason_code": route.reason_code,
        "entities": route.entities,
        "needs_retrieval": route.needs_retrieval,
        "controller_used": route.controller_used,
    }

    if route.response_mode in {"local_response", "clarification"}:
        language = _language(question)
        if route.response_mode == "clarification":
            answer = route.clarification_question or _intent_fallback(route, question)
        else:
            if route.intent == "restart":
                clear_conversation_messages(conversation["id"])
            answer = (
                INTENT_FALLBACKS["manager_contact"][language]
                if route.reason_code == "unconfirmed_manager_roles"
                else LOCAL_DIALOGUE_RESPONSES.get(route.intent, {}).get(language)
            )
            if answer is None:
                answer = _deterministic_conversational(route.intent, question) or (
                    "Чем могу помочь?" if language == "ru" else "How can I help?"
                )
        add_message(conversation["id"], "assistant", answer)
        response = _base_response(
            answer=answer, status=SUCCESS, sources=[], provider="local",
            provider_duration_ms=0.0, retrieval_duration_ms=0.0,
            retrieval_used=False, final_response_source="local_response", **common,
        )
        _log_response(conversation, document, route, response, history, [])
        return response

    if route.response_mode in {"conversational", "general_knowledge"}:
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
            if status == PROVIDER_UNAVAILABLE:
                answer = _provider_failure_answer(question)
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
        no_document_answer = _intent_fallback(route, question)
        add_message(conversation["id"], "assistant", no_document_answer)
        response = _base_response(
            answer=no_document_answer,
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
        safe_answer = _intent_fallback(route, route.standalone_question)
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
            response_mode=route.response_mode,
        )
        if result.status == INSUFFICIENT_DOCUMENT_INFORMATION:
            _record_unanswered_safely(
                question=question,
                standalone_question=route.standalone_question,
                reason="llm_insufficient_document_information",
                relevant_chunks=relevant_chunks,
            )
        answer = (
            _intent_fallback(route, route.standalone_question)
            if result.status == INSUFFICIENT_DOCUMENT_INFORMATION
            else (_provider_failure_answer(question) if result.status == PROVIDER_UNAVAILABLE else result.answer)
        )
        if result.status not in {INSUFFICIENT_DOCUMENT_INFORMATION, PROVIDER_UNAVAILABLE}:
            answer = _sanitize_grounded_answer(answer, question, relevant_chunks)
            if not answer:
                answer = _intent_fallback(route, route.standalone_question)
                result_status = INSUFFICIENT_DOCUMENT_INFORMATION
            else:
                result_status = result.status
        else:
            result_status = result.status
        diagnostics = getattr(relevant_chunks, "diagnostics", {})
        missing_categories = diagnostics.get("missing_query_categories", [])
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
                ("llm_mixed" if route.response_mode == "mixed" else "llm_verified_rag") if result.status != PROVIDER_UNAVAILABLE
                else "provider_unavailable"
            ),
            **common,
        )

    _log_response(conversation, document, route, response, history, relevant_chunks)
    return response
