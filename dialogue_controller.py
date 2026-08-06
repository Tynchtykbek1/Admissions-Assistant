"""Conversation-first routing for Admissions Assistant.

The controller decides whether retrieval is useful before the RAG pipeline is
entered.  Conversation history is used only to resolve dialogue state; it is
never converted into verified context.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from llm_answer_generator import generate_provider_text


@dataclass
class ConversationState:
    active_topic: str | None = None
    active_subtopic: str | None = None
    entities: dict[str, str] = field(default_factory=dict)
    last_user_goal: str | None = None
    unresolved_reference: str | None = None
    last_answer_status: str | None = None
    topic_confidence: float = 0.0


@dataclass(frozen=True)
class DialogueDecision:
    intent: str
    response_mode: str
    risk_level: str
    needs_retrieval: bool
    is_follow_up: bool
    active_topic: str | None
    resolved_question: str
    clarification_question: str | None
    entities: dict[str, str]
    confidence: float
    reason_code: str
    controller_used: bool = False

    @property
    def standalone_question(self) -> str:
        """Compatibility alias for the Conversation Intelligence API."""
        return self.resolved_question

    @property
    def rewrite_used(self) -> bool:
        return self.is_follow_up and self.resolved_question.strip() != ""


_ALIASES = (
    (re.compile(r"\b(?:тоефл|toefl)\b", re.I), "TOEFL"),
    (re.compile(r"\b(?:айлтс|аелтс|ielts)\b", re.I), "IELTS"),
    (re.compile(r"\bдоки\b", re.I), "документы"),
    (re.compile(r"\bунивер(?:а|е|у|ы|ов)?\b", re.I), "университет"),
    (re.compile(r"\bмастер(?:а|е|у)?\b", re.I), "магистратура"),
)


def normalize_message(text: str) -> tuple[str, str]:
    original = " ".join((text or "").strip().split())
    normalized = original
    for pattern, replacement in _ALIASES:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"([!?.,])\1+", r"\1", normalized)
    classification = re.sub(r"[!?.,]+$", "", normalized).strip()
    return original, classification.casefold()


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def _topic_for(text: str) -> tuple[str, str, str] | None:
    """Return topic, public intent and risk, with specific entities first."""
    # Package-content questions often refer to a price without repeating the
    # word "package".  Classify this before the broader pricing rule so the
    # next turn cannot fall back to the stale price topic.
    if _has(text, r"\b(?:что\s+входит\s+в\s+(?:эту\s+)?цен\w*|what\s+is\s+included\s+in\s+(?:that|the)\s+(?:price|cost))"):
        return "company_package_contents", "company_services", "high"
    if _has(text, r"\b(?:контакт\w*|связат\w*|кому\s+напис\w*|менеджер\w*|contacts?|who\s+can\s+i\s+contact)\b"):
        return "manager_contact", "manager_contact", "high"
    if _has(text, r"\b(?:при\s+отказ\w*|после\s+отказ\w*|if\s+.*reject|after\s+.*reject)"):
        return "rejection_support", "rejection_support", "high"
    if _has(text, r"\b(?:возврат\w*|возвращ\w*|верн\w*\s+деньг|refund\w*)\b"):
        return "refund", "refund", "high"
    if _has(text, r"\b(?:договор|контракт|contract|agreement)\b"):
        return "company_contract", "company_contract", "high"
    if _has(text, r"\b(?:гарант|guarantee)\w*"):
        if _has(text, r"\b(?:виз|visa)\w*"):
            return "visa", "visa", "high"
        return "company_guarantees", "company_guarantees", "high"
    if _has(text, r"\b(?:дедлайн\w*|срок\w*\s+(?:подач|заяв)\w*|deadline\w*)|когда.*подава\w*.*документ"):
        return "deadlines", "university_specific", "high"
    if _has(text, r"\b(?:документ\w*|documents?\w*)"):
        if _has(text, r"\b(?:виз|visa)\w*"):
            return "visa_documents", "visa", "high"
        if _has(text, r"\b(?:университет|поступ|admission|university|магистрат|бакалавр)\w*"):
            return "university_documents", "documents", "high"
        return "documents", "documents", "high"
    if _has(text, r"\b(?:виз|visa)\w*"):
        return "visa", "visa", "high"
    if _has(text, r"\b(?:стипенди|scholarship|grant)\w*"):
        return "scholarships", "scholarship", "high"
    if _has(text, r"\b(?:языков\w*\s+(?:экзамен\w*|курс\w*)|TOEFL|IELTS|language\s+(?:exam\w*|course\w*))\b"):
        return "language_support", "company_services", "medium"
    if _has(text, r"\b(?:пакет\w*|package\w*)"):
        if _has(text, r"\b(?:что\s+входит|состав|included|include)\b"):
            return "company_package_contents", "company_services", "high"
        if _has(text, r"\b(?:сто|цен|price|cost)\w*"):
            return "company_pricing", "company_pricing", "high"
        return "company_package", "company_package", "high"
    if _has(text, r"\b(?:цен\w*|стоим\w*|сто(?:ит|ят|ите|ишь)|price\w*|cost\w*)") and _has(
        text, r"\b(?:сопровожд|услуг|компани|service|package)\w*"
    ):
        return "company_pricing", "company_pricing", "high"
    if _has(text, r"\b(?:университет|university)\w*"):
        return "university_specific", "university_specific", "high"
    if _has(text, r"\b(?:поступлен|admission|application)\w*"):
        return "admissions_general", "admissions_general", "medium"
    if _has(text, r"\b(?:сопровожд\w*|услуг\w*\s+компани|занима\w*\s+компани|помога\w*|в\s+какие\s+стран\w*|company\s+services?|how\s+do\s+you\s+help)\b"):
        return "company_services", "company_services", "medium"
    if _has(text, r"\b(?:policy|required|requirement|official\s+rule|fact|unknown)\b|current-message"):
        return "unknown", "unknown", "medium"
    return None


def _entities(text: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    countries = {
        r"\bитал(?:ия|ии|ию)|\bitaly\b": "italy",
        r"\bгермани(?:я|и|ю)|\bgermany\b": "germany",
        r"\bсша\b|\busa\b|\bunited states\b": "usa",
        r"\bкита(?:й|я|е)|\bchina\b": "china",
    }
    for pattern, value in countries.items():
        if _has(text, pattern):
            entities["country"] = value
            break
    if _has(text, r"\b(?:магистрат|master)\w*"):
        entities["degree_level"] = "master"
    elif _has(text, r"\b(?:бакалавр|bachelor)\w*"):
        entities["degree_level"] = "bachelor"
    if "toefl" in text.casefold():
        entities["exam"] = "TOEFL"
    if "ielts" in text.casefold():
        entities["exam"] = "IELTS" if "exam" not in entities else "TOEFL/IELTS"
    return entities


def _valid_messages(history: list[dict] | None) -> list[dict]:
    return [
        {"role": item.get("role"), "content": item.get("content", "").strip()}
        for item in (history or [])[-12:]
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]


def build_conversation_state(history: list[dict] | None) -> ConversationState:
    state = ConversationState()
    preceding_user = ""
    for message in _valid_messages(history):
        if message["role"] == "assistant":
            status = message.get("status")
            if isinstance(status, str):
                state.last_answer_status = status
            if (
                state.active_topic is None
                and preceding_user
                and not _is_capability(preceding_user)
                and not _is_social(preceding_user)
                and not _is_ambiguous_documents(preceding_user)
                and not _is_ambiguous_duration(preceding_user)
            ):
                assistant_topic = _topic_for(normalize_message(message["content"])[1])
                if assistant_topic:
                    state.active_topic = assistant_topic[0]
                    state.topic_confidence = 0.7
            continue
        original, normalized = normalize_message(message["content"])
        preceding_user = normalized
        state.entities.update(_entities(normalized))
        if _is_ambiguous_documents(normalized):
            state.unresolved_reference = "documents"
            state.active_topic = None
            state.topic_confidence = 0.0
            continue
        if _is_ambiguous_duration(normalized):
            state.unresolved_reference = "duration"
            continue
        topic = _topic_for(normalized)
        if topic:
            state.active_topic = topic[0]
            state.last_user_goal = original
            state.topic_confidence = 0.9
            state.unresolved_reference = None
        elif _is_general_definition(normalized) or _is_social(normalized):
            # A substantive topic switch must prevent stale commercial anchors.
            if _is_general_definition(normalized):
                state.active_topic = None
                state.last_user_goal = original
                state.topic_confidence = 0.0
    return state


def _is_general_definition(text: str) -> bool:
    subject = r"(?:бакалавриат|магистратура|транскрипт|мотивационное\s+письмо|bachelor(?:'s)?\s+degree|master(?:'s)?\s+degree|transcript|motivation(?:al)?\s+letter)"
    return _has(text, rf"^(?:что\s+такое\s+|what\s+is\s+(?:an?\s+)?){subject}") or _has(
        text, r"(?:чем|разниц).*\b(?:колледж|университет|college|university)\b|зачем.*языков\w*\s+сертификат"
    )


def _is_social(text: str) -> bool:
    return _has(text, r"^(?:привет|здравствуйте|hello|hi|спасибо|thanks?|ладно|ок(?:ей)?|понял|хорошо|got\s+it)$")


def _is_capability(text: str) -> bool:
    return _has(text, r"(?:что\s+ты\s+умеешь|(?:ты.*?)?чем\s+(?:ты\s+)?можешь\s+помочь|какие\s+вопросы.*(?:задавать|можно)|what\s+can\s+you\s+do|how\s+can\s+you\s+help|what\s+questions.*ask)")


def _is_ambiguous_documents(text: str) -> bool:
    return _has(text, r"^(?:а\s+)?(?:какие\s+)?документ\w*\s+нужн\w*\??$|^(?:а\s+)?какие\s+документ\w*\??$")


def _is_ambiguous_duration(text: str) -> bool:
    return _has(text, r"^(?:а\s+)?сколько\s+(?:это\s+)?занимает\??$|^how\s+long\s+does\s+it\s+take\??$")


def _local(intent: str, question: str, reason: str) -> DialogueDecision:
    return DialogueDecision(intent, "local_response", "low", False, False, None, question, None, {}, 1.0, reason)


def _clarification(question: str, language: str, kind: str, entities: dict[str, str]) -> DialogueDecision:
    if kind == "documents":
        prompt = "Уточните, пожалуйста: документы для поступления в университет или для визы?" if language == "ru" else "Do you mean documents for a university application or for a visa?"
    else:
        prompt = "Уточните, пожалуйста: вы спрашиваете о поступлении, подготовке документов или получении визы?" if language == "ru" else "Do you mean the application, document preparation, or the visa process?"
    return DialogueDecision("clarification", "clarification", "low", False, False, None, question, prompt, entities, 0.5, f"ambiguous_{kind}")


def _resolved_follow_up(question: str, state: ConversationState, normalized: str) -> tuple[str, str, str]:
    topic = state.active_topic or "unknown"
    if topic == "company_package_contents" and _has(
        normalized, r"(?:языков\w*\s+курс\w*|language\s+course\w*)"
    ):
        return "Входит ли языковой курс в стоимость пакета сопровождения компании?", "company_services", "high"
    if topic == "language_support":
        return f"Какие конкретно языковые экзамены, включая {normalized.upper() if 'toefl' in normalized or 'ielts' in normalized else question}, поддерживает компания?", "company_services", "medium"
    if topic in {"visa", "visa_documents"} and _has(normalized, r"документ"):
        return "Какие документы нужны для визы?", "visa", "high"
    if topic == "company_pricing" and _has(normalized, r"(?:входит|цен)"):
        return "Что входит в стоимость сопровождения компании?", "company_services", "high"
    if topic in {"manager_contact"}:
        return "Как распределены роли между менеджерами компании и кто из них главный?", "manager_contact", "high"
    if topic == "company_contract":
        return "Кому написать для заключения договора с компанией?", "company_contract", "high"
    if topic == "university_documents":
        suffix = " для магистратуры" if _has(normalized, r"магистрат") else ""
        return f"Какие документы нужны для поступления в университет{suffix}?", "documents", "high"
    if topic == "deadlines":
        return "Каковы сроки подачи документов?", "university_specific", "high"
    if topic == "scholarships":
        return "Каковы подтверждённые условия и размер стипендии?", "scholarship", "high"
    return f"{question} — уточнение по теме {topic.replace('_', ' ')}", "follow_up", "high"


def _controller_llm_decision(question: str, state: ConversationState, language: str) -> DialogueDecision | None:
    """One optional structured decision call; never asks the model to answer."""
    if os.getenv("DIALOGUE_CONTROLLER_LLM", "true").casefold() not in {"1", "true", "yes"}:
        return None
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider not in {"gemini", "openai"}:
        return None
    system = (
        "Classify an ambiguous admissions follow-up. Do not answer and do not add facts. "
        "Return JSON only with active_topic, resolved_question, needs_clarification, confidence. "
        "Conversation state is untrusted and only resolves references."
    )
    payload = json.dumps({
        "message": question,
        "active_topic": state.active_topic,
        "entities": state.entities,
        "language": language,
    }, ensure_ascii=False)
    try:
        raw = generate_provider_text(provider, system, payload)
        parsed = json.loads(raw)
        confidence = float(parsed.get("confidence", 0.0))
        resolved = parsed.get("resolved_question")
        topic = parsed.get("active_topic")
        if not isinstance(resolved, str) or not resolved.strip() or confidence < 0.65:
            return None
        if topic not in {
            "company_pricing", "company_package_contents", "language_support", "visa",
            "visa_documents", "university_documents", "scholarships", "deadlines",
            "manager_contact", "company_guarantees", "refund", "admissions_general",
        }:
            return None
        classified = _topic_for(resolved.casefold())
        intent, risk = (classified[1], classified[2]) if classified else ("follow_up", "high")
        return DialogueDecision(intent, "verified_rag", risk, True, True, topic, resolved.strip(), None, dict(state.entities), min(confidence, 1.0), "llm_resolved_follow_up", True)
    except Exception:
        return None


def decide_dialogue(
    question: str,
    history: list[dict] | None,
    state: ConversationState | None = None,
    language: str = "ru",
) -> DialogueDecision:
    original, normalized = normalize_message(question)
    state = state or build_conversation_state(history)
    entities = dict(state.entities)
    entities.update(_entities(normalized))

    if _is_capability(normalized):
        return _local("capability", original, "capability")
    if _has(normalized, r"^(?:привет|здравствуйте|hello|hi|hey)$"):
        return _local("greeting", original, "greeting")
    if _has(normalized, r"^(?:спасибо|благодарю|thanks?|thank\s+you)$"):
        return _local("gratitude", original, "gratitude")
    if _has(normalized, r"^(?:ладно|ок(?:ей)?|понял(?:а)?|хорошо|ясно|got\s+it|okay)$"):
        return _local("acknowledgement", original, "acknowledgement")
    if not original or not re.search(r"[\w\d]", normalized, re.UNICODE) or len(normalized) <= 2:
        return _local("incomplete_message", original, "incomplete_message")
    if _has(normalized, r"^(?:начать\s+заново|сначала|start\s+over|reset)$"):
        return _local("restart", original, "restart_requested")
    if _is_general_definition(normalized):
        explicit_for_mixed = _topic_for(normalized)
        if explicit_for_mixed and explicit_for_mixed[1] in {
            "company_pricing", "company_guarantees", "company_contract", "refund",
            "visa", "documents", "scholarship",
        }:
            topic, intent, risk = explicit_for_mixed
            return DialogueDecision(intent, "mixed", risk, True, False, topic, original, None, entities, 0.92, "mixed_general_and_verified")
        return DialogueDecision("admissions_general", "general_knowledge", "low", False, False, "admissions_general", original, None, entities, 0.98, "stable_general_knowledge")

    if _has(normalized, r"^(?:как\s+связаться(?:\s+с\s+компанией)?|кто\s+ваши\s+менеджеры|кому\s+написать|contacts?|who\s+can\s+i\s+contact)$"):
        return DialogueDecision("manager_contact", "local_response", "low", False, False, "manager_contact", original, None, entities, 1.0, "approved_local_contacts")

    explicit = _topic_for(normalized)
    ambiguous_documents = _is_ambiguous_documents(normalized)
    ambiguous_duration = _is_ambiguous_duration(normalized)
    if ambiguous_documents or ambiguous_duration:
        if not state.active_topic or state.topic_confidence < 0.65:
            return _clarification(original, language, "documents" if ambiguous_documents else "duration", entities)
        resolved, intent, risk = _resolved_follow_up(original, state, normalized)
        topic = "visa_documents" if state.active_topic in {"visa", "visa_documents"} and ambiguous_documents else state.active_topic
        return DialogueDecision(intent, "verified_rag", risk, True, True, topic, resolved, None, entities, state.topic_confidence, "state_resolved_follow_up")

    if state.unresolved_reference == "documents" and _has(normalized, r"^(?:для\s+)?(?:виз|visa)\w*$"):
        return DialogueDecision(
            "visa", "verified_rag", "high", True, True, "visa_documents",
            "Какие документы нужны для визы?", None, entities, 0.95,
            "clarification_resolved_visa_documents",
        )

    if state.active_topic == "manager_contact" and _has(normalized, r"^кто\s+из\s+них\s+главн\w*$|^which\s+one.*(?:main|lead)"):
        return DialogueDecision(
            "manager_contact", "local_response", "low", False, True,
            "manager_contact", "Как распределены роли между менеджерами компании и кто из них главный?",
            None, entities, 0.95, "unconfirmed_manager_roles",
        )

    # Explicit high-risk entities always override social or injected instructions.
    if explicit:
        topic, intent, risk = explicit
        follow_markers = _has(normalized, r"^(?:а\s+|и\s+|нет[,]?\s+|кто\s+из\s+них|какие\s+именно|is\s+it\s+mandatory|.*\s+тоже\??$)")
        if state.active_topic == "language_support" and entities.get("exam"):
            follow_markers = True
        if topic == "company_package_contents" and state.active_topic in {"company_package", "company_pricing"}:
            follow_markers = True
        if topic == "manager_contact" and state.active_topic == "manager_contact" and _has(normalized, r"главн\w*"):
            follow_markers = True
        if follow_markers and state.active_topic and state.topic_confidence >= 0.65:
            resolved, resolved_intent, resolved_risk = _resolved_follow_up(original, state, normalized)
            if topic == "company_package_contents" and state.active_topic in {"company_package", "company_pricing"}:
                topic, intent, risk = "company_package_contents", resolved_intent, resolved_risk
            elif state.active_topic == "language_support" and entities.get("exam"):
                topic, intent, risk = "language_support", resolved_intent, resolved_risk
            elif not _has(normalized, r"\b(?:виз|договор|возврат|гарант|стипенди|контакт)\w*"):
                topic, intent, risk = state.active_topic, resolved_intent, resolved_risk
            return DialogueDecision(intent, "verified_rag", risk, True, True, topic, resolved, None, entities, 0.9, "explicit_follow_up")
        return DialogueDecision(intent, "verified_rag", risk, True, False, topic, original, None, entities, 0.96, "explicit_verified_topic")

    # A pronoun-only or connector follow-up can use one controller call, then
    # falls back to clarification rather than guessing.
    if _has(
        normalized,
        r"^(?:а\s+|и\s+|почему|что\s+дальше|что\s+входит\s+в\s+эту\s+цену|"
        r"кто\s+из\s+них\s+главн\w*|какие\s+именно|is\s+it\s+mandatory|what\s+about|why\b)",
    ):
        if state.active_topic and state.topic_confidence >= 0.65:
            resolved, intent, risk = _resolved_follow_up(original, state, normalized)
            resolved_topic = (
                "company_package_contents"
                if state.active_topic == "company_pricing" and _has(normalized, r"входит.*цен")
                else state.active_topic
            )
            return DialogueDecision(intent, "verified_rag", risk, True, True, resolved_topic, resolved, None, entities, state.topic_confidence, "state_resolved_follow_up")
        llm_decision = _controller_llm_decision(original, state, language)
        if llm_decision:
            return llm_decision
        return _clarification(original, language, "duration", entities)

    return DialogueDecision("small_talk", "conversational", "low", False, False, None, original, None, entities, 0.7, "safe_non_factual_conversation")
