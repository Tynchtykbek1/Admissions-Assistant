import os
import re
import logging
from dataclasses import dataclass

from app_settings import (
    QUESTION_REWRITE_ENABLED,
    REWRITE_HISTORY_CHARACTER_LIMIT,
    REWRITE_HISTORY_MESSAGE_LIMIT,
)
from llm_answer_generator import generate_provider_text


logger = logging.getLogger(__name__)
REWRITE_INSTRUCTIONS = """
Rewrite the current user message as one standalone retrieval question using recent
history only to resolve references and omitted context. Do not answer it. Do not add
facts or invent university names, contacts, deadlines, requirements, costs, or
procedures. If it is already standalone, return it unchanged. Keep the language of
the current message. Words such as это, там, тогда, он, она, они, какие, кому,
что дальше, а сколько, а раньше may refer to prior turns. If history is insufficient,
return the original question. Return only the rewritten question.
""".strip()

ADMISSIONS_SUBJECT_PATTERNS = (
    r"\b(?:дедлайн\w*|срок\w*|подач\w*|заявк\w*|документ\w*|доки|виз\w*|"
    r"ielts|апостил\w*|стипенди\w*|рекомендац\w*|мотивацион\w*|"
    r"стоимост\w*|оплат\w*|контракт\w*|зачислен\w*|поступлен\w*)\b",
    r"\b(?:deadline\w*|application|apply|document\w*|docs|visa|ielts|"
    r"apostille|scholarship\w*|recommendation|motivation|tuition|enrollment)\b",
)
UNRESOLVED_REFERENCE_PATTERNS = (
    r"\b(?:это|этот|эта|эту|эти|них|ними|потом|после\s+этого|эта\s+цена|эту\s+цену)\b",
    r"^(?:а\s+)?(?:какие\s+гарантии|какие\s+документы|документы|для\s+магистратуры|что\s+дальше|куда\s+потом\s+подавать|"
    r"это\s+обязательно|какие\s+именно|что\s+из\s+них|если\s+откажут|после\s+этого|сколько\s+это\s+занимает)\??$",
    r"^(?:and\s+what\s+is\s+included|what\s+about\s+guarantees|is\s+that\s+mandatory|"
    r"how\s+long\s+does\s+it\s+take)\??$",
    r"^(?:спасибо(?:\s+большое)?[,\s]+)?(?:а\s+)?какие\s+гарантии\??$",
    r"^(?:thanks?[,\s]+)?(?:is\s+the\s+visa\s+guaranteed|what\s+about\s+guarantees)\??$",
    r"^(?:почему|и\s+дальше|а\s+в\s+[^?]+|нет[,]?\s+(?:я\s+про|для)\s+.+|а\s+если\s+отказ)\??[.]?$",
    r"^(?:no[,]?\s+i\s+mean\s+.+)\??[.]?$",
    r"\b(?:это|этот|эта|эти|он|она|они|них|ими|там|тогда|раньше)\b",
    r"^(?:какие именно|что из них|а кому(?:\s+надо)?\s+написать|а сколько|сколько|а на каком языке|"
    r"а раньше можно|а после приезда|что после приезда)\??$",
    r"^(?:which ones|is it mandatory|how much|what about after arrival)\??$",
    r"\b(?:it|this|that|these|those|they|them|there|then)\b",
)
QUALIFIED_CONNECTOR_PATTERNS = (
    r"^(?:а|и)\s+(?:сроки|для визы|раньше можно|на каком языке|после приезда|сколько)\??$",
    r"^(?:what about)\s+(?:the\s+)?(?:visa|deadline|after arrival)\??$",
    r"^(?:and)\s+(?:the\s+)?deadline\??$",
)
@dataclass(frozen=True)
class RewriteResult:
    standalone_question: str
    rewrite_used: bool


def has_standalone_admissions_subject(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    return any(re.search(pattern, normalized) for pattern in ADMISSIONS_SUBJECT_PATTERNS)


def is_likely_follow_up(question: str, history: list[dict] | None = None) -> bool:
    normalized = " ".join(question.casefold().split())
    if not normalized or not history:
        return False

    has_unresolved_reference = any(
        re.search(pattern, normalized) for pattern in UNRESOLVED_REFERENCE_PATTERNS
    )
    has_qualified_connector = any(
        re.search(pattern, normalized) for pattern in QUALIFIED_CONNECTOR_PATTERNS
    )
    if has_unresolved_reference:
        return True
    if has_qualified_connector:
        return True
    if has_standalone_admissions_subject(normalized):
        return False
    return False


def select_rewrite_history(history: list[dict]) -> list[dict]:
    """Return valid recent messages in order within message and character budgets."""
    valid = [
        {"role": message.get("role"), "content": message.get("content", "").strip()}
        for message in history
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ][-REWRITE_HISTORY_MESSAGE_LIMIT:]
    while valid and len(_history_text(valid)) > REWRITE_HISTORY_CHARACTER_LIMIT:
        valid.pop(0)
    return valid


def _history_text(history: list[dict]) -> str:
    return "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in history
    )


def _valid_rewrite(original: str, candidate: object) -> str | None:
    if not isinstance(candidate, str):
        return None
    rewritten = " ".join(candidate.strip().split())
    if not rewritten or len(rewritten) > 500:
        return None
    if "\n" in candidate.strip() or rewritten.startswith(("{", "[", "•", "- ")):
        return None
    if re.match(
        r"^(?:ответ\s*:|вам нужно\b|нужно(?!\s+ли\b)|you should\b|"
        r"the answer\b|according to\b)",
        rewritten.casefold(),
    ):
        return None
    if len(rewritten) > max(500, len(original) * 8):
        return None
    return rewritten


def _deterministic_rewrite(question: str, history: list[dict]) -> RewriteResult:
    selected = select_rewrite_history(history)
    topic_start = None
    for index in range(len(selected)):
        message = selected[index]
        if message["role"] != "user":
            continue
        candidate = message["content"].strip()
        if not is_likely_follow_up(candidate, selected[:index]):
            topic_start = index
    if topic_start is None:
        return RewriteResult(question, False)
    topic_turns = [
        message["content"].strip()
        for message in selected[topic_start:]
        if message["role"] == "user" and message["content"].strip()
    ]
    anchor = " / ".join(topic_turns)
    if re.search(r"[А-Яа-яЁё]", question):
        standalone = f"{question.rstrip()} В контексте предыдущего вопроса: {anchor}"
    else:
        standalone = f"{question.rstrip()} In the context of the previous question: {anchor}"
    return RewriteResult(standalone[:500], True)


def rewrite_question(question: str, history: list[dict]) -> RewriteResult:
    if (
        not QUESTION_REWRITE_ENABLED
        or not history
        or not is_likely_follow_up(question, history)
    ):
        return RewriteResult(question, False)

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_override = os.getenv("QUESTION_REWRITE_MODEL", "").strip() or None
    rewrite_history = select_rewrite_history(history)
    prompt = (
        f"Recent history:\n{_history_text(rewrite_history)}\n\n"
        f"Current user message:\n{question}"
    )
    try:
        candidate = generate_provider_text(
            provider,
            REWRITE_INSTRUCTIONS,
            prompt,
            model_override=model_override,
        )
    except Exception as error:
        logger.warning(
            "Question rewrite failed safely category=%s.", type(error).__name__
        )
        return _deterministic_rewrite(question, history)
    rewritten = _valid_rewrite(question, candidate)
    if rewritten is None:
        return _deterministic_rewrite(question, history)
    return RewriteResult(rewritten, rewritten != question)
