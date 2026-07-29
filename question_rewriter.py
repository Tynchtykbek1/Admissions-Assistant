import os
import re
import logging
from dataclasses import dataclass

from app_settings import QUESTION_REWRITE_ENABLED
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
REWRITE_HISTORY_CHARACTER_LIMIT = 2000


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
    """Select the most recent user/assistant pair within a small prompt budget."""
    pairs = []
    pending_assistant = None
    for message in reversed(history):
        if message.get("role") == "assistant" and pending_assistant is None:
            pending_assistant = message
        elif message.get("role") == "user":
            pair = [message]
            if pending_assistant is not None:
                pair.append(pending_assistant)
            pairs.append(pair)
            pending_assistant = None
            if len(pairs) == 1:
                break
    selected = [message for pair in reversed(pairs) for message in pair]
    while selected and len(_history_text(selected)) > REWRITE_HISTORY_CHARACTER_LIMIT:
        selected.pop(0)
    return selected


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
        return RewriteResult(question, False)
    rewritten = _valid_rewrite(question, candidate)
    if rewritten is None:
        return RewriteResult(question, False)
    return RewriteResult(rewritten, rewritten != question)
