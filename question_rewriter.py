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

FOLLOW_UP_PATTERNS = (
    r"^(?:а|и)\b",
    r"\b(?:это|этот|эта|эти|там|тогда|он|она|они|им|ему|ей)\b",
    r"^(?:какие именно|кому|что дальше|а сколько|сколько это|это обязательно)\b",
    r"^(?:and|but|what about|which ones|who|then|how much|is it)\b",
    r"\b(?:it|this|that|they|them|there|then)\b",
)


@dataclass(frozen=True)
class RewriteResult:
    standalone_question: str
    rewrite_used: bool


def is_likely_follow_up(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in FOLLOW_UP_PATTERNS)


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
        or not is_likely_follow_up(question)
    ):
        return RewriteResult(question, False)

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_override = os.getenv("QUESTION_REWRITE_MODEL", "").strip() or None
    prompt = (
        f"Recent history:\n{_history_text(history)}\n\n"
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
