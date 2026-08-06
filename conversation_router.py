import re
from dataclasses import dataclass

from local_responses import farewell_language, greeting_language, normalize_local_text
from question_rewriter import is_likely_follow_up, rewrite_question


@dataclass(frozen=True)
class ConversationRoute:
    intent: str
    response_mode: str
    risk_level: str
    needs_retrieval: bool
    is_follow_up: bool
    standalone_question: str
    confidence: float | None = None
    rewrite_used: bool = False


_GRATITUDE = re.compile(r"\b(?:спасибо|благодарю|thanks?|thank\s+you)\b", re.I)
_EXPLAIN = re.compile(
    r"\b(?:объясни(?:те)?\s+(?:проще|ещ[её]\s+раз)|повтори(?:те)?|не\s+понял(?:а)?|"
    r"explain\s+(?:it\s+)?(?:more\s+simply|again|проще)|repeat\s+(?:that|it)|i\s+do\s+not\s+understand)\b",
    re.I,
)
_SAFE_DEFINITION = re.compile(
    r"^(?:что\s+такое|что\s+значит|what\s+is(?:\s+an?)?|what\s+does\s+.+\s+mean)\s+"
    r"(?:бакалавриат|магистратура|транскрипт|мотивационн(?:ое|ого)\s+письм[оа]|"
    r"bachelor(?:\s+s)?\s+degree|master(?:\s+s)?\s+degree|transcript|motivation(?:al)?\s+letter)\??$",
    re.I,
)

_CATEGORIES = (
    ("manager_contact", "high", r"\b(?:связат\w*(?:\s+с\s+(?:компани\w*|менеджер\w*))?|контакт\w*|кому\s+написат\w*|кто\s+(?:ваши\s+|главн\w*\s+)?менеджер\w*|contacts?|who\s+can\s+i\s+contact|contact\s+(?:a\s+)?manager)\b"),
    ("rejection_support", "high", r"\b(?:при\s+отказ\w*|после\s+отказ\w*|what\s+happens\s+(?:after|if).*(?:reject|refus))\b"),
    ("onboarding", "high", r"\b(?:начина\w*\s+сотрудничеств\w*|начать\s+сотрудничеств\w*|cooperation\s+begin|start\s+working\s+with)\b"),
    ("client_responsibilities", "high", r"\b(?:обязанност\w*\s+(?:у\s+)?клиент\w*|client(?:'s)?\s+responsibilit)\b"),
    ("company_responsibilities", "high", r"\b(?:обязанност\w*\s+(?:у\s+)?компани\w*|company(?:'s)?\s+responsibilit)\b"),
    ("company_package", "high", r"\b(?:какие\s+(?:есть|существуют)\s+пакет\w*|service\s+packages?\s+(?:are\s+)?available)\b"),
    ("refund", "high", r"\b(?:возврат\w*|возвраща\w*|refund\w*)\b"),
    ("company_pricing", "high", r"\b(?:цен[ауы]|стоимост\w*|сто(?:ит|ят|ите|им|ишь)|оплат\w*|тариф\w*|price|cost|payment)\b"),
    ("company_contract", "high", r"\b(?:договор\w*|контракт\w*|contract|agreement)\b"),
    ("company_guarantees", "high", r"\b(?:гаранти\w*|guarantee\w*)\b"),
    ("visa", "high", r"\b(?:виз\w*|visa\w*)\b"),
    ("scholarship", "high", r"\b(?:стипенди\w*|scholarship\w*|grant\w*)\b"),
    ("documents", "high", r"\b(?:документ\w*|транскрипт\w*|апостил\w*|document\w*|transcript\w*|apostille\w*)\b"),
    ("university_specific", "high", r"\b(?:университет\w*|university|sapienza|bocconi|politecnico|deadline\w*|дедлайн\w*|срок\w*|требовани\w*|requirement\w*)\b"),
    ("company_services", "medium", r"\b(?:сопровожден\w*|услуг\w*|компани\w*|помога\w*|занима\w*|service\w*|company|how\s+do\s+you\s+help)\b"),
    ("admissions_general", "medium", r"\b(?:поступлен\w*|подач\w*|бакалавр\w*|магистрат\w*|admission\w*|application\w*|bachelor\w*|master\w*)\b"),
)


def _category(text: str) -> tuple[str, str] | None:
    for intent, risk, pattern in _CATEGORIES:
        if re.search(pattern, text, re.I):
            return intent, risk
    return None


def route_conversation(
    question: str,
    history: list[dict] | None = None,
    *,
    rewrite_function=rewrite_question,
) -> ConversationRoute:
    history = history or []
    normalized = normalize_local_text(question)
    initial_category = _category(question)
    follow_up = is_likely_follow_up(question, history)
    rewrite = rewrite_function(question, history) if follow_up else None
    standalone = rewrite.standalone_question if rewrite else question
    categorized = _category(standalone) or initial_category
    if follow_up:
        risk = categorized[1] if categorized else "high"
        return ConversationRoute(
            "follow_up", "verified_rag", risk, True, True, standalone, 0.9,
            rewrite.rewrite_used if rewrite else False,
        )
    if _SAFE_DEFINITION.match(normalized):
        return ConversationRoute("admissions_general", "safe_general", "low", False, False, question, 0.98)
    if categorized:
        intent, risk = categorized
        return ConversationRoute(intent, "verified_rag", risk, True, False, question, 0.95)
    if greeting_language(question):
        return ConversationRoute("greeting", "conversational", "low", False, False, question, 1.0)
    if _GRATITUDE.search(normalized) or farewell_language(question):
        return ConversationRoute("gratitude", "conversational", "low", False, False, question, 1.0)
    if _EXPLAIN.search(normalized) or re.fullmatch(r"можно\s+подробнее", normalized, re.I):
        return ConversationRoute("explain_previous", "conversational", "low", False, bool(history), question, 0.95)
    if re.fullmatch(r"(?:как\s+дела|понятно|ясно|хорошо|ладно|ок(?:ей)?|how\s+are\s+you|got\s+it|okay)", normalized, re.I):
        return ConversationRoute("small_talk", "conversational", "low", False, False, question, 0.65)
    return ConversationRoute("unknown", "verified_rag", "medium", True, False, question, 0.5)
