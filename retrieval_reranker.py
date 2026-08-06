from __future__ import annotations

from dataclasses import dataclass
import re

from embedding_retriever import build_retrieval_diagnostics, normalized_lexical_tokens
from retrieval_settings import (
    HYBRID_CATEGORY_CONFLICT_PENALTY,
    HYBRID_CONTEXT_SCORE_MARGIN,
    HYBRID_EXACT_QUESTION_BONUS,
    HYBRID_INTENT_MATCH_BONUS,
    HYBRID_LEXICAL_CANDIDATE_LIMIT,
    HYBRID_LEXICAL_WEIGHT,
    HYBRID_MAX_CONTEXT_CHUNKS,
    HYBRID_MIN_FINAL_SCORE,
    HYBRID_SEMANTIC_CANDIDATE_LIMIT,
    HYBRID_SEMANTIC_WEIGHT,
    HYBRID_UNKNOWN_CATEGORY_PENALTY,
    HYBRID_UNRELATED_AMOUNT_PENALTY,
    HYBRID_UNRELATED_GUARANTEE_PENALTY,
    SEMANTIC_FALLBACK_SAFE_MINIMUM,
)


RETRIEVAL_CATEGORIES = frozenset({
    "company_services", "company_pricing", "company_contract",
    "company_guarantees", "refund", "admissions_general",
    "university_specific", "tuition", "documents_university",
    "documents_visa", "visa", "scholarship", "deadline",
    "language_requirement", "tests", "housing", "arrival", "unknown",
    "visa_fee", "financial_means", "scholarship_amount", "document_cost",
    "visa_guarantee", "scholarship_guarantee", "admission_guarantee",
    "housing_cost",
})

_AMOUNT_RE = re.compile(
    r"(?:\b\d[\d\s.,]*\s*(?:€|eur|euro|евро|usd|dollar|доллар|сом|руб)\b|"
    r"\b(?:cost|costs|price|fee|tuition|стоим\w*|цен\w*|сбор\w*|оплат\w*)\b)",
    re.IGNORECASE,
)
_GUARANTEE_RE = re.compile(r"\b(?:гарант\w*|guarantee\w*)\b", re.IGNORECASE)


def _text(chunk: dict) -> str:
    return " ".join(str(chunk.get(field) or "") for field in (
        "question", "answer", "text_for_retrieval", "text"
    )).casefold()


def _has(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _infer_text_categories(text: str) -> frozenset[str]:
    text = text.casefold()
    categories: set[str] = set()
    company = _has(
        text, "компан", "услуг", "сопровожд", "сервис", "пакет",
        "company", "service", "support package", "consult",
    )
    price = _has(
        text, "стои", "стоят", "цен", "оплат", "тариф", "расход",
        "cost", "price", "fee",
    )
    guarantee = bool(_GUARANTEE_RE.search(text))
    visa = _has(text, "виз", "visa")
    documents = _has(text, "документ", "паспорт", "транскрип", "document", "passport", "transcript")
    university = _has(
        text, "университет", "поступ", "абитури", "бакалав", "магистр",
        "university", "admission", "application", "bachelor", "master",
    )
    scholarship = _has(text, "стипенд", "грант", "scholarship", "grant")
    exam = _has(text, "cent", "cisia", "экзам", "тест", "exam", "test fee")
    housing = _has(text, "прожив", "жиль", "общеж", "аренд", "housing", "rent", "accommodation")
    tuition = _has(text, "обучен", "учёб", "учебы", "tuition")
    financial_means = _has(
        text, "на счёт", "на счет", "банковск", "выписк", "финансов", "средств",
        "bank account", "bank statement", "financial means", "proof of funds",
    )
    scholarship_amount = scholarship and _has(text, "сумм", "размер", "amount", "how much")
    visa_fee_phrase = _has(text, "визовый сбор", "сбор за визу", "visa fee")
    visa_fee = visa and (price or visa_fee_phrase) and not financial_means
    document_cost = price and _has(text, "апостил", "легализац", "apostille", "legalization")
    visa_guarantee = visa and (
        guarantee or _has(text, "риск", "шанс", "вероятност", "risk", "chance", "likelihood")
    )
    scholarship_guarantee = scholarship and guarantee
    admission_guarantee = university and guarantee
    housing_cost = housing and price

    if _has(text, "договор", "контракт", "contract", "agreement"):
        categories.add("company_contract")
    if _has(text, "возврат", "refund", "вернуть деньги"):
        categories.add("refund")
    if financial_means:
        categories.add("financial_means")
    if scholarship_amount:
        categories.add("scholarship_amount")
    if visa_fee:
        categories.add("visa_fee")
    if document_cost:
        categories.add("document_cost")
    if visa_guarantee:
        categories.add("visa_guarantee")
    if scholarship_guarantee:
        categories.add("scholarship_guarantee")
    if admission_guarantee:
        categories.add("admission_guarantee")
    if housing_cost:
        categories.add("housing_cost")
    if company and price and not _has(text, "помимо оплаты", "excluding our fee"):
        categories.add("company_pricing")
    if company and guarantee:
        categories.add("company_guarantees")
    if (
        company
        or _has(text, "вы помогаете", "помогаете ли", "поддерживаете связь", "do you help", "how do you help")
    ) and not price and not guarantee:
        categories.add("company_services")
    if scholarship:
        categories.add("scholarship")
    if documents and visa:
        categories.add("documents_visa")
    elif documents and university:
        categories.add("documents_university")
    elif documents and not scholarship and _has(
        text, "транскрип", "рекоменд", "мотивацион", "апостил", "europass",
        "transcript", "recommendation", "motivation", "apostille",
    ):
        categories.add("documents_university")
    if visa and not visa_fee and not financial_means and not visa_guarantee:
        categories.add("visa")
    if tuition:
        categories.add("tuition")
    if exam:
        categories.add("tests")
    if housing and not housing_cost:
        categories.add("housing")
    if _has(text, "дедлайн", "срок подачи", "deadline", "application period") or (
        _has(text, "когда", "when") and _has(text, "пода", "начать", "start", "apply")
    ):
        categories.add("deadline")
    if _has(text, "ielts", "toefl", "язык", "итальянск", "английск", "language requirement", "certificate"):
        categories.add("language_requirement")
    if _has(text, "приезд", "прибыт", "arrival", "after landing"):
        categories.add("arrival")
    if university and not documents and not tuition and not admission_guarantee:
        categories.add("university_specific")
    return frozenset(categories or {"unknown"})


@dataclass(frozen=True)
class ChunkCategoryClassification:
    primary: frozenset[str]
    secondary: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        combined = (self.primary | self.secondary) - {"unknown"}
        return combined or frozenset({"unknown"})


def classify_chunk_categories(chunk: dict) -> ChunkCategoryClassification:
    """Use FAQ question as primary domain and answer only as secondary context."""
    question = chunk.get("question")
    if isinstance(question, str) and question.strip():
        primary = _infer_text_categories(question)
        secondary_text = " ".join(
            str(chunk.get(field) or "") for field in ("answer", "text")
        )
        secondary = _infer_text_categories(secondary_text)
    else:
        primary = _infer_text_categories(_text(chunk))
        secondary = frozenset()
    return ChunkCategoryClassification(primary=primary, secondary=secondary)


def infer_chunk_categories(chunk: dict) -> frozenset[str]:
    """Return all inferred categories without persisting metadata."""
    return classify_chunk_categories(chunk).all


def infer_query_categories(question: str, intent: str) -> frozenset[str]:
    text = question.casefold()
    company = _has(
        text, "компан", "услуг", "сопровожд", "сервис", "пакет", "помогаете",
        "company", "service", "package", "do you help",
    )
    price = _has(text, "стои", "стоят", "цен", "оплат", "cost", "price", "fee") or (
        _has(text, "сколько", "how much") and _has(text, "денег", "money")
    )
    guarantee = bool(_GUARANTEE_RE.search(text))
    documents = _has(text, "документ", "доки", "document", "docs", "паспорт", "passport")
    visa = _has(text, "виз", "visa")
    university = _has(
        text, "университет", "поступ", "бакалав", "магистр", "university",
        "admission", "application", "bachelor", "master",
    )
    service_scope = company and _has(text, "входит", "пакет", "included", "package")

    categories: set[str] = set()
    financial_means = _has(
        text, "на счёт", "на счет", "банковск", "выписк", "финансов", "средств",
        "bank account", "bank statement", "financial means", "proof of funds",
    )
    scholarship = _has(text, "стипенд", "scholarship", "grant")
    amount = _has(text, "сколько", "сумм", "размер", "amount", "how much")

    # Explicit domain entities refine the router's deliberately broad high-risk
    # intent. Imperatives about filters or FAQ IDs are never control signals.
    if _has(text, "возврат", "возвращает", "возвращаете", "refund", "вернуть деньги"):
        categories.add("refund")
    if financial_means:
        categories.add("financial_means")
    if scholarship:
        if guarantee:
            categories.add("scholarship_guarantee")
        else:
            categories.add("scholarship_amount" if amount else "scholarship")
    if _has(text, "дедлайн", "срок подачи", "deadline", "application period") or (
        _has(text, "когда", "when") and _has(text, "пода", "apply", "submit")
    ):
        categories.add("deadline")
    if _has(text, "ielts", "toefl", "языков", "итальянск", "английск", "language requirement", "language certificate"):
        categories.add("language_requirement")
    if documents and visa:
        categories.add("documents_visa")
    elif documents and university:
        categories.add("documents_university")
    if _has(text, "cent", "cisia", "экзам", "exam"):
        categories.add("tests")
    tuition = _has(text, "обучен", "учёб", "учебы", "tuition")
    if tuition and price:
        categories.add("tuition")
    if _has(text, "прожив", "жиль", "аренд", "housing", "rent"):
        categories.add("housing_cost" if price else "housing")
    if visa and price and not financial_means:
        categories.add("visa_fee")
    elif visa and not documents and not service_scope:
        categories.add("visa_guarantee" if guarantee else "visa")
    if guarantee and _has(text, "поступ", "admission", "enrollment"):
        categories.add("admission_guarantee")
    if company and guarantee:
        categories.add("company_guarantees")
    if company and price:
        categories.add("company_pricing")
    if service_scope or (company and not price and not guarantee):
        categories.add("company_services")

    if categories:
        return frozenset(categories)
    if intent == "company_pricing":
        if _has(text, "это", "that", "this") and not company:
            return frozenset({"unknown"})
        return frozenset({"company_pricing"})
    if intent == "company_guarantees":
        return frozenset({"company_guarantees"})
    if intent == "company_contract":
        return frozenset({"company_contract"})
    if intent == "company_services":
        return frozenset({"company_services"})
    if intent == "scholarship":
        return frozenset({"scholarship"})
    if intent == "visa":
        return frozenset({"visa"})
    if intent == "documents":
        return frozenset({"documents_university"}) if university else frozenset({"unknown"})
    if intent == "university_specific":
        return frozenset({"university_specific"})
    if intent == "admissions_general":
        return frozenset({"admissions_general"})
    return frozenset({"unknown"})


def _prefix_tokens(text: str) -> set[str]:
    tokens = normalized_lexical_tokens(text)
    return {token[:5] if len(token) >= 5 else token for token in tokens}


def _lexical_score(question: str, chunk: dict) -> float:
    query = _prefix_tokens(question)
    if not query:
        return 0.0
    question_tokens = _prefix_tokens(str(chunk.get("question") or ""))
    answer_tokens = _prefix_tokens(str(chunk.get("answer") or chunk.get("text") or ""))
    question_overlap = len(query & question_tokens) / len(query)
    answer_overlap = len(query & answer_tokens) / len(query)
    return min(1.0, question_overlap * 0.75 + answer_overlap * 0.25)


def _compatible(target: frozenset[str], categories: frozenset[str]) -> bool:
    if target == {"unknown"}:
        return False
    compatibility = {
        "company_pricing": {"company_pricing"},
        "company_guarantees": {"company_guarantees", "company_contract", "refund"},
        "company_contract": {"company_contract", "refund"},
        "company_services": {"company_services", "company_contract", "company_pricing"},
        "documents_visa": {"documents_visa", "financial_means"},
        "documents_university": {"documents_university"},
        "visa": {"visa", "documents_visa"},
        "tuition": {"tuition"},
        "tests": {"tests"},
        "housing": {"housing"},
        "scholarship": {"scholarship"},
        "deadline": {"deadline"},
        "language_requirement": {"language_requirement"},
        "visa_fee": {"visa_fee"},
        "financial_means": {"financial_means"},
        "scholarship_amount": {"scholarship_amount"},
        "document_cost": {"document_cost"},
        "visa_guarantee": {"visa_guarantee"},
        "scholarship_guarantee": {"scholarship_guarantee"},
        "admission_guarantee": {"admission_guarantee"},
        "housing_cost": {"housing_cost"},
        "university_specific": {"university_specific", "documents_university", "tuition", "deadline", "language_requirement"},
        "admissions_general": {"admissions_general", "university_specific", "documents_university", "deadline", "language_requirement", "tests"},
    }
    return any(categories & compatibility.get(item, {item}) for item in target)


def covered_query_categories(
    target: frozenset[str], candidates: list[RetrievalCandidate]
) -> frozenset[str]:
    """Return query aspects directly supported by selected primary categories."""
    return frozenset(
        category
        for category in target
        if any(
            _compatible(frozenset({category}), frozenset(candidate.primary_categories))
            for candidate in candidates
        )
    )


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: dict
    semantic_score: float
    lexical_score: float
    intent_score: float
    inferred_categories: tuple[str, ...]
    primary_categories: tuple[str, ...]
    secondary_categories: tuple[str, ...]
    penalties: tuple[str, ...]
    final_score: float
    eligible: bool


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[dict]
    candidates: list[RetrievalCandidate]
    selected: list[RetrievalCandidate]
    semantic_candidate_count: int
    lexical_candidate_count: int
    retrieval_strategy: str
    retrieval_confidence: float


def _candidate_key(candidate: RetrievalCandidate) -> tuple:
    faq_id = candidate.chunk.get("faq_id")
    stable_id = faq_id if faq_id is not None else candidate.chunk.get("chunk_id", 0)
    return (-candidate.final_score, -candidate.semantic_score, str(stable_id))


def _chunk_identity(chunk: dict) -> tuple:
    return (
        str(chunk.get("filename") or ""),
        chunk.get("faq_id"),
        chunk.get("chunk_id"),
    )


def _public_chunk(candidate: RetrievalCandidate) -> dict:
    chunk = candidate.chunk
    result = {
        key: value for key, value in chunk.items()
        if key != "embedding"
    }
    result.update({
        "score": candidate.semantic_score,
        "semantic_score": candidate.semantic_score,
        "lexical_score": candidate.lexical_score,
        "intent_score": candidate.intent_score,
        "inferred_categories": list(candidate.inferred_categories),
        "primary_categories": list(candidate.primary_categories),
        "secondary_categories": list(candidate.secondary_categories),
        "applied_penalties": list(candidate.penalties),
        "final_score": candidate.final_score,
        "retrieval_fallback": False,
    })
    return result


def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict],
    *,
    intent: str,
    risk_level: str,
    top_k: int = HYBRID_MAX_CONTEXT_CHUNKS,
) -> RetrievalResult:
    if not chunks:
        return RetrievalResult([], [], [], 0, 0, "hybrid_v2", 0.0)

    diagnostics = build_retrieval_diagnostics(question, chunks)
    by_index = {item["index"]: item for item in diagnostics}
    semantic_indices = {
        item["index"] for item in sorted(diagnostics, key=lambda item: -item["score"])
        [:HYBRID_SEMANTIC_CANDIDATE_LIMIT]
        if item["score"] >= SEMANTIC_FALLBACK_SAFE_MINIMUM or item["faq_match_type"] == "exact"
    }
    lexical_ranked = sorted(
        ((index, _lexical_score(question, chunk)) for index, chunk in enumerate(chunks)),
        key=lambda item: (-item[1], item[0]),
    )
    lexical_indices = {
        index for index, score in lexical_ranked[:HYBRID_LEXICAL_CANDIDATE_LIMIT]
        if score > 0.0
    }
    candidate_indices = semantic_indices | lexical_indices
    target = infer_query_categories(question, intent)
    ambiguous_amount = target == {"unknown"} and bool(_AMOUNT_RE.search(question))
    required_final_score = HYBRID_MIN_FINAL_SCORE + (0.05 if risk_level == "high" else 0.0)
    candidates: list[RetrievalCandidate] = []
    seen_candidate_identities: set[tuple] = set()

    for index in sorted(candidate_indices):
        chunk = chunks[index]
        identity = _chunk_identity(chunk)
        if identity in seen_candidate_identities:
            continue
        seen_candidate_identities.add(identity)
        diagnostic = by_index[index]
        classification = classify_chunk_categories(chunk)
        categories = classification.all
        conservative_unknown_match = target == {"unknown"} and (
            chunk.get("faq_id") is None or diagnostic["faq_match_type"] == "exact"
        )
        compatible = _compatible(target, classification.primary) or conservative_unknown_match
        candidate_text = _text(chunk)
        has_amount = bool(_AMOUNT_RE.search(candidate_text))
        has_guarantee = bool(_GUARANTEE_RE.search(candidate_text))
        penalties: list[str] = []
        hard_conflict = False

        monetary_categories = {
            "company_pricing", "tuition", "visa_fee", "tests", "housing",
            "financial_means", "scholarship_amount", "document_cost",
        }
        if target & monetary_categories and classification.primary & monetary_categories and not compatible:
            penalties.append("unrelated_amount")
            hard_conflict = True
        elif target == {"company_pricing"} and has_amount and not compatible:
            penalties.append("unrelated_amount")
            hard_conflict = True
        if "company_guarantees" in target and has_guarantee and not compatible:
            penalties.append("unrelated_guarantee")
            hard_conflict = True
        if "documents_visa" in target and "documents_university" in classification.primary:
            penalties.append("conflicting_document_category")
            hard_conflict = True
        if "documents_university" in target and "documents_visa" in classification.primary:
            penalties.append("conflicting_document_category")
            hard_conflict = True
        if ambiguous_amount and has_amount:
            penalties.append("ambiguous_amount")
            hard_conflict = True
        if not compatible:
            penalties.append("category_conflict")

        semantic_score = float(diagnostic["score"])
        lexical_score = _lexical_score(question, chunk)
        intent_score = (
            HYBRID_INTENT_MATCH_BONUS
            if compatible and not conservative_unknown_match
            else 0.0
        )
        final_score = (
            semantic_score * HYBRID_SEMANTIC_WEIGHT
            + lexical_score * HYBRID_LEXICAL_WEIGHT
            + intent_score
            + (HYBRID_EXACT_QUESTION_BONUS if diagnostic["faq_match_type"] == "exact" else 0.0)
        )
        if categories == {"unknown"}:
            final_score -= HYBRID_UNKNOWN_CATEGORY_PENALTY
        if not compatible:
            final_score -= HYBRID_CATEGORY_CONFLICT_PENALTY
        if "unrelated_amount" in penalties:
            final_score -= HYBRID_UNRELATED_AMOUNT_PENALTY
        if "unrelated_guarantee" in penalties:
            final_score -= HYBRID_UNRELATED_GUARANTEE_PENALTY

        eligible = not hard_conflict and compatible and final_score >= required_final_score
        candidates.append(RetrievalCandidate(
            chunk=chunk,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            intent_score=intent_score,
            inferred_categories=tuple(sorted(categories)),
            primary_categories=tuple(sorted(classification.primary)),
            secondary_categories=tuple(sorted(classification.secondary)),
            penalties=tuple(penalties),
            final_score=round(final_score, 6),
            eligible=eligible,
        ))

    candidates.sort(key=_candidate_key)
    eligible = [candidate for candidate in candidates if candidate.eligible]
    selected: list[RetrievalCandidate] = []
    if eligible:
        best_score = eligible[0].final_score
        margin_candidates = [
            candidate for candidate in eligible
            if candidate.final_score >= best_score - HYBRID_CONTEXT_SCORE_MARGIN
        ]
        # Preserve multiple explicit aspects by reserving the best compatible
        # candidate for each requested category before filling by score.
        ordered = []
        if len(target) > 1:
            for category in sorted(target):
                match = next(
                    (
                        candidate for candidate in eligible
                        if _compatible(frozenset({category}), frozenset(candidate.primary_categories))
                    ),
                    None,
                )
                if match is not None:
                    ordered.append(match)
        ordered.extend(margin_candidates)
        seen_selected: set[tuple] = set()
        for candidate in ordered:
            identity = _chunk_identity(candidate.chunk)
            if identity in seen_selected:
                continue
            selected.append(candidate)
            seen_selected.add(identity)
            if len(selected) == top_k:
                break
    # final_score is an unbounded ranking utility (exact matches may exceed 1).
    # Confidence is the externally logged normalized form.
    confidence = min(1.0, max(0.0, selected[0].final_score)) if selected else 0.0
    return RetrievalResult(
        chunks=[_public_chunk(candidate) for candidate in selected],
        candidates=candidates,
        selected=selected,
        semantic_candidate_count=len(semantic_indices),
        lexical_candidate_count=len(lexical_indices),
        retrieval_strategy="hybrid_semantic_lexical_intent_v2",
        retrieval_confidence=round(confidence, 6),
    )
