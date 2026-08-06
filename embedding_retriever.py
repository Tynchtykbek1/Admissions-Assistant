from embedding_model import get_embedding_model
import numpy as np
import re


EXACT_FAQ_MATCH_BOOST = 2.01
PARTIAL_FAQ_MATCH_BOOST = 0.15
MAX_LEXICAL_RANKING_BONUS = 0.20

RETRIEVAL_QUERY_REPLACEMENTS = (
    (r"(?<!\w)доки\s+для\s+визы(?!\w)", "документы для студенческой визы"),
    (r"(?<!\w)доки\s+для\s+поступления(?!\w)", "документы для поступления"),
    (r"(?<!\w)дедлайны?(?!\w)", r"\g<0> сроки подачи документов сроки подачи заявления"),
    (r"(?<!\w)когда\s+подаваться(?!\w)", r"\g<0> сроки подачи заявления период подачи"),
    (r"(?<!\w)когда\s+подавать(?!\w)", r"\g<0> сроки подачи документов период подачи"),
    (r"(?<!\w)сроки(?!\w)", r"\g<0> сроки подачи заявления"),
    (r"(?<!\w)заявка(?!\w)", r"\g<0> заявление на поступление"),
    (r"(?<!\w)подача(?!\w)", r"\g<0> подача документов заявление"),
    (r"(?<!\w)доки(?!\w)", "документы"),
    (r"(?<!\w)студенческ(?:ая|ой)\s+виз(?:а|ы)(?!\w)|(?<!\w)виз(?:а|ы)\s+студента(?!\w)|(?<!\w)виз(?:а|ы)(?!\w)", "студенческая виза"),
    (r"(?<!\w)апостилировать(?!\w)|(?<!\w)апостильнуть(?!\w)|(?<!\w)апостиль(?!\w)", r"\g<0> апостиль документов"),
    (r"(?<!\w)ielts(?!\w)", r"\g<0> языковой сертификат language certificate"),
    (r"(?<!\w)языковой\s+сертификат(?!\w)", r"\g<0> подтверждение языка"),
    (r"(?<!\w)стипендия(?!\w)|(?<!\w)scholarship(?!\w)", r"\g<0> стипендия scholarship"),
    (r"(?<!\w)рекомендация(?!\w)|(?<!\w)рекомендательное\s+письмо(?!\w)", r"\g<0> рекомендательное письмо"),
    (r"(?<!\w)мотивационное\s+письмо(?!\w)", r"\g<0> motivation letter"),
    # Do not expand every price/payment query toward tuition: that caused
    # company-service questions to retrieve neighbouring university fees.
    (r"(?<!\w)стоимость\s+обучения(?!\w)|(?<!\w)оплата\s+обучения(?!\w)", r"\g<0> tuition fee payment"),
    (r"(?<!\w)зачисление(?!\w)", r"\g<0> поступление enrollment"),
    (r"(?<!\w)application\s+period(?!\w)|(?<!\w)when\s+can\s+i\s+apply(?!\w)", r"\g<0> application deadlines application period"),
    (r"(?<!\w)deadlines?(?!\w)", r"\g<0> application period submission deadline"),
    (r"(?<!\w)required\s+documents(?!\w)|(?<!\w)docs(?!\w)", r"\g<0> required documents"),
    (r"(?<!\w)student\s+visa(?!\w)", r"\g<0> student visa documents"),
    (r"(?<!\w)apostille(?!\w)", r"\g<0> document apostille"),
    (r"(?<!\w)language\s+certificate(?!\w)", r"\g<0> language proficiency certificate"),
    (r"(?<!\w)recommendation\s+letter(?!\w)", r"\g<0> letter of recommendation"),
    (r"(?<!\w)motivation\s+letter(?!\w)", r"\g<0> statement of motivation"),
    (r"(?<!\w)tuition(?!\w)", r"\g<0> tuition fee payment"),
    (r"(?<!\w)enrollment(?!\w)", r"\g<0> admission enrollment"),
)

CANONICAL_QUERY_REPLACEMENTS = (
    (r"(?<!\w)доки\s+для\s+поступления(?!\w)", "документы для поступления"),
    (r"(?<!\w)application\s+docs(?!\w)", "application documents"),
    (r"(?<!\w)доки(?!\w)", "документы"),
    (r"(?<!\w)docs(?!\w)", "documents"),
    (r"(?<!\w)универ(?!\w)", "университет"),
    (r"(?<!\w)поступать(?!\w)", "поступление"),
    (r"(?<!\w)апостильнуть(?!\w)", "апостилировать"),
    (r"(?<!\w)uni(?!\w)", "university"),
)

LEXICAL_STOPWORDS = frozenset({
    "а", "и", "в", "во", "для", "ли", "на", "нужен", "нужна", "нужны", "какие",
    "что", "это", "есть", "по", "при", "the", "a", "an", "and", "are", "can",
    "do", "for", "i", "is", "of", "what", "when", "required", "needed",
    "документы", "поступление", "application", "admission",
})


def normalize_retrieval_query(text: str) -> str:
    """Expand a small set of admissions terms for embedding and search only."""
    normalized = " ".join(text.split())
    for pattern, replacement in CANONICAL_QUERY_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    # Match every rule against the original normalized text so expansions cannot
    # recursively trigger later rules.
    additions = []
    for pattern, replacement in RETRIEVAL_QUERY_REPLACEMENTS:
        expanded = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        if expanded != normalized:
            additions.append(expanded)
    if not additions:
        return normalized
    added_terms = []
    normalized_tokens = normalized.casefold().split()
    for expanded in additions:
        for token in expanded.split():
            if token.casefold() not in normalized_tokens and token.casefold() not in {
                item.casefold() for item in added_terms
            }:
                added_terms.append(token)
    return " ".join([normalized, *added_terms])


def normalized_lexical_tokens(text: str | None) -> set[str]:
    tokens = set(re.findall(r"[^\W_]+", (text or "").casefold(), flags=re.UNICODE))
    return {token for token in tokens if len(token) > 1 and token not in LEXICAL_STOPWORDS}


def calculate_lexical_score(query: str, chunk: dict) -> tuple[float, float, float]:
    """Return bounded total, question, and answer token-overlap scores."""
    query_tokens = normalized_lexical_tokens(query)
    if not query_tokens:
        return 0.0, 0.0, 0.0
    question_overlap = len(query_tokens & normalized_lexical_tokens(chunk.get("question", ""))) / len(query_tokens)
    answer_overlap = len(query_tokens & normalized_lexical_tokens(chunk.get("answer", ""))) / len(query_tokens)
    score = min(1.0, question_overlap * 0.75 + answer_overlap * 0.25)
    return score, question_overlap, answer_overlap


def normalize_faq_question(text: str) -> str:
    """Normalize a user or FAQ question for deterministic FAQ matching."""
    without_parentheses = re.sub(r"\([^)]*\)", " ", text.casefold())
    letters_and_numbers = re.sub(r"[^\w]+", " ", without_parentheses, flags=re.UNICODE)
    return " ".join(letters_and_numbers.split())


def _get_faq_match_for_text(query: str, chunk: dict) -> tuple[str | None, float]:
    if "faq_id" not in chunk or not chunk.get("question"):
        return None, 0.0

    normalized_query = normalize_faq_question(query)
    normalized_question = normalize_faq_question(chunk["question"])

    if not normalized_query or not normalized_question:
        return None, 0.0
    if normalized_query == normalized_question:
        return "exact", EXACT_FAQ_MATCH_BOOST
    if (
        normalized_query.startswith(normalized_question)
        or normalized_question.startswith(normalized_query)
        or normalized_query in normalized_question
        or normalized_question in normalized_query
    ):
        return "partial", PARTIAL_FAQ_MATCH_BOOST
    return None, 0.0


def get_faq_match(
    query: str,
    chunk: dict,
    retrieval_query: str | None = None
) -> tuple[str | None, float]:
    """Use original text first, while allowing expanded text to improve matching."""
    matches = [_get_faq_match_for_text(query, chunk)]
    if retrieval_query and retrieval_query != query:
        matches.append(_get_faq_match_for_text(retrieval_query, chunk))

    return max(matches, key=lambda match: match[1])


def build_retrieval_diagnostics(
    question: str,
    chunks: list[dict]
) -> list[dict]:
    """Return complete ranked candidates without applying top-K or thresholds."""
    if not chunks:
        return []

    retrieval_query = normalize_retrieval_query(question)
    model = get_embedding_model()
    question_embedding = model.encode(
        retrieval_query,
        normalize_embeddings=True
    )
    chunk_embeddings = np.stack([chunk["embedding"] for chunk in chunks])
    scores = np.dot(chunk_embeddings, question_embedding)
    ranked_candidates = []

    for index, semantic_score in enumerate(scores):
        chunk = chunks[index]
        match_type, match_boost = get_faq_match(
            question,
            chunk,
            retrieval_query=retrieval_query
        )
        lexical_score, question_overlap, answer_overlap = calculate_lexical_score(
            retrieval_query, chunk
        )
        lexical_bonus = min(MAX_LEXICAL_RANKING_BONUS, lexical_score * MAX_LEXICAL_RANKING_BONUS)
        ranked_candidates.append({
            "index": index,
            "original_query": question,
            "retrieval_query": retrieval_query,
            "score": float(semantic_score),
            "faq_match_type": match_type,
            "faq_match_boost": match_boost,
            "lexical_score": lexical_score,
            "lexical_question_overlap": question_overlap,
            "lexical_answer_overlap": answer_overlap,
            "lexical_bonus": lexical_bonus,
            "final_score": float(semantic_score) + match_boost + lexical_bonus,
            "source": chunk["filename"],
            "faq_id": chunk.get("faq_id"),
            "question": chunk.get("question"),
            "preview": chunk["text"][:200]
        })

    ranked_candidates.sort(key=lambda candidate: (
        -candidate["final_score"],
        -candidate["score"],
        candidate["faq_id"] if candidate["faq_id"] is not None else chunks[candidate["index"]]["chunk_id"],
    ))
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = rank
    return ranked_candidates


def find_relevant_chunks_semantic(
    question: str,
    chunks: list[dict],
    top_k: int = 3,
    min_score: float = 0.30,
    min_context_chunks: int = 0,
    fallback_score_threshold: float | None = None,
    context_score_margin: float = 0.12,
    intent: str | None = None,
    risk_level: str = "medium",
) -> list[dict]:
    if not chunks:
        return []

    # Existing callers retain semantic-only behavior. Conversation routing
    # supplies an intent and activates Retrieval v2 through this compatible API.
    if intent is not None:
        from retrieval_reranker import (
            covered_query_categories,
            infer_query_categories,
            retrieve_relevant_chunks,
        )

        result = retrieve_relevant_chunks(
            question,
            chunks,
            intent=intent,
            risk_level=risk_level,
            top_k=top_k,
        )
        relevant = RetrievalChunkList(result.chunks)
        query_categories = infer_query_categories(question, intent)
        covered_categories = covered_query_categories(query_categories, result.selected)
        relevant.diagnostics = {
            "candidate_count": len(result.candidates),
            "max_candidate_semantic_score": max(
                (candidate.semantic_score for candidate in result.candidates),
                default=None,
            ),
            "semantic_candidate_count": result.semantic_candidate_count,
            "lexical_candidate_count": result.lexical_candidate_count,
            "selected_count": len(result.selected),
            "selected_faq_ids": [
                candidate.chunk.get("faq_id") for candidate in result.selected
                if candidate.chunk.get("faq_id") is not None
            ],
            "semantic_scores": [round(candidate.semantic_score, 4) for candidate in result.selected],
            "lexical_scores": [round(candidate.lexical_score, 4) for candidate in result.selected],
            "final_scores": [round(candidate.final_score, 4) for candidate in result.selected],
            "inferred_categories": [list(candidate.inferred_categories) for candidate in result.selected],
            "knowledge_scopes": sorted({
                candidate.chunk.get("knowledge_scope")
                for candidate in result.selected
                if candidate.chunk.get("knowledge_scope")
            }),
            "applied_penalties": sorted({
                penalty
                for candidate in result.candidates
                for penalty in candidate.penalties
            }),
            "retrieval_strategy": result.retrieval_strategy,
            "retrieval_confidence": result.retrieval_confidence,
            "query_categories": sorted(query_categories),
            "covered_query_categories": sorted(covered_categories),
            "missing_query_categories": sorted(query_categories - covered_categories),
        }
        return relevant

    ranked_candidates = build_retrieval_diagnostics(question, chunks)
    eligible_candidates = []
    for candidate in ranked_candidates:
        is_primary = (
            candidate["score"] >= min_score
            or candidate["faq_match_type"] == "exact"
        )
        is_fallback = (
            not is_primary
            and fallback_score_threshold is not None
            and candidate["score"] >= fallback_score_threshold
        )
        if is_primary or is_fallback:
            eligible_candidates.append((candidate, is_fallback))

    if not np.isfinite(context_score_margin) or not 0.0 <= context_score_margin <= 1.0:
        raise ValueError("context_score_margin must be a finite value between 0 and 1")

    selected_candidates = []
    if eligible_candidates:
        best_final_score = eligible_candidates[0][0]["final_score"]
        selected_candidates = [
            item for position, item in enumerate(eligible_candidates)
            if position == 0
            or item[0]["faq_match_type"] == "exact"
            or item[0]["final_score"] >= best_final_score - context_score_margin
        ]

    relevant_chunks = []
    seen_chunk_ids = set()

    for candidate, is_fallback in selected_candidates:
        index = candidate["index"]
        score = candidate["score"]

        chunk = chunks[index]
        if chunk["chunk_id"] in seen_chunk_ids:
            continue

        result = {
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "score": score,
            "faq_match_type": candidate["faq_match_type"],
            "faq_match_boost": candidate["faq_match_boost"],
            "final_score": candidate["final_score"],
            "lexical_score": candidate["lexical_score"],
            "lexical_bonus": candidate["lexical_bonus"],
            "retrieval_fallback": is_fallback,
        }

        if "faq_id" in chunk:
            result["faq_id"] = chunk["faq_id"]
        for field in ("question", "answer", "text_for_retrieval"):
            if chunk.get(field) is not None:
                result[field] = chunk[field]

        relevant_chunks.append(result)
        seen_chunk_ids.add(chunk["chunk_id"])
        if len(relevant_chunks) == top_k:
            break

    return relevant_chunks


class RetrievalChunkList(list):
    """List-compatible selected context carrying safe internal diagnostics."""

    diagnostics: dict
