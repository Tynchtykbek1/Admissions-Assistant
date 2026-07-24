from embedding_model import get_embedding_model
import numpy as np
import re


EXACT_FAQ_MATCH_BOOST = 2.01
PARTIAL_FAQ_MATCH_BOOST = 0.15

RETRIEVAL_QUERY_REPLACEMENTS = (
    (r"(?<!\w)доки\s+для\s+поступления(?!\w)", "документы для поступления"),
    (r"(?<!\w)виза\s+студента(?!\w)", "студенческая виза"),
    (r"(?<!\w)application\s+docs(?!\w)", "application documents"),
    (r"(?<!\w)доки(?!\w)", "документы"),
    (r"(?<!\w)универ(?!\w)", "университет"),
    (r"(?<!\w)поступать(?!\w)", "поступление"),
    (r"(?<!\w)подача(?!\w)(?!\s+документ)", "подача документов"),
    (r"(?<!\w)апостильнуть(?!\w)", "апостилировать"),
    (r"(?<!\w)docs(?!\w)", "documents"),
    (r"(?<!\w)uni(?!\w)", "university"),
)


def normalize_retrieval_query(text: str) -> str:
    """Expand a small set of admissions terms for embedding and search only."""
    normalized = " ".join(text.split())
    for pattern, replacement in RETRIEVAL_QUERY_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


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
        ranked_candidates.append({
            "index": index,
            "original_query": question,
            "retrieval_query": retrieval_query,
            "score": float(semantic_score),
            "faq_match_type": match_type,
            "faq_match_boost": match_boost,
            "final_score": float(semantic_score) + match_boost,
            "source": chunk["filename"],
            "faq_id": chunk.get("faq_id"),
            "question": chunk.get("question"),
            "preview": chunk["text"][:200]
        })

    ranked_candidates.sort(key=lambda candidate: candidate["final_score"], reverse=True)
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = rank
    return ranked_candidates


def find_relevant_chunks_semantic(
    question: str,
    chunks: list[dict],
    top_k: int = 3,
    min_score: float = 0.30,
    min_context_chunks: int = 0
) -> list[dict]:
    if not chunks:
        return []

    ranked_candidates = build_retrieval_diagnostics(question, chunks)
    eligible_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate["score"] >= min_score or candidate["faq_match_type"] == "exact"
    ]
    top_candidates = eligible_candidates[:top_k]

    relevant_chunks = []

    for candidate in top_candidates:
        index = candidate["index"]
        score = candidate["score"]

        chunk = chunks[index]

        result = {
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "score": score,
            "faq_match_type": candidate["faq_match_type"],
            "faq_match_boost": candidate["faq_match_boost"],
            "final_score": candidate["final_score"],
            "retrieval_fallback": False
        }

        if "faq_id" in chunk:
            result["faq_id"] = chunk["faq_id"]

        relevant_chunks.append(result)

    fallback_limit = min(min_context_chunks, top_k, len(chunks))

    if len(relevant_chunks) < fallback_limit:
        existing_ids = {chunk["chunk_id"] for chunk in relevant_chunks}

        for candidate in ranked_candidates:
            index = candidate["index"]
            chunk = chunks[index]

            if chunk["chunk_id"] in existing_ids:
                continue

            result = {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "text": chunk["text"],
                "score": candidate["score"],
                "faq_match_type": candidate["faq_match_type"],
                "faq_match_boost": candidate["faq_match_boost"],
                "final_score": candidate["final_score"],
                "retrieval_fallback": True
            }

            if "faq_id" in chunk:
                result["faq_id"] = chunk["faq_id"]

            relevant_chunks.append(result)
            existing_ids.add(chunk["chunk_id"])

            if len(relevant_chunks) == fallback_limit:
                break

    return relevant_chunks
