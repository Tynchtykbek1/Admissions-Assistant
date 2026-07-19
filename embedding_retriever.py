from embedding_model import get_embedding_model
import numpy as np


def find_relevant_chunks_semantic(
    question: str,
    chunks: list[dict],
    top_k: int = 3,
    min_score: float = 0.30,
    min_context_chunks: int = 0
) -> list[dict]:
    if not chunks:
        return []

    model = get_embedding_model()

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    chunk_embeddings = np.stack([
        chunk["embedding"] for chunk in chunks
    ])

    scores = np.dot(chunk_embeddings, question_embedding)

    top_indexes = np.argsort(scores)[::-1][:top_k]

    relevant_chunks = []

    for index in top_indexes:
        score = float(scores[index])

        if score < min_score:
            continue

        chunk = chunks[index]

        result = {
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "score": score,
            "retrieval_fallback": False
        }

        if "faq_id" in chunk:
            result["faq_id"] = chunk["faq_id"]

        relevant_chunks.append(result)

    fallback_limit = min(min_context_chunks, top_k, len(chunks))

    if len(relevant_chunks) < fallback_limit:
        existing_ids = {chunk["chunk_id"] for chunk in relevant_chunks}

        for index in top_indexes:
            chunk = chunks[index]

            if chunk["chunk_id"] in existing_ids:
                continue

            result = {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "text": chunk["text"],
                "score": float(scores[index]),
                "retrieval_fallback": True
            }

            if "faq_id" in chunk:
                result["faq_id"] = chunk["faq_id"]

            relevant_chunks.append(result)
            existing_ids.add(chunk["chunk_id"])

            if len(relevant_chunks) == fallback_limit:
                break

    return relevant_chunks
