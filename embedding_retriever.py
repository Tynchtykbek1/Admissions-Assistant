from embedding_model import get_embedding_model
import numpy as np


def find_relevant_chunks_semantic(
    question: str,
    chunks: list[dict],
    top_k: int = 3,
    min_score: float = 0.30
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

        relevant_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "score": score
        })

    return relevant_chunks
