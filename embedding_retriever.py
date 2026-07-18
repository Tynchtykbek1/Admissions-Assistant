from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")


def find_relevant_chunks_semantic(
    question: str,
    chunks: list[dict],
    top_k: int = 3
) -> list[dict]:
    if not chunks:
        return []

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    scores = np.dot(chunk_embeddings, question_embedding)

    top_indexes = np.argsort(scores)[::-1][:top_k]

    relevant_chunks = []

    for index in top_indexes:
        chunk = chunks[index]

        relevant_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "score": float(scores[index])
        })

    return relevant_chunks