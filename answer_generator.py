import re
import numpy as np
from embedding_model import get_embedding_model


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def generate_basic_answer(question: str, relevant_chunks: list[dict]) -> str:
    if not relevant_chunks:
        return "There is not enough information in the uploaded document to answer this question."

    candidate_sentences = []

    for chunk in relevant_chunks:
        sentences = split_into_sentences(chunk["text"])
        candidate_sentences.extend(sentences)

    if not candidate_sentences:
        return (
            "Based on the uploaded document, I found relevant information, "
            "but I could not extract a short answer clearly."
        )

    model = get_embedding_model()

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    sentence_embeddings = model.encode(
        candidate_sentences,
        normalize_embeddings=True
    )

    scores = np.dot(sentence_embeddings, question_embedding)

    top_indexes = np.argsort(scores)[::-1][:1]

    selected_sentences = []
    seen_sentences = set()

    for index in top_indexes:
        sentence = candidate_sentences[index]

        if sentence not in seen_sentences:
            selected_sentences.append(sentence)
            seen_sentences.add(sentence)

    answer_text = " ".join(selected_sentences)

    return f"Based on the uploaded document: {answer_text}"
