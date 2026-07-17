import re
from retriever import tokenize


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def generate_basic_answer(question: str, relevant_chunks: list[dict]) -> str:
    if not relevant_chunks:
        return "I do not have enough information in the uploaded document to answer this question."

    question_words = tokenize(question)
    best_chunk = relevant_chunks[0]
    sentences = split_into_sentences(best_chunk["text"])

    scored_sentences = []

    for sentence in sentences:
        sentence_words = tokenize(sentence)
        score = len(question_words.intersection(sentence_words))

        if score > 0:
            scored_sentences.append({
                "sentence": sentence,
                "score": score
            })

    if not scored_sentences:
        return (
            "Based on the uploaded document, I found relevant information, "
            "but I could not extract a short answer clearly."
        )

    scored_sentences.sort(key=lambda item: item["score"], reverse=True)

    best_sentence = scored_sentences[0]["sentence"]

    return f"Based on the uploaded document: {best_sentence}"