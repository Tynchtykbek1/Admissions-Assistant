import re
from retriever import tokenize


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def generate_basic_answer(question: str, relevant_chunks: list[dict]) -> str:
    if not relevant_chunks:
        return "I do not have enough information in the uploaded document to answer this question."

    question_words = tokenize(question)
    scored_sentences = []

    for chunk in relevant_chunks:
        sentences = split_into_sentences(chunk["text"])

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

    selected_sentences = []
    seen_sentences = set()

    for item in scored_sentences:
        sentence = item["sentence"]

        if sentence not in seen_sentences:
            selected_sentences.append(sentence)
            seen_sentences.add(sentence)

        if len(selected_sentences) == 3:
            break

    answer_text = " ".join(selected_sentences)

    return f"Based on the uploaded document: {answer_text}"