import re


STOP_WORDS = {
    "what", "do", "does", "i", "you", "we", "they",
    "is", "are", "am", "the", "a", "an",
    "to", "for", "of", "in", "on", "at", "by",
    "and", "or", "with", "from", "as",
    "can", "could", "should", "would",
    "need", "tell", "me", "my", "your"
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"\b\w+\b", text.lower())

    meaningful_words = {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }

    return meaningful_words


def find_relevant_chunks(question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    question_words = tokenize(question)
    scored_chunks = []

    for chunk in chunks:
        chunk_words = tokenize(chunk["text"])
        score = len(question_words.intersection(chunk_words))

        if score > 0:
            scored_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "text": chunk["text"],
                "score": score
            })

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return scored_chunks[:top_k]