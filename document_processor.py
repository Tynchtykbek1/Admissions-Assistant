from pathlib import Path


def extract_text_from_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text

def split_text_into_chunks(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks