from pathlib import Path


def extract_text_from_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks