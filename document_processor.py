from pathlib import Path
import re
import pdfplumber


def extract_text_from_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def extract_text_from_pdf(file_path: Path) -> str:
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def parse_faq_entries(text: str) -> list[dict]:
    question_pattern = re.compile(
        r"(?m)^\s*(\d+)[.)]\s*(.+?\?)\s*$"
    )
    matches = list(question_pattern.finditer(text))
    faq_entries = []

    for index, match in enumerate(matches):
        answer_start = match.end()
        answer_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        answer = text[answer_start:answer_end].strip()

        if not answer:
            continue

        question = match.group(2).strip()
        faq_entries.append({
            "faq_id": int(match.group(1)),
            "question": question,
            "answer": answer,
            "text_for_retrieval": f"{question}\n{answer}",
            "text": answer
        })

    return faq_entries


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
