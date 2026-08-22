import argparse
import os
import sqlite3
from pathlib import Path

from admissions_rag_assistant.database import get_database_path
from admissions_rag_assistant.embedding_model import get_embedding_model_name


def audit_document_index(document_id: int, database: Path | None = None) -> dict:
    database_path = database or get_database_path()
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        document = connection.execute(
            "SELECT embedding_model_name FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if document is None:
            return {
                "document_exists": False,
                "total_chunks": 0,
                "faq_chunks": 0,
                "chunks_with_question": 0,
                "chunks_with_answer": 0,
                "chunks_with_text_for_retrieval": 0,
                "faq_chunks_missing_required_field": 0,
                "stored_embedding_model_name": None,
                "configured_embedding_model_name": get_embedding_model_name(),
            }
        counts = connection.execute(
            """
            SELECT COUNT(*) total_chunks,
              SUM(CASE WHEN faq_id IS NOT NULL THEN 1 ELSE 0 END) faq_chunks,
              SUM(CASE WHEN question IS NOT NULL AND TRIM(question) <> '' THEN 1 ELSE 0 END) chunks_with_question,
              SUM(CASE WHEN answer IS NOT NULL AND TRIM(answer) <> '' THEN 1 ELSE 0 END) chunks_with_answer,
              SUM(CASE WHEN text_for_retrieval IS NOT NULL AND TRIM(text_for_retrieval) <> '' THEN 1 ELSE 0 END) chunks_with_text_for_retrieval,
              SUM(CASE WHEN faq_id IS NOT NULL AND (
                question IS NULL OR TRIM(question) = '' OR
                answer IS NULL OR TRIM(answer) = '' OR
                text_for_retrieval IS NULL OR TRIM(text_for_retrieval) = ''
              ) THEN 1 ELSE 0 END) faq_chunks_missing_required_field
            FROM chunks WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    return {
        "document_exists": True,
        **{key: int(counts[key] or 0) for key in counts.keys()},
        "stored_embedding_model_name": document["embedding_model_name"],
        "configured_embedding_model_name": get_embedding_model_name(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only document index count audit.")
    parser.add_argument("--document-id", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    report = audit_document_index(parse_args().document_id)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0 if report["document_exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
