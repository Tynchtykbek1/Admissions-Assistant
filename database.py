import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DATABASE_PATH = Path("admissions.db")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                document_type TEXT,
                uploaded_at TEXT
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_id INTEGER,
                faq_id INTEGER NULL,
                question TEXT NULL,
                answer TEXT NULL,
                text TEXT,
                text_for_retrieval TEXT,
                embedding TEXT,
                filename TEXT
            )
        """)


def insert_document(filename: str, document_type: str) -> int:
    uploaded_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO documents (filename, document_type, uploaded_at) VALUES (?, ?, ?)",
            (filename, document_type, uploaded_at)
        )
        return cursor.lastrowid


def insert_chunk(document_id: int, chunk: dict) -> None:
    embedding_json = json.dumps(chunk["embedding"].tolist())

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO chunks (
                document_id, chunk_id, faq_id, question, answer,
                text, text_for_retrieval, embedding, filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document_id,
            chunk["chunk_id"],
            chunk.get("faq_id"),
            chunk.get("question"),
            chunk.get("answer"),
            chunk["text"],
            chunk["text_for_retrieval"],
            embedding_json,
            chunk["filename"]
        ))


def load_latest_document() -> list[dict]:
    with get_connection() as connection:
        latest_document = connection.execute(
            "SELECT id FROM documents ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if latest_document is None:
            return []

        rows = connection.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY id",
            (latest_document["id"],)
        ).fetchall()

    chunks = []

    for row in rows:
        chunk = {
            "chunk_id": row["chunk_id"],
            "filename": row["filename"],
            "text": row["text"],
            "text_for_retrieval": row["text_for_retrieval"],
            "embedding": json.loads(row["embedding"])
        }

        if row["faq_id"] is not None:
            chunk["faq_id"] = row["faq_id"]
            chunk["question"] = row["question"]
            chunk["answer"] = row["answer"]

        chunks.append(chunk)

    return chunks


def clear_document(document_id: int) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
