import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from embedding_model import get_embedding_model, get_embedding_model_name


load_dotenv()
logger = logging.getLogger(__name__)


def get_database_path() -> Path:
    database_path = os.getenv("DATABASE_PATH", "").strip()
    return Path(database_path or "admissions.db")



def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(get_database_path())
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with closing(get_connection()) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                document_type TEXT,
                uploaded_at TEXT,
                embedding_model_name TEXT
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

        document_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }

        if "embedding_model_name" not in document_columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN embedding_model_name TEXT"
            )


def insert_document(
    filename: str,
    document_type: str,
    embedding_model_name: str
) -> int:
    uploaded_at = datetime.now(timezone.utc).isoformat()

    with closing(get_connection()) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (
                filename, document_type, uploaded_at, embedding_model_name
            ) VALUES (?, ?, ?, ?)
            """,
            (filename, document_type, uploaded_at, embedding_model_name)
        )
        return cursor.lastrowid


def insert_chunk(document_id: int, chunk: dict) -> None:
    embedding = chunk["embedding"]
    embedding_values = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    embedding_json = json.dumps(embedding_values)

    with closing(get_connection()) as connection, connection:
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
    with closing(get_connection()) as connection, connection:
        latest_document = connection.execute(
            """
            SELECT id, embedding_model_name
            FROM documents
            ORDER BY id DESC
            LIMIT 1
            """
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

    configured_model_name = get_embedding_model_name()
    stored_model_name = latest_document["embedding_model_name"]

    if chunks and stored_model_name != configured_model_name:
        logger.warning(
            "Recomputing embeddings because the stored model differs "
            "from the configured model."
        )
        model = get_embedding_model()
        embeddings = model.encode(
            [chunk["text_for_retrieval"] for chunk in chunks],
            normalize_embeddings=True
        )

        with closing(get_connection()) as connection, connection:
            for row, chunk, embedding in zip(rows, chunks, embeddings):
                chunk["embedding"] = embedding
                connection.execute(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    (json.dumps(embedding.tolist()), row["id"])
                )

            connection.execute(
                "UPDATE documents SET embedding_model_name = ? WHERE id = ?",
                (configured_model_name, latest_document["id"])
            )

    return chunks


def clear_document(document_id: int) -> None:
    with closing(get_connection()) as connection, connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
