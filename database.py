import json
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from embedding_model import get_embedding_model, get_embedding_model_name


load_dotenv()
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_database_path() -> Path:
    database_path = os.getenv("DATABASE_PATH", "").strip()
    return Path(database_path or "admissions.db")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(get_database_path())
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def initialize_database() -> None:
    """Apply additive, backward-compatible SQLite migrations."""
    with closing(get_connection()) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                document_type TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                embedding_model_name TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                faq_id INTEGER NULL,
                question TEXT NULL,
                answer TEXT NULL,
                text TEXT NOT NULL,
                text_for_retrieval TEXT NOT NULL,
                embedding TEXT NOT NULL,
                filename TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                external_chat_id TEXT NOT NULL,
                external_user_id TEXT NULL,
                active_document_id INTEGER NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(active_document_id) REFERENCES documents(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_chunk
                ON chunks(document_id, chunk_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_external
                ON conversations(channel, external_chat_id, external_user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                ON messages(conversation_id, id);
            """
        )
        _add_column_if_missing(
            connection, "documents", "embedding_model_name", "TEXT"
        )
        _add_column_if_missing(connection, "documents", "original_filename", "TEXT")
        _add_column_if_missing(connection, "documents", "stored_filename", "TEXT")
        connection.execute(
            """
            UPDATE documents
            SET original_filename = COALESCE(original_filename, filename),
                stored_filename = COALESCE(stored_filename, filename)
            WHERE original_filename IS NULL OR stored_filename IS NULL
            """
        )


def insert_document(
    filename: str,
    document_type: str,
    embedding_model_name: str,
    *,
    stored_filename: str | None = None,
    connection: sqlite3.Connection | None = None,
) -> int:
    owns_connection = connection is None
    db = connection or get_connection()
    try:
        cursor = db.execute(
            """
            INSERT INTO documents (
                filename, original_filename, stored_filename, document_type,
                uploaded_at, embedding_model_name
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                filename,
                stored_filename or filename,
                document_type,
                _utc_now(),
                embedding_model_name,
            ),
        )
        if owns_connection:
            db.commit()
        return int(cursor.lastrowid)
    finally:
        if owns_connection:
            db.close()


def insert_chunk(
    document_id: int,
    chunk: dict,
    *,
    connection: sqlite3.Connection | None = None,
) -> None:
    embedding = chunk["embedding"]
    embedding_values = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    owns_connection = connection is None
    db = connection or get_connection()
    try:
        db.execute(
            """
            INSERT INTO chunks (
                document_id, chunk_id, faq_id, question, answer,
                text, text_for_retrieval, embedding, filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                chunk["chunk_id"],
                chunk.get("faq_id"),
                chunk.get("question"),
                chunk.get("answer"),
                chunk["text"],
                chunk["text_for_retrieval"],
                json.dumps(embedding_values),
                chunk["filename"],
            ),
        )
        if owns_connection:
            db.commit()
    finally:
        if owns_connection:
            db.close()


def insert_document_with_chunks(
    filename: str,
    stored_filename: str,
    document_type: str,
    embedding_model_name: str,
    chunks: list[dict],
    *,
    activate_conversation_id: str | None = None,
) -> int:
    """Persist a complete document atomically."""
    with closing(get_connection()) as connection, connection:
        document_id = insert_document(
            filename,
            document_type,
            embedding_model_name,
            stored_filename=stored_filename,
            connection=connection,
        )
        for chunk in chunks:
            insert_chunk(document_id, chunk, connection=connection)
        if activate_conversation_id is not None:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET active_document_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (document_id, _utc_now(), activate_conversation_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Conversation not found.")
        return document_id


def get_document(document_id: int) -> dict | None:
    with closing(get_connection()) as connection:
        row = connection.execute(
            """
            SELECT id, filename, original_filename, stored_filename,
                   document_type, uploaded_at, embedding_model_name
            FROM documents WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def get_latest_document() -> dict | None:
    with closing(get_connection()) as connection:
        row = connection.execute(
            """
            SELECT id, filename, original_filename, stored_filename,
                   document_type, uploaded_at, embedding_model_name
            FROM documents ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
    return dict(row) if row else None


def count_document_chunks(document_id: int) -> int:
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS chunk_count FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return int(row["chunk_count"])


def load_document_chunks(document_id: int) -> list[dict]:
    with closing(get_connection()) as connection:
        document = connection.execute(
            "SELECT embedding_model_name FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if document is None:
            return []
        rows = connection.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY id",
            (document_id,),
        ).fetchall()

    chunks = []
    for row in rows:
        chunk = {
            "chunk_id": row["chunk_id"],
            "filename": row["filename"],
            "text": row["text"],
            "text_for_retrieval": row["text_for_retrieval"],
            "embedding": json.loads(row["embedding"]),
        }
        if row["faq_id"] is not None:
            chunk.update(
                faq_id=row["faq_id"],
                question=row["question"],
                answer=row["answer"],
            )
        chunks.append(chunk)

    configured_model_name = get_embedding_model_name()
    if chunks and document["embedding_model_name"] != configured_model_name:
        logger.warning(
            "Recomputing document embeddings because the configured model changed."
        )
        embeddings = get_embedding_model().encode(
            [chunk["text_for_retrieval"] for chunk in chunks],
            normalize_embeddings=True,
        )
        with closing(get_connection()) as connection, connection:
            for row, chunk, embedding in zip(rows, chunks, embeddings):
                chunk["embedding"] = embedding
                connection.execute(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    (json.dumps(embedding.tolist()), row["id"]),
                )
            connection.execute(
                "UPDATE documents SET embedding_model_name = ? WHERE id = ?",
                (configured_model_name, document_id),
            )
    return chunks


def load_latest_document() -> list[dict]:
    latest = get_latest_document()
    return load_document_chunks(latest["id"]) if latest else []


def get_or_create_conversation(
    channel: str,
    external_chat_id: str,
    external_user_id: str | None = None,
    *,
    conversation_id: str | None = None,
    default_document_id: int | None = None,
) -> dict:
    now = _utc_now()
    with closing(get_connection()) as connection, connection:
        row = None
        if conversation_id:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if (
                row is not None
                and channel == "telegram"
                and (
                    row["channel"] != channel
                    or row["external_chat_id"] != external_chat_id
                    or (row["external_user_id"] or "") != (external_user_id or "")
                )
            ):
                raise ValueError("Conversation identifiers do not match.")
        if row is None:
            row = connection.execute(
                """
                SELECT * FROM conversations
                WHERE channel = ? AND external_chat_id = ?
                  AND COALESCE(external_user_id, '') = COALESCE(?, '')
                ORDER BY created_at LIMIT 1
                """,
                (channel, external_chat_id, external_user_id),
            ).fetchone()
        if row is None:
            new_id = conversation_id or uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO conversations (
                    id, channel, external_chat_id, external_user_id,
                    active_document_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id,
                    channel,
                    external_chat_id,
                    external_user_id,
                    default_document_id,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (new_id,),
            ).fetchone()
    return dict(row)


def get_conversation(conversation_id: str) -> dict | None:
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def update_active_document(conversation_id: str, document_id: int | None) -> None:
    with closing(get_connection()) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET active_document_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (document_id, _utc_now(), conversation_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Conversation not found.")


def synchronize_conversations_active_document(document_id: int) -> int:
    """Point every conversation at one document in a single transaction."""
    now = _utc_now()
    with closing(get_connection()) as connection, connection:
        document = connection.execute(
            "SELECT 1 FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if document is None:
            raise ValueError("Document not found.")
        cursor = connection.execute(
            """
            UPDATE conversations
            SET active_document_id = ?, updated_at = ?
            WHERE active_document_id IS NULL OR active_document_id != ?
            """,
            (document_id, now, document_id),
        )
        return cursor.rowcount


def add_message(conversation_id: str, role: str, content: str) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be user or assistant.")
    if not content.strip():
        raise ValueError("Message content must not be empty.")
    now = _utc_now()
    with closing(get_connection()) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content.strip(), now),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        return int(cursor.lastrowid)


def get_recent_messages(conversation_id: str, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def clear_conversation_messages(conversation_id: str) -> int:
    with closing(get_connection()) as connection, connection:
        cursor = connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utc_now(), conversation_id),
        )
        return cursor.rowcount


def database_is_ready() -> bool:
    try:
        with closing(get_connection()) as connection:
            connection.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def clear_document(document_id: int) -> None:
    with closing(get_connection()) as connection, connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
