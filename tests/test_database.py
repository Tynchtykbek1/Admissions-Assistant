import json
import os
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import database


class DatabaseCompatibilityTests(unittest.TestCase):
    def test_empty_database_path_uses_default(self):
        with patch.dict(os.environ, {"DATABASE_PATH": ""}):
            self.assertEqual(database.get_database_path(), Path("admissions.db"))

    def test_old_embeddings_are_migrated_and_recomputed(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "old_database.db"
            self.create_old_database(database_path)

            class FakeMultilingualModel:
                def encode(self, texts, normalize_embeddings=True):
                    return np.array([[0.25, 0.75] for _ in texts])

            with patch.dict(os.environ, {"DATABASE_PATH": str(database_path)}):
                database.initialize_database()

                with patch.object(
                    database,
                    "get_embedding_model_name",
                    return_value="new-multilingual-model"
                ), patch.object(
                    database,
                    "get_embedding_model",
                    return_value=FakeMultilingualModel()
                ):
                    chunks = database.load_latest_document()

            self.assertEqual(chunks[0]["embedding"].tolist(), [0.25, 0.75])

            with closing(sqlite3.connect(database_path)) as connection, connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(documents)")
                }
                stored_model = connection.execute(
                    "SELECT embedding_model_name FROM documents"
                ).fetchone()[0]
                stored_embedding = json.loads(
                    connection.execute("SELECT embedding FROM chunks").fetchone()[0]
                )

            self.assertIn("embedding_model_name", columns)
            self.assertIn("original_filename", columns)
            self.assertEqual(stored_model, "new-multilingual-model")
            self.assertEqual(stored_embedding, [0.25, 0.75])

            with closing(sqlite3.connect(database_path)) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                    1,
                )

    def test_conversation_messages_reset_and_active_document(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "conversation.db"
            with patch.dict(os.environ, {"DATABASE_PATH": str(database_path)}):
                database.initialize_database()
                document_id = database.insert_document(
                    "one.txt", "standard", "test-model"
                )
                first = database.get_or_create_conversation(
                    "telegram", "chat-1", "user-1"
                )
                second = database.get_or_create_conversation(
                    "telegram", "chat-2", "user-2"
                )
                database.update_active_document(first["id"], document_id)
                database.add_message(first["id"], "user", "first")
                database.add_message(first["id"], "assistant", "second")
                database.add_message(first["id"], "user", "third")
                database.add_message(second["id"], "user", "keep me")

                recent = database.get_recent_messages(first["id"], 2)
                cleared = database.clear_conversation_messages(first["id"])

                self.assertEqual([item["content"] for item in recent], ["second", "third"])
                self.assertEqual(cleared, 3)
                self.assertEqual(database.get_recent_messages(first["id"], 10), [])
                self.assertEqual(
                    database.get_recent_messages(second["id"], 10)[0]["content"],
                    "keep me",
                )
                self.assertEqual(
                    database.get_conversation(first["id"])["active_document_id"],
                    document_id,
                )

    def test_connections_use_wal_foreign_keys_and_busy_timeout(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "sqlite-settings.db"
            with patch.dict(os.environ, {"DATABASE_PATH": str(database_path)}):
                database.initialize_database()
                with closing(database.get_connection()) as connection:
                    journal_mode = connection.execute(
                        "PRAGMA journal_mode"
                    ).fetchone()[0]
                    busy_timeout = connection.execute(
                        "PRAGMA busy_timeout"
                    ).fetchone()[0]
                    foreign_keys = connection.execute(
                        "PRAGMA foreign_keys"
                    ).fetchone()[0]

            self.assertEqual(journal_mode.casefold(), "wal")
            self.assertEqual(busy_timeout, 30000)
            self.assertEqual(foreign_keys, 1)

    def test_concurrent_get_or_create_returns_one_conversation(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "concurrent.db"
            with patch.dict(os.environ, {"DATABASE_PATH": str(database_path)}):
                database.initialize_database()

                def create_conversation(_index):
                    return database.get_or_create_conversation(
                        "telegram",
                        "same-chat",
                        "same-user",
                    )["id"]

                with ThreadPoolExecutor(max_workers=8) as executor:
                    conversation_ids = list(executor.map(create_conversation, range(24)))

                with closing(database.get_connection()) as connection:
                    count = connection.execute(
                        "SELECT COUNT(*) FROM conversations"
                    ).fetchone()[0]

            self.assertEqual(len(set(conversation_ids)), 1)
            self.assertEqual(count, 1)

    def test_duplicate_conversation_migration_preserves_all_messages(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "duplicates.db"
            with patch.dict(os.environ, {"DATABASE_PATH": str(database_path)}):
                database.initialize_database()
                with closing(database.get_connection()) as connection, connection:
                    connection.execute(
                        "DROP INDEX uq_conversations_external_identity"
                    )
                    connection.executemany(
                        """
                        INSERT INTO conversations (
                            id, channel, external_chat_id, external_user_id,
                            active_document_id, created_at, updated_at
                        ) VALUES (?, 'telegram', 'same-chat', NULL, NULL, ?, ?)
                        """,
                        [
                            ("oldest", "2026-01-01", "2026-01-01"),
                            ("duplicate", "2026-01-02", "2026-01-02"),
                        ],
                    )
                    connection.executemany(
                        """
                        INSERT INTO messages (
                            conversation_id, role, content, created_at
                        ) VALUES (?, 'user', ?, '2026-01-03')
                        """,
                        [
                            ("oldest", "first message"),
                            ("duplicate", "second message"),
                        ],
                    )

                database.initialize_database()

                with closing(database.get_connection()) as connection:
                    conversations = connection.execute(
                        "SELECT id FROM conversations"
                    ).fetchall()
                    messages = connection.execute(
                        """
                        SELECT conversation_id, content
                        FROM messages ORDER BY id
                        """
                    ).fetchall()

            self.assertEqual([row["id"] for row in conversations], ["oldest"])
            self.assertEqual(
                [row["content"] for row in messages],
                ["first message", "second message"],
            )
            self.assertTrue(
                all(row["conversation_id"] == "oldest" for row in messages)
            )

    @staticmethod
    def create_old_database(database_path: Path) -> None:
        with closing(sqlite3.connect(database_path)) as connection, connection:
            connection.execute("""
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    document_type TEXT,
                    uploaded_at TEXT
                )
            """)
            connection.execute("""
                CREATE TABLE chunks (
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
            connection.execute("""
                INSERT INTO documents (filename, document_type, uploaded_at)
                VALUES ('faq.txt', 'faq', '2026-01-01T00:00:00+00:00')
            """)
            connection.execute("""
                INSERT INTO chunks (
                    document_id, chunk_id, faq_id, question, answer,
                    text, text_for_retrieval, embedding, filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                1,
                32,
                32,
                "Какие дедлайны?",
                "Подача начинается в декабре.",
                "Подача начинается в декабре.",
                "Какие дедлайны? Подача начинается в декабре.",
                json.dumps([1.0, 0.0]),
                "faq.txt"
            ))


if __name__ == "__main__":
    unittest.main()
