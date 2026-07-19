import json
import os
import sqlite3
import unittest
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
            self.assertEqual(stored_model, "new-multilingual-model")
            self.assertEqual(stored_embedding, [0.25, 0.75])

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
