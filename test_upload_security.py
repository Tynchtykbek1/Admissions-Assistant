import importlib
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0] for _ in texts])


def load_upload_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "uploads.db"))
    import database
    import app

    database = importlib.reload(database)
    app = importlib.reload(app)
    upload_dir = tmp_path / "files"
    upload_dir.mkdir()
    monkeypatch.setattr(app, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app, "get_embedding_model", lambda: FakeModel())
    return app, database, upload_dir


def test_upload_rejects_unsupported_empty_malformed_and_oversized(
    tmp_path, monkeypatch
):
    app_module, _database, upload_dir = load_upload_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "MAX_UPLOAD_SIZE_BYTES", 8)
    with TestClient(app_module.app) as client:
        unsupported = client.post(
            "/upload", files={"file": ("bad.docx", b"text", "application/octet-stream")}
        )
        empty = client.post(
            "/upload", files={"file": ("empty.txt", b"", "text/plain")}
        )
        malformed = client.post(
            "/upload", files={"file": ("bad.pdf", b"not-pdf", "application/pdf")}
        )
        oversized = client.post(
            "/upload", files={"file": ("large.txt", b"0123456789", "text/plain")}
        )
    assert unsupported.status_code == 400
    assert empty.status_code == 400
    assert malformed.status_code == 400
    assert oversized.status_code == 413
    assert list(upload_dir.iterdir()) == []


def test_duplicate_original_names_use_unique_storage_and_no_path_leak(
    tmp_path, monkeypatch
):
    app_module, database, upload_dir = load_upload_app(tmp_path, monkeypatch)
    with TestClient(app_module.app) as client:
        first = client.post(
            "/upload", files={"file": ("guide.txt", b"First useful text.", "text/plain")}
        )
        second = client.post(
            "/upload", files={"file": ("guide.txt", b"Second useful text.", "text/plain")}
        )
    assert first.status_code == second.status_code == 200
    assert "saved_to" not in first.json()
    assert first.json()["document_id"] != second.json()["document_id"]
    stored_names = sorted(path.name for path in upload_dir.iterdir())
    assert len(stored_names) == 2
    assert stored_names[0] != stored_names[1]
    assert all(Path(name).suffix == ".txt" for name in stored_names)
    assert database.get_document(first.json()["document_id"])["filename"] == "guide.txt"


def test_persistence_failure_cleans_file_and_does_not_activate_document(
    tmp_path, monkeypatch
):
    app_module, database, upload_dir = load_upload_app(tmp_path, monkeypatch)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            app_module,
            "insert_document_with_chunks",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic")),
        )
        with TestClient(app_module.app) as client:
            response = client.post(
                "/upload",
                files={"file": ("guide.txt", b"Useful text.", "text/plain")},
            )
    assert response.status_code == 500
    assert list(upload_dir.iterdir()) == []
    conversation = database.get_or_create_conversation("web", "default-local")
    assert conversation["active_document_id"] is None
