import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient


FAQ_DOCUMENT = """
1. Is admission guaranteed?
Admission is never guaranteed. The university makes the final decision after reviewing the application.

2. Which documents are required for the visa?
Visa applicants need a passport, admission letter, and proof of financial means.
"""


def upload_text(client: TestClient, filename: str, text: str) -> dict:
    response = client.post(
        "/upload",
        files={"file": (filename, text.encode("utf-8"), "text/plain")}
    )
    response.raise_for_status()
    result = response.json()
    from admissions_rag_assistant import app_settings
    from admissions_rag_assistant import conversation_service

    app_settings.SYSTEM_DOCUMENT_ID = result["document_id"]
    app_settings.SYSTEM_DOCUMENT_ID_INVALID = False
    conversation_service.synchronize_system_document_conversations()
    return result


def run_health_test(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health endpoint test: PASS")


def run_faq_upload_test(client: TestClient) -> None:
    upload_result = upload_text(
        client,
        "admissions_faq_document.txt",
        FAQ_DOCUMENT.strip()
    )
    assert upload_result["document_type"] == "faq"
    assert upload_result["entries_count"] == upload_result["chunks_count"]
    print("FAQ-style document upload test: PASS")


def run_database_restore_test(app_module, database_module, temporary_db: Path) -> None:
    restored_chunks = database_module.load_latest_document()

    assert temporary_db.exists()
    assert temporary_db.resolve() != Path("admissions.db").resolve()
    assert restored_chunks
    assert restored_chunks[0]["filename"] == "admissions_faq_document.txt"
    assert len(restored_chunks[0]["embedding"]) > 0

    print("Temporary SQLite restore test: PASS")


def main() -> None:
    original_database_path = os.environ.get("DATABASE_PATH")

    with TemporaryDirectory() as temporary_directory:
        temporary_db = Path(temporary_directory) / "test_admissions.db"
        os.environ["DATABASE_PATH"] = str(temporary_db)

        from admissions_rag_assistant import database
        from admissions_rag_assistant import app

        database = importlib.reload(database)
        app = importlib.reload(app)

        with TestClient(app.app) as client:
            run_health_test(client)
            run_faq_upload_test(client)
            run_database_restore_test(app, database, temporary_db)

    if original_database_path is None:
        os.environ.pop("DATABASE_PATH", None)
    else:
        os.environ["DATABASE_PATH"] = original_database_path

    assert not temporary_db.exists()
    print("Temporary database cleanup: PASS")


if __name__ == "__main__":
    main()
