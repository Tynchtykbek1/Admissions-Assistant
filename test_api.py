import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient


SAMPLE_DOCUMENT = """
University Admissions Sample Document

University: University of Milan
Country: Italy
Program: Bachelor in International Relations
Language of instruction: English

Admission Requirements:
Applicants must have completed secondary school education or an equivalent qualification.
Applicants must provide a valid passport, high school diploma, transcript of records, motivation letter, and proof of English language proficiency.
For English-taught bachelor programs, the minimum English requirement is IELTS 6.0 or an equivalent certificate such as TOEFL or Cambridge English.

Application Deadline:
The standard application deadline for non-EU students is 30 April.
Late applications may not be accepted unless the university officially extends the deadline.
Students should always check the official university website before applying.

Tuition Fees:
The estimated tuition fee is 3000 EUR per year.

Visa Documents:
Non-EU students usually need a passport, admission letter, proof of financial means, accommodation proof, health insurance, and visa application form.
"""


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
    import app_settings
    import conversation_service

    app_settings.SYSTEM_DOCUMENT_ID = result["document_id"]
    app_settings.SYSTEM_DOCUMENT_ID_INVALID = False
    conversation_service.synchronize_system_document_conversations()
    return result


def ask_semantic(
    client: TestClient,
    question: str,
    document_id: int,
) -> dict:
    response = client.post(
        "/ask-semantic",
        json={
            "question": question,
            "external_chat_id": "manual-test-chat",
            "external_user_id": "manual-test-user",
            "document_id": document_id,
        },
    )
    response.raise_for_status()
    return response.json()


def run_standard_document_tests(client: TestClient) -> None:
    upload_result = upload_text(
        client,
        "admissions_sample_document.txt",
        SAMPLE_DOCUMENT.strip()
    )
    assert upload_result["document_type"] == "standard"

    test_cases = [
        ("What English level is required?", "IELTS 6.0"),
        ("What is the application deadline?", "30 April")
    ]

    for question, expected in test_cases:
        result = ask_semantic(client, question, upload_result["document_id"])
        assert expected.casefold() in result["answer"].casefold()

    print("Standard document API tests: PASS")


def run_health_test(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health endpoint test: PASS")


def run_faq_test(client: TestClient) -> None:
    upload_result = upload_text(
        client,
        "admissions_faq_document.txt",
        FAQ_DOCUMENT.strip()
    )
    assert upload_result["document_type"] == "faq"
    assert upload_result["entries_count"] == upload_result["chunks_count"]

    result = ask_semantic(
        client,
        "Do I have guaranteed admission?",
        upload_result["document_id"],
    )
    answer = result["answer"].casefold()
    assert "admission is never guaranteed" in answer
    assert "is admission guaranteed?" not in answer

    print("FAQ-style document API test: PASS")


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

        import database
        import app

        database = importlib.reload(database)
        app = importlib.reload(app)

        with TestClient(app.app) as client:
            run_health_test(client)
            run_standard_document_tests(client)
            run_faq_test(client)
            run_database_restore_test(app, database, temporary_db)

    if original_database_path is None:
        os.environ.pop("DATABASE_PATH", None)
    else:
        os.environ["DATABASE_PATH"] = original_database_path

    assert not temporary_db.exists()
    print("Temporary database cleanup: PASS")


if __name__ == "__main__":
    main()
