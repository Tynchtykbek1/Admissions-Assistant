import importlib
import json

import numpy as np
from fastapi.testclient import TestClient


def _chunk(filename: str, text: str) -> dict:
    return {
        "chunk_id": 1,
        "faq_id": 1,
        "question": "What is required?",
        "answer": text,
        "filename": filename,
        "text": text,
        "text_for_retrieval": text,
        "embedding": np.array([1.0, 0.0]),
    }


def _load_modules(tmp_path, monkeypatch, system_document_id: str = "1"):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "system-document.db"))
    monkeypatch.setenv("SYSTEM_DOCUMENT_ID", system_document_id)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")

    import app_settings
    import database
    import conversation_service
    import rag_service
    import app

    app_settings = importlib.reload(app_settings)
    database = importlib.reload(database)
    conversation_service = importlib.reload(conversation_service)
    rag_service = importlib.reload(rag_service)
    app = importlib.reload(app)
    return app_settings, database, conversation_service, rag_service, app


def _add_document(database, filename: str, *, with_chunks: bool = True) -> int:
    chunks = [_chunk(filename, f"Supported fact from {filename}.")] if with_chunks else []
    return database.insert_document_with_chunks(
        filename,
        f"stored-{filename}",
        "faq",
        database.get_embedding_model_name(),
        chunks,
    )


def test_new_telegram_and_web_conversations_receive_system_document(
    tmp_path, monkeypatch
):
    _, database, service, _, _ = _load_modules(tmp_path, monkeypatch)
    system_id = _add_document(database, "system.txt")

    telegram = service.resolve_conversation(
        conversation_id=None,
        external_chat_id="telegram-chat",
        external_user_id="telegram-user",
        allow_latest_document_default=False,
    )
    web = service.resolve_conversation(
        conversation_id=None,
        external_chat_id=None,
        external_user_id=None,
        allow_latest_document_default=True,
    )

    assert telegram["active_document_id"] == system_id
    assert web["active_document_id"] == system_id


def test_existing_null_and_old_document_conversations_are_repaired(
    tmp_path, monkeypatch
):
    _, database, service, _, _ = _load_modules(tmp_path, monkeypatch, "2")
    old_id = _add_document(database, "old.txt")
    system_id = _add_document(database, "system.txt")
    without_document = database.get_or_create_conversation("web", "null-document")
    old_document = database.get_or_create_conversation(
        "telegram",
        "old-document",
        default_document_id=old_id,
    )
    database.add_message(old_document["id"], "user", "Keep this message")

    service.resolve_conversation(
        conversation_id=without_document["id"],
        external_chat_id=None,
        external_user_id=None,
        allow_latest_document_default=True,
    )

    assert system_id == 2
    assert (
        database.get_conversation(without_document["id"])["active_document_id"]
        == system_id
    )
    assert (
        database.get_conversation(old_document["id"])["active_document_id"]
        == system_id
    )
    assert database.get_recent_messages(old_document["id"], 10)[0]["content"] == (
        "Keep this message"
    )


def test_changing_system_document_switches_all_conversations(
    tmp_path, monkeypatch
):
    settings, database, service, _, _ = _load_modules(tmp_path, monkeypatch)
    first_id = _add_document(database, "first.txt")
    second_id = _add_document(database, "second.txt")
    conversation = service.resolve_conversation(
        conversation_id=None,
        external_chat_id="chat",
        external_user_id="user",
        allow_latest_document_default=False,
    )
    assert conversation["active_document_id"] == first_id

    monkeypatch.setattr(settings, "SYSTEM_DOCUMENT_ID", second_id)
    service.resolve_conversation(
        conversation_id=conversation["id"],
        external_chat_id="chat",
        external_user_id="user",
        allow_latest_document_default=False,
    )

    assert (
        database.get_conversation(conversation["id"])["active_document_id"]
        == second_id
    )


def test_status_and_reset_keep_system_document_and_isolate_histories(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch)
    system_id = _add_document(database, "system.txt")

    with TestClient(app_module.app) as client:
        first_status = client.get(
            "/conversation/status",
            params={"external_chat_id": "chat-1", "external_user_id": "user-1"},
        )
        second_status = client.get(
            "/conversation/status",
            params={"external_chat_id": "chat-2", "external_user_id": "user-2"},
        )
        first_id = first_status.json()["conversation_id"]
        second_id = second_status.json()["conversation_id"]
        database.add_message(first_id, "user", "clear me")
        database.add_message(second_id, "user", "keep me")
        reset = client.post(
            "/conversation/reset",
            json={"external_chat_id": "chat-1", "external_user_id": "user-1"},
        )

    assert first_status.status_code == 200
    assert first_status.json()["active_document_id"] == system_id
    assert first_status.json()["active_document_filename"] == "system.txt"
    assert second_status.json()["active_document_id"] == system_id
    assert reset.json()["active_document_id"] == system_id
    assert database.get_recent_messages(first_id, 10) == []
    assert database.get_recent_messages(second_id, 10)[0]["content"] == "keep me"


def test_client_cannot_override_system_document_on_any_question_endpoint(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch)
    _add_document(database, "system.txt")
    other_id = _add_document(database, "other.txt")

    with TestClient(app_module.app) as client:
        for endpoint in ("/chat", "/ask-llm", "/ask", "/ask-semantic"):
            response = client.post(
                endpoint,
                json={"question": "What is required?", "document_id": other_id},
            )
            assert response.status_code == 409, endpoint


def test_missing_or_empty_system_document_is_controlled(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch, "2")
    _add_document(database, "empty.txt", with_chunks=False)

    with TestClient(app_module.app) as client:
        missing = client.post("/chat", json={"question": "What is required?"})
    assert missing.status_code == 503
    assert missing.json()["status"] == "system_document_unavailable"
    assert missing.json()["sources"] == []
    assert "2" not in missing.json()["answer"]

    monkeypatch.setenv("SYSTEM_DOCUMENT_ID", "1")
    import app_settings

    importlib.reload(app_settings)
    with TestClient(app_module.app) as client:
        empty = client.post("/chat", json={"question": "What is required?"})
    assert empty.status_code == 503
    assert empty.json()["status"] == "system_document_unavailable"


def test_ready_requires_available_system_document_with_chunks(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch)

    with TestClient(app_module.app) as client:
        unavailable = client.get("/ready")
    assert unavailable.status_code == 503
    assert unavailable.json()["system_document_configured"] is True
    assert unavailable.json()["system_document_available"] is False

    _add_document(database, "system.txt")
    with TestClient(app_module.app) as client:
        ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "database": "ok",
        "provider_configured": True,
        "system_document_configured": True,
        "system_document_available": True,
    }


def test_chat_uses_system_document_and_accepts_matching_id(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch)
    system_id = _add_document(database, "system.txt")

    class FakeModel:
        def encode(self, _text, normalize_embeddings=True):
            return np.array([1.0, 0.0])

    monkeypatch.setattr(
        "embedding_retriever.get_embedding_model",
        lambda: FakeModel(),
    )
    monkeypatch.setattr(
        "llm_answer_generator.generate_gemini_answer",
        lambda _question, _context, **_kwargs: json.dumps(
            {"status": "success", "answer": "Supported system answer."}
        ),
    )
    with TestClient(app_module.app) as client:
        response = client.post(
            "/chat",
            json={"question": "What is required?", "document_id": system_id},
        )

    assert response.status_code == 200
    assert response.json()["document_id"] == system_id
    assert response.json()["sources"][0]["filename"] == "system.txt"


def test_invalid_system_document_setting_never_falls_back_to_latest(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch, "-4")
    latest_id = _add_document(database, "latest.txt")

    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"question": "What is required?"})

    assert latest_id == 1
    assert response.status_code == 503
    assert response.json()["status"] == "system_document_unavailable"
