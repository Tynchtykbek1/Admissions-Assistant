import importlib
import json

import numpy as np
import pytest
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


def test_new_telegram_conversation_receives_system_document(tmp_path, monkeypatch):
    _, database, service, _, _ = _load_modules(tmp_path, monkeypatch)
    system_id = _add_document(database, "system.txt")

    telegram = service.resolve_conversation(
        conversation_id=None,
        external_chat_id="telegram-chat",
        external_user_id="telegram-user",
    )

    assert telegram["active_document_id"] == system_id


def test_startup_sync_repairs_all_conversations_and_preserves_messages(
    tmp_path, monkeypatch
):
    _, database, service, _, _ = _load_modules(tmp_path, monkeypatch, "2")
    old_id = _add_document(database, "old.txt")
    system_id = _add_document(database, "system.txt")
    without_document = database.get_or_create_conversation(
        "telegram", "null-document", "null-user"
    )
    old_document = database.get_or_create_conversation(
        "telegram",
        "old-document",
        default_document_id=old_id,
    )
    database.add_message(old_document["id"], "user", "Keep this message")

    state = service.synchronize_system_document_conversations()

    assert system_id == 2
    assert state.document["id"] == system_id
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
    )
    assert conversation["active_document_id"] == first_id

    monkeypatch.setattr(settings, "SYSTEM_DOCUMENT_ID", second_id)
    service.synchronize_system_document_conversations()

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


def test_client_cannot_override_system_document_on_chat(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch)
    _add_document(database, "system.txt")
    other_id = _add_document(database, "other.txt")

    with TestClient(app_module.app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "What is required?",
                "document_id": other_id,
                "external_chat_id": "chat",
                "external_user_id": "user",
            },
        )
        assert response.status_code == 409


def test_missing_or_empty_system_document_is_controlled(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch, "2")
    _add_document(database, "empty.txt", with_chunks=False)
    monkeypatch.setattr(
        "rag_service.generate_conversation_answer",
        lambda *_args: pytest.fail("LLM/provider must not be called"),
    )
    monkeypatch.setattr(
        "rag_service.find_relevant_chunks_semantic",
        lambda **_kwargs: pytest.fail("retrieval must not be called"),
    )

    with TestClient(app_module.app) as client:
        missing = client.post(
            "/chat",
            json={
                "question": "What is required?",
                "external_chat_id": "chat",
                "external_user_id": "user",
            },
        )
    assert missing.status_code == 503
    assert missing.json()["status"] == "system_document_unavailable"
    assert missing.json()["sources"] == []
    assert "2" not in missing.json()["answer"]

    monkeypatch.setenv("SYSTEM_DOCUMENT_ID", "1")
    import app_settings

    importlib.reload(app_settings)
    with TestClient(app_module.app) as client:
        empty = client.post(
            "/chat",
            json={
                "question": "What is required?",
                "external_chat_id": "chat",
                "external_user_id": "user",
            },
        )
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
        "embedding": "ok",
        "provider_configured": True,
        "system_document_configured": True,
        "system_document_available": True,
        "demo_mode": False,
    }


def test_missing_telegram_identity_is_rejected_by_all_conversation_endpoints(
    tmp_path, monkeypatch
):
    _, database, _, _, app_module = _load_modules(tmp_path, monkeypatch)
    _add_document(database, "system.txt")

    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"question": "Question?"})
        assert response.status_code == 400
        reset = client.post("/conversation/reset", json={})
        status = client.get("/conversation/status")

    assert reset.status_code == 400
    assert status.status_code == 400


def test_foreign_conversation_id_cannot_be_reused(tmp_path, monkeypatch):
    _, database, service, _, app_module = _load_modules(tmp_path, monkeypatch)
    _add_document(database, "system.txt")
    owner = service.resolve_conversation(
        conversation_id=None,
        external_chat_id="owner-chat",
        external_user_id="owner-user",
    )

    with TestClient(app_module.app) as client:
        response = client.get(
            "/conversation/status",
            params={
                "conversation_id": owner["id"],
                "external_chat_id": "attacker-chat",
                "external_user_id": "attacker-user",
            },
        )

    assert response.status_code == 403


def test_normal_request_repairs_only_current_conversation(tmp_path, monkeypatch):
    _, database, service, _, _ = _load_modules(tmp_path, monkeypatch, "2")
    old_id = _add_document(database, "old.txt")
    system_id = _add_document(database, "system.txt")
    current = database.get_or_create_conversation(
        "telegram", "current-chat", "current-user", default_document_id=old_id
    )
    untouched = database.get_or_create_conversation(
        "telegram", "other-chat", "other-user", default_document_id=old_id
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            service,
            "synchronize_conversations_active_document",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("bulk synchronization ran during a request")
            ),
        )
        resolved = service.resolve_conversation(
            conversation_id=current["id"],
            external_chat_id="current-chat",
            external_user_id="current-user",
        )

    assert resolved["active_document_id"] == system_id
    assert database.get_conversation(untouched["id"])["active_document_id"] == old_id


def test_correct_conversation_does_not_update_document(tmp_path, monkeypatch):
    _, database, service, _, _ = _load_modules(tmp_path, monkeypatch)
    system_id = _add_document(database, "system.txt")
    conversation = database.get_or_create_conversation(
        "telegram",
        "correct-chat",
        "correct-user",
        default_document_id=system_id,
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            service,
            "update_active_document",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected document update")
            ),
        )
        resolved = service.resolve_conversation(
            conversation_id=conversation["id"],
            external_chat_id="correct-chat",
            external_user_id="correct-user",
        )

    assert resolved["active_document_id"] == system_id
