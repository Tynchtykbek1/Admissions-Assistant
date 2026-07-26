import importlib
import json

import numpy as np
from fastapi.testclient import TestClient


SYNTHETIC_CHUNK = {
    "chunk_id": 11,
    "faq_id": 11,
    "question": "What is the deadline?",
    "filename": "synthetic_faq.txt",
    "text": "The deadline is 30 April.",
    "text_for_retrieval": "What is the deadline?\nThe deadline is 30 April.",
    "embedding": np.array([1.0, 0.0]),
}


class FakeModel:
    def encode(self, _text, normalize_embeddings=True):
        return np.array([1.0, 0.0])


def load_test_app(tmp_path, monkeypatch, *, with_document=True):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api_llm.db"))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-tests")
    monkeypatch.setenv("GEMINI_MODEL", "fake-gemini-model")
    import database
    import conversation_service
    import rag_service
    import app

    database = importlib.reload(database)
    importlib.reload(conversation_service)
    rag_service = importlib.reload(rag_service)
    app = importlib.reload(app)
    if with_document:
        database.insert_document_with_chunks(
            "synthetic_faq.txt",
            "stored-synthetic.txt",
            "faq",
            database.get_embedding_model_name(),
            [SYNTHETIC_CHUNK],
        )
    rag_service.invalidate_document_cache()
    monkeypatch.setattr("embedding_retriever.get_embedding_model", lambda: FakeModel())
    return app


def test_chat_success_has_conversation_status_timings_and_sources(
    tmp_path, monkeypatch
):
    app_module = load_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "llm_answer_generator.generate_gemini_answer",
        lambda _question, _context, **_kwargs: json.dumps({
            "status": "success",
            "answer": "The deadline is 30 April.",
        }),
    )

    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"question": "What is the deadline?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["standalone_question"] == "What is the deadline?"
    assert body["conversation_id"]
    assert body["retrieval_duration_ms"] >= 0
    assert body["sources"][0]["faq_id"] == 11


def test_ask_llm_backward_compatibility(tmp_path, monkeypatch):
    app_module = load_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "llm_answer_generator.generate_gemini_answer",
        lambda _question, _context, **_kwargs: json.dumps({
            "status": "success",
            "answer": "The deadline is 30 April.",
        }),
    )
    with TestClient(app_module.app) as client:
        response = client.post(
            "/ask-llm", json={"question": "What is the deadline?"}
        )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_provider_429_returns_controlled_503(tmp_path, monkeypatch):
    app_module = load_test_app(tmp_path, monkeypatch)
    error = type("Provider429", (Exception,), {"status_code": 429})()

    def fail_provider(_question, _context, **_kwargs):
        raise error

    monkeypatch.setattr(
        "llm_answer_generator.generate_gemini_answer", fail_provider
    )
    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"question": "What is the deadline?"})
    assert response.status_code == 503
    assert response.json()["status"] == "provider_unavailable"
    assert response.json()["sources"] == []


def test_no_active_document_is_controlled_for_telegram(tmp_path, monkeypatch):
    app_module = load_test_app(tmp_path, monkeypatch, with_document=False)
    with TestClient(app_module.app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "Unknown policy?",
                "external_chat_id": "telegram-chat",
                "external_user_id": "telegram-user",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_document_information"
    assert body["sources"] == []
