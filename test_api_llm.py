import importlib

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


def load_test_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api_llm.db"))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-tests")
    monkeypatch.setenv("GEMINI_MODEL", "fake-gemini-model")
    import database
    import app

    importlib.reload(database)
    return importlib.reload(app)


def test_successful_provider_answer_has_status_and_sources(tmp_path, monkeypatch):
    app_module = load_test_app(tmp_path, monkeypatch)
    app_module.DOCUMENT_CHUNKS[:] = [SYNTHETIC_CHUNK]
    monkeypatch.setattr(app_module, "get_embedding_model", lambda: FakeModel())
    monkeypatch.setattr("embedding_retriever.get_embedding_model", lambda: FakeModel())
    monkeypatch.setattr(
        "llm_answer_generator.generate_gemini_answer",
        lambda _question, _context: "The deadline is 30 April.",
    )

    with TestClient(app_module.app) as client:
        response = client.post("/ask-llm", json={"question": "What is the deadline?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["answer"] == "The deadline is 30 April."
    assert body["sources"][0]["faq_id"] == 11


def test_provider_429_returns_controlled_503_without_fallback(tmp_path, monkeypatch):
    app_module = load_test_app(tmp_path, monkeypatch)
    app_module.DOCUMENT_CHUNKS[:] = [SYNTHETIC_CHUNK]
    monkeypatch.setattr("embedding_retriever.get_embedding_model", lambda: FakeModel())
    error = type("Provider429", (Exception,), {"status_code": 429})()

    def fail_provider(_question, _context):
        raise error

    monkeypatch.setattr("llm_answer_generator.generate_gemini_answer", fail_provider)

    with TestClient(app_module.app) as client:
        response = client.post("/ask-llm", json={"question": "Нужна ли виза?"})

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "provider_unavailable"
    assert "without it" not in body["answer"].casefold()


def test_empty_retrieval_is_insufficient_not_provider_error(tmp_path, monkeypatch):
    app_module = load_test_app(tmp_path, monkeypatch)
    app_module.DOCUMENT_CHUNKS.clear()

    with TestClient(app_module.app) as client:
        response = client.post("/ask-llm", json={"question": "Unknown policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_document_information"
    assert "not enough information" in body["answer"].casefold()
    assert body["sources"] == []
