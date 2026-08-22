from unittest.mock import Mock

from fastapi.testclient import TestClient


class WarmupModel:
    def __init__(self):
        self.encode = Mock(return_value=[[1.0, 0.0]])


def _make_readiness_dependencies_available(monkeypatch, app_module):
    monkeypatch.setattr(app_module, "database_is_ready", lambda: True)
    monkeypatch.setattr(app_module, "_provider_configuration_ready", lambda: True)
    monkeypatch.setattr(app_module, "is_system_document_configured", lambda: True)
    monkeypatch.setattr(
        app_module,
        "get_system_document_state",
        lambda: type("State", (), {"document": {"id": 1}})(),
    )


def test_embedding_initializes_once_and_warms_up_during_startup(monkeypatch):
    from admissions_rag_assistant import app

    model = WarmupModel()
    loader = Mock(return_value=model)
    monkeypatch.setattr(app, "get_embedding_model", loader)
    app.app.state.embedding_ready = False
    with TestClient(app.app):
        assert app.app.state.embedding_ready is True
    with TestClient(app.app):
        assert app.app.state.embedding_ready is True

    loader.assert_called_once_with()
    model.encode.assert_called_once_with(
        ["embedding warmup"],
        normalize_embeddings=True,
    )


def test_ready_succeeds_only_after_successful_warmup(monkeypatch):
    from admissions_rag_assistant import app

    _make_readiness_dependencies_available(monkeypatch, app)
    monkeypatch.setattr(app, "get_embedding_model", lambda: WarmupModel())
    app.app.state.embedding_ready = False
    assert app.ready().status_code == 503

    with TestClient(app.app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["embedding"] == "ok"


def test_failed_warmup_keeps_ready_unavailable(monkeypatch):
    from admissions_rag_assistant import app

    _make_readiness_dependencies_available(monkeypatch, app)

    def fail_initialization():
        raise RuntimeError("synthetic local failure")

    monkeypatch.setattr(app, "get_embedding_model", fail_initialization)
    app.app.state.embedding_ready = False
    with TestClient(app.app) as client:
        health_response = client.get("/health")
        ready_response = client.get("/ready")

    assert health_response.status_code == 200
    assert ready_response.status_code == 503
    assert ready_response.json()["embedding"] == "unavailable"
