import importlib

from fastapi.testclient import TestClient


def test_health_uses_temporary_database_without_loading_embedding_model(
    tmp_path, monkeypatch
):
    temporary_database = tmp_path / "health_test.db"
    monkeypatch.setenv("DATABASE_PATH", str(temporary_database))

    import database
    import app

    importlib.reload(database)
    app_module = importlib.reload(app)

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "demo_mode": False}
    assert temporary_database.exists()
    assert database.get_database_path() == temporary_database


def test_browser_ui_is_not_exposed(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "no-ui.db"))
    import database
    import app

    importlib.reload(database)
    app_module = importlib.reload(app)
    with TestClient(app_module.app) as client:
        response = client.get("/ui")

    assert response.status_code == 404


def test_ready_checks_database_and_configuration_without_provider_call(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ready.db"))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "model")
    import database
    import app

    importlib.reload(database)
    app_module = importlib.reload(app)
    with TestClient(app_module.app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["database"] == "ok"
    assert response.json()["provider_configured"] is False
    assert response.json()["demo_mode"] is False


def test_demo_mode_is_explicitly_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "demo-health.db"))
    monkeypatch.setenv("DEMO_MODE", "true")
    import database
    import app
    importlib.reload(database)
    app_module = importlib.reload(app)
    with TestClient(app_module.app) as client:
        assert client.get("/health").json()["demo_mode"] is True
        assert client.get("/ready").json()["demo_mode"] is True
