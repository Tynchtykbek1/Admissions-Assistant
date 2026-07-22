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
    assert response.json() == {"status": "ok"}
    assert temporary_database.exists()
    assert database.get_database_path() == temporary_database
