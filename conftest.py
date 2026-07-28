import numpy as np
import pytest


class _OfflineEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([1.0, 0.0])
        return np.array([[1.0, 0.0] for _ in texts])


@pytest.fixture(autouse=True)
def _prevent_model_downloads(monkeypatch):
    import embedding_model

    embedding_model.load_embedding_model.cache_clear()
    monkeypatch.setattr(
        embedding_model,
        "load_embedding_model",
        lambda _model_name: _OfflineEmbeddingModel(),
    )
