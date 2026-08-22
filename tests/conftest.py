import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class _OfflineEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([1.0, 0.0])
        return np.array([[1.0, 0.0] for _ in texts])


@pytest.fixture(autouse=True)
def _prevent_model_downloads(monkeypatch):
    from admissions_rag_assistant import embedding_model

    embedding_model.load_embedding_model.cache_clear()
    monkeypatch.setattr(
        embedding_model,
        "load_embedding_model",
        lambda _model_name: _OfflineEmbeddingModel(),
    )
