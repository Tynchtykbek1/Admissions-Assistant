import os
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

DEFAULT_EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def get_embedding_model_name() -> str:
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "").strip()
    return model_name or DEFAULT_EMBEDDING_MODEL_NAME


@lru_cache(maxsize=None)
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)


def get_embedding_model():
    return load_embedding_model(get_embedding_model_name())
