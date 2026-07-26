import os

from dotenv import load_dotenv


load_dotenv()


def read_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


CHAT_HISTORY_LIMIT = read_positive_int("CHAT_HISTORY_LIMIT", 8)
QUESTION_REWRITE_ENABLED = read_bool("QUESTION_REWRITE_ENABLED", True)
MAX_UPLOAD_SIZE_MB = read_positive_int("MAX_UPLOAD_SIZE_MB", 15)
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
DOCUMENT_CACHE_SIZE = read_positive_int("DOCUMENT_CACHE_SIZE", 8)
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024
