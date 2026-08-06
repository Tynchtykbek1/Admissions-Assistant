import os

from dotenv import load_dotenv


load_dotenv()


def read_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def read_optional_positive_int(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None


def read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


CHAT_HISTORY_LIMIT = read_positive_int("CHAT_HISTORY_LIMIT", 8)
CHAT_HISTORY_CHARACTER_LIMIT = read_positive_int("CHAT_HISTORY_CHARACTER_LIMIT", 4000)
REWRITE_HISTORY_MESSAGE_LIMIT = read_positive_int("REWRITE_HISTORY_MESSAGE_LIMIT", 10)
REWRITE_HISTORY_CHARACTER_LIMIT = read_positive_int("REWRITE_HISTORY_CHARACTER_LIMIT", 2000)
QUESTION_REWRITE_ENABLED = read_bool("QUESTION_REWRITE_ENABLED", True)
MAX_UPLOAD_SIZE_MB = read_positive_int("MAX_UPLOAD_SIZE_MB", 15)
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
DOCUMENT_CACHE_SIZE = read_positive_int("DOCUMENT_CACHE_SIZE", 8)
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024
SYSTEM_DOCUMENT_ID: int | None = read_optional_positive_int("SYSTEM_DOCUMENT_ID")
SYSTEM_DOCUMENT_ID_INVALID = bool(
    os.getenv("SYSTEM_DOCUMENT_ID", "").strip()
) and SYSTEM_DOCUMENT_ID is None
