import os

from dotenv import load_dotenv


load_dotenv()


def read_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def read_score_threshold(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if 0.0 <= value <= 1.0 else default
    except ValueError:
        return default


SEMANTIC_TOP_K = read_positive_int("SEMANTIC_TOP_K", 5)
SEMANTIC_SCORE_THRESHOLD = read_score_threshold("SEMANTIC_SCORE_THRESHOLD", 0.20)
SEMANTIC_FALLBACK_SAFE_MINIMUM = read_score_threshold(
    "SEMANTIC_FALLBACK_SAFE_MINIMUM", 0.15
)
SEMANTIC_FALLBACK_SCORE_THRESHOLD = max(
    read_score_threshold("SEMANTIC_FALLBACK_SCORE_THRESHOLD", 0.18),
    SEMANTIC_FALLBACK_SAFE_MINIMUM,
)
