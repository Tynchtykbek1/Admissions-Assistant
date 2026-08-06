import os
import math

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
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else default
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
CONTEXT_SCORE_MARGIN = read_score_threshold("CONTEXT_SCORE_MARGIN", 0.12)

# Retrieval v2 first builds a wider local candidate pool, then keeps only a
# small category-compatible context. These are deterministic CPU operations;
# they do not add provider or embedding calls.
HYBRID_SEMANTIC_CANDIDATE_LIMIT = read_positive_int(
    "HYBRID_SEMANTIC_CANDIDATE_LIMIT", 15
)
HYBRID_LEXICAL_CANDIDATE_LIMIT = read_positive_int(
    "HYBRID_LEXICAL_CANDIDATE_LIMIT", 15
)
HYBRID_MAX_CONTEXT_CHUNKS = read_positive_int("HYBRID_MAX_CONTEXT_CHUNKS", 5)

# Fusion weights. Semantic similarity remains the main relevance signal;
# lexical overlap and intent compatibility can correct close semantic matches.
HYBRID_SEMANTIC_WEIGHT = 0.55
HYBRID_LEXICAL_WEIGHT = 0.25
HYBRID_INTENT_MATCH_BONUS = 0.30
HYBRID_EXACT_QUESTION_BONUS = 0.35
HYBRID_UNKNOWN_CATEGORY_PENALTY = 0.08
HYBRID_CATEGORY_CONFLICT_PENALTY = 0.70
HYBRID_UNRELATED_AMOUNT_PENALTY = 0.90
HYBRID_UNRELATED_GUARANTEE_PENALTY = 0.90
HYBRID_MIN_FINAL_SCORE = 0.25
HYBRID_CONTEXT_SCORE_MARGIN = 0.18
