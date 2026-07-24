import logging
import re


_SENSITIVE_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "telegram",
    "openai",
    "google",
)

_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)\bauthorization\b\s*[:=]\s*\S+(?:\s+\S+)?"
    r"|\b(api[_-]?key|token)\b\s*[:=]\s*\S+"
)
_TELEGRAM_TOKEN_PATTERN = re.compile(r"\b\d{5,}:[A-Za-z0-9_-]{20,}\b")


def safe_log_text(value: str, max_length: int = 160) -> str:
    """Return a short diagnostic representation that cannot include full URLs or tokens."""
    flattened = " ".join(value.split())
    flattened = _URL_PATTERN.sub("[url]", flattened)
    flattened = _AUTHORIZATION_PATTERN.sub("[credential-redacted]", flattened)
    flattened = _TELEGRAM_TOKEN_PATTERN.sub("[telegram-token]", flattened)
    return flattened[:max_length]


def configure_logging() -> None:
    """Keep application diagnostics while silencing credential-bearing HTTP logs."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    for logger_name in _SENSITIVE_THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.handlers.clear()
        third_party_logger.addHandler(logging.NullHandler())
        third_party_logger.propagate = False
        third_party_logger.setLevel(logging.CRITICAL + 1)
