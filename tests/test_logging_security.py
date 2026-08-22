import logging

from admissions_rag_assistant.logging_config import configure_logging
from admissions_rag_assistant.logging_config import safe_log_text


def test_third_party_http_logs_cannot_emit_secret_bearing_urls(caplog):
    fake_token = "123456:FAKE_SENTINEL_BOT_TOKEN"
    fake_gemini_key = "FAKE_SENTINEL_GEMINI_KEY"
    fake_authorization = "Bearer FAKE_SENTINEL_AUTHORIZATION"
    fake_url = (
        f"https://api.telegram.invalid/bot{fake_token}/sendMessage"
        f"?key={fake_gemini_key}&authorization={fake_authorization}"
    )

    configure_logging()
    with caplog.at_level(logging.DEBUG):
        for name in ("httpx", "httpcore", "urllib3", "telegram", "google", "openai"):
            logging.getLogger(name).error("request %s", fake_url)
        logging.getLogger("application_regression_test").warning("safe application warning")

    output = caplog.text
    assert "safe application warning" in output
    assert fake_token not in output
    assert fake_gemini_key not in output
    assert fake_authorization not in output
    assert fake_url not in output


def test_application_diagnostics_redact_urls_and_secret_shapes(caplog):
    fake_token = "123456789:FAKE_SENTINEL_BOT_TOKEN_123456789"
    fake_key = "FAKE_SENTINEL_GEMINI_KEY"
    fake_url = f"https://api.telegram.invalid/bot{fake_token}/send?key={fake_key}"
    unsafe_query = f"check {fake_url} Authorization: Bearer_FAKE_SENTINEL"

    with caplog.at_level(logging.INFO):
        logging.getLogger("application_regression_test").info(
            "query=%r",
            safe_log_text(unsafe_query),
        )

    output = caplog.text
    assert "[url]" in output
    assert fake_token not in output
    assert fake_key not in output
    assert fake_url not in output
    assert "Bearer_FAKE_SENTINEL" not in output
