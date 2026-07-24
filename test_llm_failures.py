from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import errors as gemini_errors

import llm_answer_generator as llm


CHUNKS = [
    {
        "chunk_id": 7,
        "faq_id": 7,
        "filename": "synthetic.txt",
        "text": "DDV is required. Yes, without it the process cannot continue.",
        "score": 0.8,
    }
]


@pytest.fixture(autouse=True)
def fake_provider_configuration(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-tests")
    monkeypatch.setenv("GEMINI_MODEL", "fake-gemini-model")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-for-tests")
    monkeypatch.setenv("OPENAI_MODEL", "fake-openai-model")


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (SimpleNamespace(), "unexpected_provider_error"),
        (TimeoutError(), "timeout"),
        (ConnectionError(), "connection"),
    ],
)
def test_provider_errors_never_use_basic_fallback(monkeypatch, error, expected_category):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    if isinstance(error, SimpleNamespace):
        error = type("Provider429", (Exception,), {"status_code": 429})()

    with (
        patch("llm_answer_generator.generate_gemini_answer", side_effect=error),
        patch("answer_generator.generate_basic_answer") as basic,
    ):
        result = llm.generate_llm_answer("Нужна ли виза?", CHUNKS)

    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == (
        "rate_limited" if getattr(error, "status_code", None) == 429 else expected_category
    )
    assert result.answer == llm.PROVIDER_UNAVAILABLE_ANSWER
    basic.assert_not_called()


def test_authentication_error_is_safe(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    error = type("AuthenticationFailure", (Exception,), {"status_code": 401})()
    with patch("llm_answer_generator.generate_openai_answer", side_effect=error):
        result = llm.generate_llm_answer("question", CHUNKS)
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "authentication"


def test_realistic_gemini_resource_exhausted_is_rate_limited(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    error = gemini_errors.ClientError(
        429,
        {"error": {"message": "synthetic overload", "status": "RESOURCE_EXHAUSTED"}},
    )
    with patch("llm_answer_generator.generate_gemini_answer", side_effect=error):
        result = llm.generate_llm_answer("question", CHUNKS)
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "rate_limited"
    assert result.provider_status == 429


def test_insufficient_information_is_distinct_and_skips_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with patch("llm_answer_generator.generate_gemini_answer") as provider:
        result = llm.generate_llm_answer("unknown", [])
    assert result.status == llm.INSUFFICIENT_DOCUMENT_INFORMATION
    assert result.answer == llm.INSUFFICIENT_INFORMATION_ANSWER
    provider.assert_not_called()


def test_successful_provider_answer_is_preserved(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value="The deadline is 30 April.",
    ):
        result = llm.generate_llm_answer("deadline?", CHUNKS)
    assert result.status == llm.SUCCESS
    assert result.answer == "The deadline is 30 April."


def test_provider_insufficient_answer_has_distinct_status(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value="There is not enough information in the uploaded document.",
    ):
        result = llm.generate_llm_answer("visa?", CHUNKS)
    assert result.status == llm.INSUFFICIENT_DOCUMENT_INFORMATION


@pytest.mark.parametrize("provider", ["", "unsupported"])
def test_invalid_provider_configuration_is_controlled(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    result = llm.generate_llm_answer("question", CHUNKS)
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "invalid_configuration"


@pytest.mark.parametrize("malformed_answer", [None, "", "   ", {"text": "answer"}])
def test_malformed_provider_response_is_controlled(monkeypatch, malformed_answer):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=malformed_answer,
    ):
        result = llm.generate_llm_answer("question", CHUNKS)
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "malformed_response"
