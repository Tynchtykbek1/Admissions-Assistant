from types import SimpleNamespace
import json
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


def provider_answer(status: str, answer: str) -> str:
    return json.dumps({"status": status, "answer": answer}, ensure_ascii=False)


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
        return_value=provider_answer("success", "The deadline is 30 April."),
    ):
        result = llm.generate_llm_answer("deadline?", CHUNKS)
    assert result.status == llm.SUCCESS
    assert result.answer == "The deadline is 30 April."


def test_status_is_not_inferred_from_answer_wording(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    answer = (
        "Недостаточно информации для полного списка, но упоминаются "
        "пренролмент и документы о доходах."
    )
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=provider_answer("partial_information", answer),
    ):
        result = llm.generate_llm_answer("visa?", CHUNKS)
    assert result.status == llm.PARTIAL_INFORMATION
    assert result.answer == answer


@pytest.mark.parametrize("provider", ["", "unsupported"])
def test_invalid_provider_configuration_is_controlled(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    result = llm.generate_llm_answer("question", CHUNKS)
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "invalid_configuration"


@pytest.mark.parametrize(
    "malformed_answer",
    [
        None,
        "",
        "   ",
        {"text": "answer"},
        "plain text is not a structured result",
        '{"status":"unknown","answer":"text"}',
    ],
)
def test_malformed_provider_response_is_controlled(monkeypatch, malformed_answer):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=malformed_answer,
    ):
        result = llm.generate_llm_answer("question", CHUNKS)
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "malformed_response"


def test_visa_partial_facts_preserve_structured_partial_status(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    answer = (
        "В документе нет полного перечня документов для визы, но упоминаются:\n"
        "• пренролмент\n• гарантийное письмо\n• документы о доходах."
    )
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=provider_answer("partial_information", answer),
    ):
        result = llm.generate_llm_answer("Какой перечень документов для визы?", CHUNKS)

    assert result.status == llm.PARTIAL_INFORMATION
    assert result.answer == answer
    assert "пренролмент" in result.answer


@pytest.mark.parametrize(
    ("question", "answer", "required_text"),
    [
        (
            "Мне нужна информация по поступлению",
            "Из документа:\n• подача — с декабря по май\n• нужны переводы и апостиль.",
            "апостиль",
        ),
        (
            "На какие университеты могу податься?",
            "Загруженный документ не содержит списка университетов или критериев выбора.",
            "не содержит",
        ),
        (
            "Какие дедлайны есть?",
            "Подача проходит с середины декабря до середины мая.",
            "середины мая",
        ),
    ],
)
def test_document_based_plain_text_behaviours(
    monkeypatch, question, answer, required_text
):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=provider_answer("success", answer),
    ):
        result = llm.generate_llm_answer(question, CHUNKS)
    assert result.status == llm.SUCCESS
    assert result.answer == answer
    assert required_text in result.answer


def test_university_absence_answer_has_no_invented_or_adjacent_facts(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    answer = "Загруженный документ не содержит списка университетов или критериев выбора."
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=provider_answer("insufficient_document_information", answer),
    ):
        result = llm.generate_llm_answer(
            "На какие университеты могу податься?", CHUNKS
        )
    assert result.status == llm.INSUFFICIENT_DOCUMENT_INFORMATION
    assert "дедлайн" not in result.answer.casefold()
    assert "виз" not in result.answer.casefold()
    assert not any(
        name in result.answer.casefold()
        for name in ("болон", "сапиенц", "паду", "милан")
    )


def test_markdown_fenced_json_is_accepted(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    response = "```json\n" + provider_answer("success", "Supported answer.") + "\n```"
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=response,
    ):
        result = llm.generate_llm_answer("question", CHUNKS)
    assert result.status == llm.SUCCESS
    assert result.answer == "Supported answer."
