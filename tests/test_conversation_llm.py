from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import errors as gemini_errors
from google.genai import types

from admissions_rag_assistant import llm_answer_generator as llm


@pytest.fixture(autouse=True)
def fake_provider_configuration(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-tests")
    monkeypatch.setenv("GEMINI_MODEL", "fake-gemini-model")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-for-tests")
    monkeypatch.setenv("OPENAI_MODEL", "fake-openai-model")


def _native_tool_response():
    part = types.Part(
        function_call=types.FunctionCall(
            name="search_knowledge", args={"query": "verified query"}
        ),
        thought_signature=b"signed-by-provider",
    )
    content = types.Content(role="model", parts=[part])
    return SimpleNamespace(
        function_calls=[part.function_call],
        candidates=[SimpleNamespace(content=content)],
    ), content


def test_native_tool_round_trip_preserves_content_and_diagnostics():
    first, original_content = _native_tool_response()
    with patch("admissions_rag_assistant.llm_answer_generator.genai.Client") as client_factory:
        client = client_factory.return_value
        client.models.generate_content.side_effect = [
            first,
            SimpleNamespace(text="Grounded answer."),
        ]
        tool_output = {"VERIFIED_CONTEXT": [{"text": "Verified fact."}]}
        result = llm._generate_gemini_conversation(
            "factual question", [], lambda _query: tool_output
        )

    second_contents = client.models.generate_content.call_args_list[1].kwargs["contents"]
    final_config = client.models.generate_content.call_args_list[1].kwargs["config"]
    assert second_contents[-2] is original_content
    assert second_contents[-2].parts[0].thought_signature == b"signed-by-provider"
    assert final_config.tools
    assert final_config.tool_config.function_calling_config.mode == "NONE"
    assert result.answer == "Grounded answer."
    assert result.tool_called is True
    assert result.tool_name == "search_knowledge"
    assert result.tool_query == "verified query"
    assert result.tool_output == tool_output


def test_post_tool_provider_error_preserves_tool_diagnostics():
    first, _ = _native_tool_response()
    with patch("admissions_rag_assistant.llm_answer_generator.genai.Client") as client_factory:
        client = client_factory.return_value
        client.models.generate_content.side_effect = [first, TimeoutError()]
        tool_output = {"VERIFIED_CONTEXT": [{"id": "knowledge-1"}]}
        result = llm._generate_gemini_conversation(
            "factual question", [], lambda _query: tool_output
        )

    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "timeout"
    assert result.tool_called is True
    assert result.tool_query == "verified query"
    assert result.tool_output == tool_output


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (TimeoutError(), "timeout"),
        (ConnectionError(), "connection"),
        (type("AuthenticationFailure", (Exception,), {"status_code": 401})(), "authentication"),
        (type("ProviderFailure", (Exception,), {"status_code": 500})(), "provider_http_error"),
    ],
)
def test_active_provider_errors_are_classified(monkeypatch, error, expected_category):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "generate_provider_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    result = llm.generate_conversation_answer("question", [], lambda _query: {})
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.answer == llm.PROVIDER_UNAVAILABLE_ANSWER
    assert result.error_category == expected_category


def test_realistic_gemini_resource_exhausted_is_rate_limited(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    error = gemini_errors.ClientError(
        429,
        {"error": {"message": "synthetic overload", "status": "RESOURCE_EXHAUSTED"}},
    )
    with patch("admissions_rag_assistant.llm_answer_generator.genai.Client") as client_factory:
        client_factory.return_value.models.generate_content.side_effect = error
        result = llm.generate_conversation_answer("question", [], lambda _query: {})
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "rate_limited"


@pytest.mark.parametrize("provider", ["", "unsupported"])
def test_invalid_provider_configuration_is_controlled(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    result = llm.generate_conversation_answer("question", [], lambda _query: {})
    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "invalid_configuration"


def test_gemini_direct_answer_is_returned_unchanged(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    expected = "Provider answer with DSU 2026 unchanged."
    with patch("admissions_rag_assistant.llm_answer_generator.genai.Client") as client_factory:
        client_factory.return_value.models.generate_content.return_value = SimpleNamespace(
            function_calls=[], text=expected
        )
        result = llm.generate_conversation_answer("question", [], lambda _query: {})
    assert result.status == llm.SUCCESS
    assert result.answer == expected
