from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import types

import llm_answer_generator as llm


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


def test_native_tool_round_trip_preserves_original_content(monkeypatch):
    first, original_content = _native_tool_response()
    generate = patch("llm_answer_generator.genai.Client")
    with generate as client_factory:
        client = client_factory.return_value
        client.models.generate_content.side_effect = [
            first,
            SimpleNamespace(text="Grounded answer."),
        ]
        result = llm._generate_gemini_conversation(
            "generic factual question", [], lambda _query: {"results": []}
        )

    second_call = client.models.generate_content.call_args_list[1]
    second_contents = second_call.kwargs["contents"]
    final_config = second_call.kwargs["config"]
    assert second_contents[-2] is original_content
    assert second_contents[-2].parts[0].thought_signature == b"signed-by-provider"
    assert final_config.tools
    assert final_config.tool_config.function_calling_config.mode == "NONE"
    assert result.tool_called is True


def test_post_tool_provider_error_preserves_tool_diagnostics(monkeypatch):
    first, _ = _native_tool_response()
    with patch("llm_answer_generator.genai.Client") as client_factory:
        client = client_factory.return_value
        client.models.generate_content.side_effect = [first, TimeoutError()]
        tool_output = {"results": [{"id": "knowledge-1"}]}
        result = llm._generate_gemini_conversation(
            "generic factual question", [], lambda _query: tool_output
        )

    assert result.status == llm.PROVIDER_UNAVAILABLE
    assert result.error_category == "timeout"
    assert result.tool_called is True
    assert result.tool_query == "verified query"
    assert result.tool_output == tool_output
