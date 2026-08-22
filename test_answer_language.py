import json
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

import llm_answer_generator as llm


LANGUAGE_RULE = "Respond in the language of CURRENT_MESSAGE."


def _native_tool_response():
    part = types.Part(function_call=types.FunctionCall(
        name="search_knowledge", args={"query": "verified query"}
    ))
    return SimpleNamespace(
        function_calls=[part.function_call],
        candidates=[SimpleNamespace(content=types.Content(role="model", parts=[part]))],
    )


def test_gemini_initial_and_final_calls_include_language_rule(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    with patch("llm_answer_generator.genai.Client") as client_factory:
        client = client_factory.return_value
        client.models.generate_content.side_effect = [
            _native_tool_response(),
            SimpleNamespace(text="Естественный русский ответ."),
        ]
        result = llm._generate_gemini_conversation(
            "Русский вопрос", [], lambda _query: {"VERIFIED_CONTEXT": "English fact"}
        )

    initial = client.models.generate_content.call_args_list[0].kwargs["config"]
    final = client.models.generate_content.call_args_list[1].kwargs["config"]
    assert LANGUAGE_RULE in initial.system_instruction
    assert LANGUAGE_RULE in final.system_instruction
    assert result.answer == "Естественный русский ответ."


def test_openai_initial_and_final_calls_include_language_rule(monkeypatch):
    calls = []

    def provider_text(provider, instructions, input_text, **_kwargs):
        calls.append((provider, instructions, input_text))
        if len(calls) == 1:
            return json.dumps({"action": "search_knowledge", "query": "verified query"})
        return "Natural English answer."

    monkeypatch.setattr(llm, "generate_provider_text", provider_text)
    result = llm._generate_structured_conversation(
        "openai", "English question", [],
        lambda _query: {"VERIFIED_CONTEXT": "Подтверждённый факт"},
    )

    assert len(calls) == 2
    assert all(LANGUAGE_RULE in instructions for _, instructions, _ in calls)
    assert "CURRENT_MESSAGE:\nEnglish question" in calls[0][2]
    assert "VERIFIED_CONTEXT" in calls[1][2]
    assert result.answer == "Natural English answer."


def test_direct_provider_answers_are_returned_without_translation(monkeypatch):
    expected = "Точный ответ провайдера: DSU 2026."
    monkeypatch.setattr(
        llm,
        "generate_provider_text",
        lambda *_args, **_kwargs: json.dumps({"action": "answer", "answer": expected}),
    )
    result = llm._generate_structured_conversation(
        "openai", "Русский вопрос", [], lambda _query: None
    )
    assert result.answer == expected
