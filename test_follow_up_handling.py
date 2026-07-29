from unittest.mock import patch

import pytest

from llm_answer_generator import _build_answer_input
from question_rewriter import (
    is_likely_follow_up,
    rewrite_question,
    select_rewrite_history,
)


HISTORY = [
    {"role": "user", "content": "Когда начинается подача?"},
    {"role": "assistant", "content": "Ответ из документа."},
]


@pytest.mark.parametrize("question", [
    "Дедлайны напиши",
    "Какие сроки подачи?",
    "А документы для визы какие нужны?",
    "А IELTS обязателен?",
    "А стипендия есть?",
    "И какие документы нужны для поступления?",
    "Документы для визы",
    "Application deadlines?",
    "And what documents are required for the visa?",
    "Is IELTS required?",
])
def test_clear_admissions_questions_are_standalone(question):
    assert is_likely_follow_up(question, HISTORY) is False
    with patch("question_rewriter.generate_provider_text") as provider:
        result = rewrite_question(question, HISTORY)
    provider.assert_not_called()
    assert result.standalone_question == question


@pytest.mark.parametrize("question", [
    "А сроки?",
    "А для визы?",
    "А раньше можно?",
    "А на каком языке?",
    "Какие именно?",
    "Что из них?",
    "А после приезда?",
    "А сколько?",
    "What about the visa?",
    "And the deadline?",
    "Which ones?",
    "What about after arrival?",
    "Is it mandatory?",
])
def test_ambiguous_contextual_questions_are_follow_ups_only_with_history(question):
    assert is_likely_follow_up(question, HISTORY) is True
    assert is_likely_follow_up(question, []) is False


def test_follow_up_calls_provider_once_with_minimal_recent_history():
    history = [
        {"role": "user", "content": "Unrelated old tuition question"},
        {"role": "assistant", "content": "Unrelated old tuition answer"},
        {"role": "user", "content": "Какие документы нужны?"},
        {"role": "assistant", "content": "Паспорт и анкета."},
    ]
    with patch(
        "question_rewriter.generate_provider_text",
        return_value="Какие сроки подачи документов?",
    ) as provider:
        result = rewrite_question("А сроки?", history)
    provider.assert_called_once()
    prompt = provider.call_args.args[2]
    assert "Какие документы нужны?" in prompt
    assert "Паспорт и анкета." in prompt
    assert "А сроки?" in prompt
    assert result.standalone_question == "Какие сроки подачи документов?"


def test_history_selection_is_chronological_bounded_and_recent():
    history = []
    for index in range(5):
        history.extend([
            {"role": "user", "content": f"user-{index}"},
            {"role": "assistant", "content": f"assistant-{index}"},
        ])
    selected = select_rewrite_history(history)
    assert [item["content"] for item in selected] == [
        "user-4", "assistant-4"
    ]


def test_rewrite_failure_falls_back_without_blocking():
    with patch("question_rewriter.generate_provider_text", side_effect=TimeoutError):
        result = rewrite_question("А сроки?", HISTORY)
    assert result.standalone_question == "А сроки?"
    assert result.rewrite_used is False


def test_final_factual_input_contains_only_standalone_question_and_context():
    prompt = _build_answer_input(
        "А сроки?",
        "Какие сроки подачи документов?",
        [{"role": "assistant", "content": "RAW OLD ANSWER"}],
        "ACCEPTED CONTEXT",
    )
    assert "Какие сроки подачи документов?" in prompt
    assert "ACCEPTED CONTEXT" in prompt
    assert "RAW OLD ANSWER" not in prompt
    assert "А сроки?" not in prompt
