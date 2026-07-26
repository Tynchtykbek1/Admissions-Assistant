from unittest.mock import patch

import pytest

from question_rewriter import rewrite_question


@pytest.mark.parametrize(
    ("history", "current", "rewritten", "required"),
    [
        (
            [
                {"role": "user", "content": "Что делать после начала подачи?"},
                {
                    "role": "assistant",
                    "content": "Нужно заключить договор и начать подготовку документов.",
                },
            ],
            "А кому надо написать?",
            "Кому нужно написать для заключения договора и начала подготовки документов?",
            "заключения договора",
        ),
        (
            [
                {"role": "user", "content": "Когда начинается подача?"},
                {"role": "assistant", "content": "С середины декабря."},
            ],
            "А раньше можно?",
            "Можно ли подать документы раньше середины декабря?",
            "раньше середины декабря",
        ),
        (
            [
                {
                    "role": "user",
                    "content": "Кто может написать рекомендательное письмо?",
                },
                {
                    "role": "assistant",
                    "content": "Научный руководитель или руководитель с работы.",
                },
            ],
            "А на каком языке?",
            "На каком языке должно быть рекомендательное письмо?",
            "рекомендательное письмо",
        ),
    ],
)
def test_contextual_follow_ups_are_rewritten(
    history, current, rewritten, required
):
    with patch(
        "question_rewriter.generate_provider_text",
        return_value=rewritten,
    ) as provider:
        result = rewrite_question(current, history)
    provider.assert_called_once()
    assert result.rewrite_used is True
    assert required in result.standalone_question


def test_standalone_question_skips_rewrite_provider():
    question = "Какие документы нужны для студенческой визы?"
    with patch("question_rewriter.generate_provider_text") as provider:
        result = rewrite_question(question, [{"role": "user", "content": "Earlier"}])
    provider.assert_not_called()
    assert result.standalone_question == question
    assert result.rewrite_used is False


def test_ambiguous_follow_up_without_history_is_preserved():
    question = "А кому написать?"
    with patch("question_rewriter.generate_provider_text") as provider:
        result = rewrite_question(question, [])
    provider.assert_not_called()
    assert result.standalone_question == question


@pytest.mark.parametrize("bad_rewrite", ["", "• The answer is X", "x" * 501, None])
def test_rewrite_failure_falls_back_to_original(bad_rewrite):
    question = "А раньше можно?"
    history = [{"role": "assistant", "content": "С середины декабря."}]
    with patch(
        "question_rewriter.generate_provider_text",
        return_value=bad_rewrite,
    ):
        result = rewrite_question(question, history)
    assert result.standalone_question == question
    assert result.rewrite_used is False
