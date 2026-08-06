from unittest.mock import patch

import pytest

from conversation_router import route_conversation


PRICING_HISTORY = [
    {"role": "user", "content": "Сколько стоит сопровождение?"},
    {"role": "assistant", "content": "В подтверждённой базе пока нет информации о стоимости."},
]


@pytest.mark.parametrize("question", [
    "Что входит в эту цену?",
    "А какие гарантии?",
    "А для магистратуры?",
    "And what is included?",
    "What about guarantees?",
    "Is that mandatory?",
    "How long does it take?",
])
def test_required_follow_ups_rewrite_without_provider_dependency(question):
    with patch("question_rewriter.generate_provider_text", return_value=None):
        route = route_conversation(question, PRICING_HISTORY)
    assert route.is_follow_up is True
    assert route.rewrite_used is True
    assert "сопровожд" in route.standalone_question.casefold()


def test_three_turn_company_regression_preserves_company_service_topic():
    with patch("question_rewriter.generate_provider_text", return_value=None):
        price_route = route_conversation("Что входит в эту цену?", PRICING_HISTORY)
        extended_history = PRICING_HISTORY + [
            {"role": "user", "content": "Что входит в эту цену?"},
            {"role": "assistant", "content": "Подтверждённого ответа пока нет."},
        ]
        guarantee_route = route_conversation("А какие гарантии?", extended_history)
    assert price_route.rewrite_used is True
    assert guarantee_route.rewrite_used is True
    assert "сопровожд" in price_route.standalone_question.casefold()
    assert "цен" in guarantee_route.standalone_question.casefold() or "сопровожд" in guarantee_route.standalone_question.casefold()


def test_topic_switch_does_not_reintroduce_old_pricing_topic():
    history = PRICING_HISTORY + [
        {"role": "user", "content": "Что такое бакалавриат?"},
        {"role": "assistant", "content": "Бакалавриат — уровень образования."},
    ]
    with patch("question_rewriter.generate_provider_text", return_value=None):
        route = route_conversation("А какие гарантии?", history)
    assert route.rewrite_used is True
    assert "бакалавр" in route.standalone_question.casefold()
    assert "цен" not in route.standalone_question.casefold()
    assert "сопровожд" not in route.standalone_question.casefold()


def test_deterministic_rewrite_preserves_refinements_since_latest_topic_switch():
    history = [
        {"role": "user", "content": "Какие документы нужны?"},
        {"role": "user", "content": "Нет, я про визу."},
        {"role": "user", "content": "А для магистратуры?"},
    ]
    with patch("question_rewriter.generate_provider_text", return_value=None):
        mandatory = route_conversation("Это обязательно?", history)
        duration = route_conversation(
            "Сколько это занимает?",
            history + [{"role": "user", "content": "Это обязательно?"}],
        )
    for route in (mandatory, duration):
        standalone = route.standalone_question.casefold()
        assert "документ" in standalone
        assert "виз" in standalone
        assert "магистрат" in standalone


@pytest.mark.parametrize("question", [
    "А вместе с визовыми расходами?",
    "А обучение сюда входит?",
    "Какие гарантии по этому пакету?",
])
def test_package_refinements_are_follow_ups(question):
    with patch("question_rewriter.generate_provider_text", return_value=None):
        route = route_conversation(question, PRICING_HISTORY)
    assert route.is_follow_up is True
    assert route.rewrite_used is True
    assert "сопровожд" in route.standalone_question.casefold()
