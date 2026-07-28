import pytest

from local_responses import LOCAL_RESPONSES, resolve_local_response


@pytest.mark.parametrize(
    ("text", "language", "intent"),
    [
        ("Как тебя зовут?", "ru", "identity"),
        ("КТО ТЫ???", "ru", "identity"),
        ("ты кто", "ru", "identity"),
        ("What is your name?", "en", "identity"),
        ("WHO ARE YOU?", "en", "identity"),
        ("what's your name", "en", "identity"),
        ("Что ты умеешь?", "ru", "capabilities"),
        ("На какие вопросы ты можешь ответить?", "ru", "capabilities"),
        ("чем ты можешь помочь", "ru", "capabilities"),
        ("What can you do?", "en", "capabilities"),
        ("What questions can I ask?", "en", "capabilities"),
        ("How can you help?", "en", "capabilities"),
        ("Кто твой менеджер?", "ru", "manager"),
        ("Как связаться с менеджером?", "ru", "manager"),
        ("Кому написать по поводу поступления?", "ru", "manager"),
        ("Кто может помочь с поступлением?", "ru", "manager"),
        ("Who is your manager?", "en", "manager"),
        ("How can I contact a manager?", "en", "manager"),
        ("Who should I contact about admissions?", "en", "manager"),
        ("Can I speak to a human?", "en", "manager"),
        ("Какая сегодня погода?", "ru", "out_of_scope"),
        ("Расскажи анекдот", "ru", "out_of_scope"),
        ("Сколько будет 2+2?", "ru", "out_of_scope"),
        ("What is the weather?", "en", "out_of_scope"),
        ("Tell me a joke", "en", "out_of_scope"),
    ],
)
def test_resolves_supported_local_intents(text, language, intent):
    assert resolve_local_response(text, language) == LOCAL_RESPONSES[intent][language]


@pytest.mark.parametrize(
    ("text", "language", "intent"),
    [
        ("   КАК   ТЕБЯ   ЗОВУТ???   ", "ru", "identity"),
        ("\tчЕм ты   можешь помочь!!!!!!", "ru", "capabilities"),
        ("  HOW can I contact a MANAGER???  ", "en", "manager"),
        ("...Tell   me a joke!!!", "en", "out_of_scope"),
    ],
)
def test_normalizes_case_punctuation_and_whitespace(text, language, intent):
    assert resolve_local_response(text, language) == LOCAL_RESPONSES[intent][language]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Какие документы нужны для поступления?", "ru"),
        ("Когда начинается подача?", "ru"),
        ("Нужна ли студенческая виза?", "ru"),
        ("Какие документы должен проверить менеджер перед подачей?", "ru"),
        ("В документе указан контакт международного отдела?", "ru"),
        ("What documents should the programme manager verify?", "en"),
        ("Does the application require a manager’s signature?", "en"),
        ("Who is the admissions manager mentioned in the uploaded document?", "en"),
    ],
)
def test_admissions_questions_are_not_intercepted(text, language):
    assert resolve_local_response(text, language) is None
