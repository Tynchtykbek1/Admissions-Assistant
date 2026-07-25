import json
from unittest.mock import patch

import pytest

import llm_answer_generator as llm


VISA_CONTEXT = [
    {
        "chunk_id": 31,
        "faq_id": 31,
        "filename": "FAQ.docx.pdf",
        "text": "Пренролмент — пригласительное письмо для подачи на визу.",
        "score": 0.8,
    },
    {
        "chunk_id": 30,
        "faq_id": 30,
        "filename": "FAQ.docx.pdf",
        "text": "Гарантийное письмо подтверждает финансовую поддержку родителем.",
        "score": 0.7,
    },
]


@pytest.fixture(autouse=True)
def provider_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_MODEL", "fake")


def provider_result(status, answer):
    return json.dumps({"status": status, "answer": answer}, ensure_ascii=False)


def test_visa_document_partial_list_is_explicit_and_supported():
    answer = (
        "В документе нет полного перечня документов для визы, но упоминаются:\n"
        "• пренролмент\n"
        "• гарантийное письмо о финансовой поддержке\n\n"
        "Полный список нужно уточнить в консульстве для вашей ситуации."
    )
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=provider_result("partial_information", answer),
    ):
        result = llm.generate_llm_answer(
            "Какой перечень документов для визы?", VISA_CONTEXT
        )
    assert result.status == llm.PARTIAL_INFORMATION
    assert "нет полного перечня" in result.answer
    assert "пренролмент" in result.answer
    assert "гарантийное письмо" in result.answer


def test_broad_admissions_overview_is_not_generic_fallback():
    context = [{
        "chunk_id": 1,
        "filename": "fixture.txt",
        "text": "Подача идет с декабря по май. Нужны переводы и апостиль.",
        "score": 0.9,
    }]
    answer = "Из документа:\n• подача — с декабря по май\n• нужны переводы и апостиль."
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=provider_result("partial_information", answer),
    ):
        result = llm.generate_llm_answer(
            "Мне нужна информация по поступлению", context
        )
    assert result.status in {llm.SUCCESS, llm.PARTIAL_INFORMATION}
    assert result.answer == answer
    assert result.answer != llm.INSUFFICIENT_INFORMATION_ANSWER


def test_missing_university_list_does_not_invent_universities():
    context = [{
        "chunk_id": 1,
        "filename": "fixture.txt",
        "text": (
            "Подача идет с декабря по май. Для визы нужен пренролмент. "
            "Других сведений о поступлении нет."
        ),
        "score": 0.9,
    }]
    answer = (
        "В загруженном документе нет списка университетов или критериев, "
        "по которым можно определить, в какие университеты вы можете податься."
    )
    with patch(
        "llm_answer_generator.generate_gemini_answer",
        return_value=provider_result("insufficient_document_information", answer),
    ):
        result = llm.generate_llm_answer(
            "На какие университеты могу податься?", context
        )
    assert result.status == llm.INSUFFICIENT_DOCUMENT_INFORMATION
    assert result.answer == answer
    assert "декабр" not in result.answer.casefold()
    assert "пренролмент" not in result.answer.casefold()
    assert not any(
        name in result.answer.casefold()
        for name in ("болон", "сапиенц", "паду", "милан")
    )


def test_instructions_require_direct_facts_and_include_structured_examples():
    instructions = llm.RAG_INSTRUCTIONS
    assert "central information requested" in instructions
    assert "at least one concrete fact" in instructions
    assert "same general admissions topic" in instructions
    assert "do not append adjacent deadlines" in instructions
    assert (
        '{"status":"insufficient_document_information","answer":"В загруженном '
        "документе нет списка университетов"
    ) in instructions
    assert '{"status":"partial_information","answer":"В документе нет полного' in instructions
