import pytest

from llm_answer_generator import RAG_INSTRUCTIONS, _build_answer_input


@pytest.mark.parametrize(
    ("question", "expected", "unexpected"),
    [
        ("Какие документы нужны для визы?", "Russian. Answer in Russian.", "Answer in English."),
        ("Нужен IELTS?", "Russian. Answer in Russian.", "Answer in English."),
        ("What documents are required for a student visa?", "English. Answer in English.", "Answer in Russian."),
    ],
)
def test_final_question_determines_required_answer_language(
    question, expected, unexpected
):
    prompt = _build_answer_input(question, question, None, "Retrieved context")
    assert expected in prompt
    assert unexpected not in prompt


def test_english_question_with_russian_context_requires_english():
    prompt = _build_answer_input(
        "What documents are required for a student visa?",
        None,
        None,
        "Для визы нужны документы на русском языке.",
    )
    assert "English. Answer in English." in prompt


def test_russian_question_with_english_context_requires_russian():
    prompt = _build_answer_input(
        "Какие документы нужны для студенческой визы?",
        None,
        None,
        "A passport and admission letter are required.",
    )
    assert "Russian. Answer in Russian." in prompt


@pytest.mark.parametrize(
    ("original", "standalone", "expected"),
    [
        ("What about the visa?", "What documents are required for the visa?", "English. Answer in English."),
        ("А для визы?", "Какие документы нужны для визы?", "Russian. Answer in Russian."),
    ],
)
def test_rewritten_follow_up_language_uses_standalone_question(
    original, standalone, expected
):
    prompt = _build_answer_input(original, standalone, None, "Context")
    assert expected in prompt


def test_provider_instructions_make_language_source_explicit():
    assert "same language as the final standalone question" in RAG_INSTRUCTIONS
    assert "never from retrieved context" in RAG_INSTRUCTIONS
    assert "mainly English question" in RAG_INSTRUCTIONS
    assert "mainly Russian" in RAG_INSTRUCTIONS
    assert '\"status\":\"success|partial_information|insufficient_document_information\"' in RAG_INSTRUCTIONS
    assert '\"answer\":\"concise user-facing answer\"' in RAG_INSTRUCTIONS
