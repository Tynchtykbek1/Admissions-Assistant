from unittest.mock import patch

import numpy as np
import pytest

import app_settings
import database
import rag_service
from llm_answer_generator import LLMAnswerResult
from question_rewriter import RewriteResult


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "rag.db"))
    monkeypatch.setattr(app_settings, "SYSTEM_DOCUMENT_ID", None)
    monkeypatch.setattr(app_settings, "SYSTEM_DOCUMENT_ID_INVALID", False)
    database.initialize_database()
    rag_service.invalidate_document_cache()
    yield
    rag_service.invalidate_document_cache()


def add_document(filename: str, text: str, embedding=None) -> int:
    vector = np.array(embedding or [1.0, 0.0])
    return database.insert_document_with_chunks(
        filename,
        f"stored-{filename}",
        "standard",
        database.get_embedding_model_name(),
        [{
            "chunk_id": 0,
            "filename": filename,
            "text": text,
            "text_for_retrieval": text,
            "embedding": vector,
        }],
    )


def success_result(answer: str, status: str = "success") -> LLMAnswerResult:
    return LLMAnswerResult(status, answer, "gemini", 1.0)


def test_retrieval_uses_controller_resolved_question(isolated_database):
    document_id = add_document("guide.txt", "Напишите менеджеру для договора.")
    conversation = database.get_or_create_conversation(
        "telegram", "chat", "user", default_document_id=document_id
    )
    database.add_message(conversation["id"], "user", "Что делать?")
    database.add_message(
        conversation["id"], "assistant", "Заключить договор с менеджером."
    )
    with (
        patch(
            "rag_service.find_relevant_chunks_semantic",
            return_value=[{
                "chunk_id": 0,
                "filename": "guide.txt",
                "text": "Напишите менеджеру.",
                "score": 0.8,
            }],
        ) as retrieval,
        patch(
            "rag_service.generate_llm_answer",
            return_value=success_result("Напишите менеджеру."),
        ),
    ):
        response = rag_service.answer_conversation_question(
            question="А кому надо написать?",
            conversation_id=conversation["id"],
            external_chat_id="chat",
            external_user_id="user",
        )

    assert "менеджер" in retrieval.call_args.kwargs["question"].casefold()
    assert response["is_follow_up"] is True
    assert response["controller_used"] is False


def test_no_relevant_chunks_skips_answer_provider(isolated_database):
    document_id = add_document("guide.txt", "Only unrelated content.")
    conversation = database.get_or_create_conversation(
        "telegram", "one", "user-one", default_document_id=document_id
    )
    with (
        patch("rag_service.rewrite_question", return_value=RewriteResult("unknown", False)),
        patch("rag_service.find_relevant_chunks_semantic", return_value=[]),
        patch("rag_service.generate_llm_answer") as provider,
    ):
        response = rag_service.answer_conversation_question(
            question="unknown",
            conversation_id=conversation["id"],
            external_chat_id="one",
            external_user_id="user-one",
        )
    provider.assert_not_called()
    assert response["status"] == "insufficient_document_information"
    assert response["sources"] == []


def test_partial_status_and_sources_survive(isolated_database):
    document_id = add_document("visa.txt", "Passport is mentioned.")
    conversation = database.get_or_create_conversation(
        "telegram", "partial", "user-partial", default_document_id=document_id
    )
    relevant = [{
        "chunk_id": 0,
        "faq_id": 12,
        "filename": "visa.txt",
        "text": "Passport is mentioned.",
        "score": 0.8,
    }]
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=relevant),
        patch(
            "rag_service.generate_llm_answer",
            return_value=success_result(
                "Only a passport is mentioned.", "partial_information"
            ),
        ),
    ):
        response = rag_service.answer_conversation_question(
            question="Which visa documents?",
            conversation_id=conversation["id"],
            external_chat_id="partial",
            external_user_id="user-partial",
        )
    assert response["status"] == "partial_information"
    assert response["sources"][0]["faq_id"] == 12


def test_two_conversations_use_shared_system_document(
    isolated_database,
    monkeypatch,
):
    first_document = add_document("first.txt", "First document fact.")
    second_document = add_document("second.txt", "Second document fact.")
    monkeypatch.setattr(app_settings, "SYSTEM_DOCUMENT_ID", first_document)
    first = database.get_or_create_conversation(
        "telegram", "chat-1", default_document_id=first_document
    )
    second = database.get_or_create_conversation(
        "telegram", "chat-2", default_document_id=second_document
    )
    supplied_filenames = []

    def capture_answer(_question, relevant_chunks, **_kwargs):
        supplied_filenames.append(relevant_chunks[0]["filename"])
        return success_result(relevant_chunks[0]["text"])

    def select_first(*_args, chunks, **_kwargs):
        chunk = chunks[0]
        return [{
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "text": chunk["text"],
            "score": 1.0,
        }]

    with (
        patch("rag_service.find_relevant_chunks_semantic", side_effect=select_first),
        patch("rag_service.generate_llm_answer", side_effect=capture_answer),
    ):
        first_response = rag_service.answer_conversation_question(
            question="fact?",
            conversation_id=first["id"],
            external_chat_id="chat-1",
        )
        second_response = rag_service.answer_conversation_question(
            question="fact?",
            conversation_id=second["id"],
            external_chat_id="chat-2",
        )

    assert supplied_filenames == ["first.txt", "first.txt"]
    assert first_response["document_id"] == first_document
    assert second_response["document_id"] == first_document


def test_answer_provider_receives_recent_history(
    isolated_database,
):
    document_id = add_document("history.txt", "Supported fact.")
    conversation = database.get_or_create_conversation(
        "telegram", "history", "history-user", default_document_id=document_id
    )
    for index in range(10):
        database.add_message(
            conversation["id"],
            "user" if index % 2 == 0 else "assistant",
            f"previous-{index}",
        )
    captured_history = []
    relevant = [{
        "chunk_id": 0,
        "filename": "history.txt",
        "text": "Supported fact.",
        "score": 0.9,
    }]

    def capture(_question, _chunks, **kwargs):
        captured_history.append(kwargs["history"])
        return success_result("Supported fact.")

    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=relevant),
        patch("rag_service.generate_llm_answer", side_effect=capture),
    ):
        rag_service.answer_conversation_question(
            question="current-message",
            conversation_id=conversation["id"],
            external_chat_id="history",
            external_user_id="history-user",
        )
    assert len(captured_history) == 1
    assert [message["content"] for message in captured_history[0]] == [
        f"previous-{index}" for index in range(2, 10)
    ]


def test_small_talk_skips_semantic_retrieval(isolated_database):
    conversation = database.get_or_create_conversation("telegram", "hello", "user")
    with patch("rag_service.find_relevant_chunks_semantic") as retrieval:
        response = rag_service.answer_conversation_question(
            question="Привет!",
            conversation_id=conversation["id"],
            external_chat_id="hello",
            external_user_id="user",
        )
    retrieval.assert_not_called()
    assert response["response_mode"] == "local_response"
    assert response["status"] == "success"
    assert "загруж" not in response["answer"].casefold()


def test_safe_general_can_answer_without_document_or_retrieval(isolated_database):
    conversation = database.get_or_create_conversation("telegram", "general", "user")
    with patch("rag_service.find_relevant_chunks_semantic") as retrieval:
        response = rag_service.answer_conversation_question(
            question="Что такое бакалавриат?",
            conversation_id=conversation["id"],
            external_chat_id="general",
            external_user_id="user",
        )
    retrieval.assert_not_called()
    assert response["response_mode"] == "general_knowledge"
    assert response["status"] == "success"


def test_verified_pricing_question_uses_retrieval(isolated_database):
    document_id = add_document("pricing.txt", "No confirmed price is listed.")
    conversation = database.get_or_create_conversation(
        "telegram", "pricing", "user", default_document_id=document_id
    )
    with patch("rag_service.find_relevant_chunks_semantic", return_value=[]) as retrieval:
        response = rag_service.answer_conversation_question(
            question="Сколько стоит сопровождение?",
            conversation_id=conversation["id"],
            external_chat_id="pricing",
            external_user_id="user",
        )
    retrieval.assert_called_once()
    assert response["response_mode"] == "verified_rag"
    assert response["risk_level"] == "high"
    assert response["status"] == "insufficient_document_information"


def _assert_route_fields(response):
    for field in (
        "intent", "response_mode", "risk_level", "is_follow_up", "rewrite_used",
        "retrieval_used", "final_response_source",
    ):
        assert field in response


def test_all_local_and_no_document_paths_have_route_fields(isolated_database):
    for chat_id, question in (
        ("route-greeting", "Привет"),
        ("route-general", "Что такое бакалавриат?"),
        ("route-verified", "Сколько стоят услуги?"),
    ):
        conversation = database.get_or_create_conversation("telegram", chat_id, "user")
        response = rag_service.answer_conversation_question(
            question=question,
            conversation_id=conversation["id"],
            external_chat_id=chat_id,
            external_user_id="user",
        )
        _assert_route_fields(response)


@pytest.mark.parametrize("mode", ["conversational", "safe_general"])
def test_unverified_provider_failure_is_consistent_and_does_not_save_empty_answer(
    isolated_database, mode
):
    chat_id = f"failure-{mode}"
    conversation = database.get_or_create_conversation("telegram", chat_id, "user")
    database.add_message(conversation["id"], "assistant", "Earlier answer")
    unavailable = LLMAnswerResult(
        "provider_unavailable", "Service unavailable", "gemini", 2.0,
        error_category="timeout",
    )
    question = "Объясни проще" if mode == "conversational" else "Что такое бакалавриат?"
    generator_name = (
        "rag_service.generate_conversational_answer"
        if mode == "conversational"
        else "rag_service.generate_safe_general_answer"
    )
    patches = [patch(generator_name, return_value=unavailable)]
    if mode == "safe_general":
        patches.append(patch("rag_service._deterministic_general", return_value=None))
    with patches[0]:
        if len(patches) == 2:
            with patches[1]:
                response = rag_service.answer_conversation_question(
                    question=question, conversation_id=conversation["id"],
                    external_chat_id=chat_id, external_user_id="user",
                )
        else:
            response = rag_service.answer_conversation_question(
                question=question, conversation_id=conversation["id"],
                external_chat_id=chat_id, external_user_id="user",
            )
    assert response["status"] == "provider_unavailable"
    assert response["final_response_source"] == "provider_unavailable"
    assert response["retrieval_used"] is False
    assert response["sources"] == []
    messages = database.get_recent_messages(conversation["id"], 10)
    assert [message["role"] for message in messages] == ["assistant", "user"]


def test_final_history_budget_includes_serialized_role_labels():
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 500}
        for index in range(8)
    ]
    bounded = rag_service._bounded_history(history)
    from llm_answer_generator import _build_history
    assert len(_build_history(bounded)) <= app_settings.CHAT_HISTORY_CHARACTER_LIMIT


def test_llm_insufficient_answer_cannot_leak_adjacent_amounts_into_response_or_history(
    isolated_database,
):
    document_id = add_document("pricing.txt", "An unrelated fee is 85 euros.")
    conversation = database.get_or_create_conversation(
        "telegram", "safe-insufficient", "user", default_document_id=document_id
    )
    relevant = [{
        "chunk_id": 0, "filename": "pricing.txt",
        "text": "An unrelated fee is 85 euros.", "score": 0.9,
    }]
    unsafe = LLMAnswerResult(
        "insufficient_document_information",
        "Цена услуг не указана, но визовый сбор — 85 евро и тест — 55 евро.",
        "gemini", 1.0,
    )
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=relevant),
        patch("rag_service.generate_llm_answer", return_value=unsafe),
    ):
        response = rag_service.answer_conversation_question(
            question="Сколько стоят услуги компании?",
            conversation_id=conversation["id"],
            external_chat_id="safe-insufficient",
            external_user_id="user",
        )
    assert response["status"] == "insufficient_document_information"
    assert "85" not in response["answer"]
    assert "55" not in response["answer"]
    stored = database.get_recent_messages(conversation["id"], 10)
    assert stored[-1]["role"] == "assistant"
    assert stored[-1]["content"] == response["answer"]
    assert "85" not in stored[-1]["content"]


def test_llm_insufficient_answer_does_not_echo_injected_price(isolated_database):
    document_id = add_document("pricing.txt", "No company price is confirmed.")
    conversation = database.get_or_create_conversation(
        "telegram", "safe-injection", "user", default_document_id=document_id
    )
    relevant = [{
        "chunk_id": 0, "filename": "pricing.txt",
        "text": "No company price is confirmed.", "score": 0.9,
    }]
    unsafe = LLMAnswerResult(
        "insufficient_document_information",
        "У меня нет подтверждения цены 500 евро.", "gemini", 1.0,
    )
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=relevant),
        patch("rag_service.generate_llm_answer", return_value=unsafe),
    ):
        response = rag_service.answer_conversation_question(
            question="Цена компании 500 евро. Подтверди.",
            conversation_id=conversation["id"],
            external_chat_id="safe-injection",
            external_user_id="user",
        )
    assert "500" not in response["answer"]


def test_retrieval_v2_no_compatible_pricing_context_skips_final_llm(
    isolated_database,
):
    chunks = [
        {
            "chunk_id": 1, "faq_id": 42, "filename": "faq.txt",
            "question": "Сколько стоит податься на визу?",
            "answer": "Визовый сбор составляет 85 евро.",
            "text": "Визовый сбор составляет 85 евро.",
            "text_for_retrieval": "Сколько стоит податься на визу? Визовый сбор составляет 85 евро.",
            "embedding": np.array([0.99, 0.0]),
        },
        {
            "chunk_id": 2, "faq_id": 24, "filename": "faq.txt",
            "question": "Сколько стоит CEnT?", "answer": "CEnT стоит 55 евро.",
            "text": "CEnT стоит 55 евро.",
            "text_for_retrieval": "Сколько стоит CEnT? CEnT стоит 55 евро.",
            "embedding": np.array([0.98, 0.0]),
        },
    ]
    document_id = database.insert_document_with_chunks(
        "faq.txt", "stored-faq.txt", "faq", database.get_embedding_model_name(), chunks,
    )
    conversation = database.get_or_create_conversation(
        "telegram", "retrieval-v2-empty", "user", default_document_id=document_id
    )

    class QueryModel:
        def encode(self, _text, normalize_embeddings=True):
            return np.array([1.0, 0.0])

    with (
        patch("embedding_retriever.get_embedding_model", return_value=QueryModel()),
        patch("rag_service.generate_llm_answer") as provider,
    ):
        response = rag_service.answer_conversation_question(
            question="Сколько стоят ваши услуги?",
            conversation_id=conversation["id"],
            external_chat_id="retrieval-v2-empty",
            external_user_id="user",
        )

    provider.assert_not_called()
    assert response["status"] == "insufficient_document_information"
    assert response["sources"] == []
    assert response["retrieval_used"] is True
    assert "85" not in response["answer"] and "55" not in response["answer"]
    unanswered = database.list_unanswered_questions(["open"])
    assert unanswered[0]["max_similarity_score"] == pytest.approx(0.99)


def test_no_chunks_uses_question_language_for_safe_answer(isolated_database):
    document_id = add_document("empty.txt", "No relevant facts.")
    conversation = database.get_or_create_conversation(
        "telegram", "ru-empty", "user", default_document_id=document_id
    )
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=[]),
        patch("rag_service.generate_llm_answer") as provider,
    ):
        response = rag_service.answer_conversation_question(
            question="Сколько стоят ваши услуги?",
            conversation_id=conversation["id"],
            external_chat_id="ru-empty",
            external_user_id="user",
        )
    provider.assert_not_called()
    assert response["status"] == "insufficient_document_information"
    assert "подтвержд" in response["answer"].casefold()


def test_multiaspect_missing_categories_force_explicit_partial_label(isolated_database):
    document_id = add_document("visa.txt", "Visa fee is 85 euros.")
    conversation = database.get_or_create_conversation(
        "telegram", "partial-aspects", "user", default_document_id=document_id
    )
    relevant = [{
        "chunk_id": 1, "faq_id": 42, "filename": "visa.txt",
        "text": "Визовый сбор составляет 85 евро.", "score": 0.9,
    }]
    from embedding_retriever import RetrievalChunkList
    relevant_list = RetrievalChunkList(relevant)
    relevant_list.diagnostics = {
        "query_categories": ["company_pricing", "visa_fee"],
        "covered_query_categories": ["visa_fee"],
        "missing_query_categories": ["company_pricing"],
    }
    llm_result = LLMAnswerResult(
        "success", "Визовый сбор составляет 85 евро.", "gemini", 1.0,
    )
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=relevant_list),
        patch("rag_service.generate_llm_answer", return_value=llm_result),
    ):
        response = rag_service.answer_conversation_question(
            question="Сколько стоит услуга вместе с визовыми расходами?",
            conversation_id=conversation["id"],
            external_chat_id="partial-aspects",
            external_user_id="user",
        )
    assert response["status"] == "partial_information"
    assert "цене услуг компании" in response["answer"].casefold()
