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


def test_retrieval_uses_rewritten_standalone_question(isolated_database):
    document_id = add_document("guide.txt", "Напишите менеджеру для договора.")
    conversation = database.get_or_create_conversation(
        "telegram", "chat", "user", default_document_id=document_id
    )
    database.add_message(conversation["id"], "user", "Что делать?")
    database.add_message(
        conversation["id"], "assistant", "Заключить договор с менеджером."
    )
    rewritten = "Кому написать для заключения договора?"

    with (
        patch(
            "rag_service.rewrite_question",
            return_value=RewriteResult(rewritten, True),
        ),
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

    assert retrieval.call_args.kwargs["question"] == rewritten
    assert response["standalone_question"] == rewritten


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


def test_answer_uses_bounded_canonical_history_without_current_duplicate(
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
        captured_history.extend(kwargs["history"])
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
    assert len(captured_history) == rag_service.CHAT_HISTORY_LIMIT
    assert captured_history[0]["content"] == "previous-2"
    assert all(item["content"] != "current-message" for item in captured_history)
