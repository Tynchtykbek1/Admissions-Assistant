import csv
import json
from unittest.mock import patch

import numpy as np
import pytest

import app_settings
import database
import rag_service
from llm_answer_generator import LLMAnswerResult
from question_rewriter import RewriteResult
from scripts.export_unanswered_questions import export_csv


@pytest.fixture
def unanswered_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "unanswered.db"))
    monkeypatch.setattr(app_settings, "SYSTEM_DOCUMENT_ID", None)
    monkeypatch.setattr(app_settings, "SYSTEM_DOCUMENT_ID_INVALID", False)
    database.initialize_database()
    document_id = database.insert_document_with_chunks(
        "faq.txt",
        "stored-faq.txt",
        "faq",
        database.get_embedding_model_name(),
        [{
            "chunk_id": 7,
            "faq_id": 7,
            "question": "Known question?",
            "answer": "Known answer.",
            "filename": "faq.txt",
            "text": "Known answer.",
            "text_for_retrieval": "Known question? Known answer.",
            "embedding": np.array([1.0, 0.0]),
        }],
    )
    conversation = database.get_or_create_conversation(
        "telegram",
        "chat",
        "user",
        default_document_id=document_id,
    )
    rag_service.invalidate_document_cache()
    yield conversation
    rag_service.invalidate_document_cache()


def _ask(conversation: dict, question: str = "Unknown policy?") -> dict:
    return rag_service.answer_conversation_question(
        question=question,
        conversation_id=conversation["id"],
        external_chat_id="chat",
        external_user_id="user",
    )


def _provider_result(status: str) -> LLMAnswerResult:
    return LLMAnswerResult(
        status,
        "Controlled answer.",
        "gemini",
        1.0,
    )


def test_zero_retrieval_and_equivalent_repeat_create_one_record(
    unanswered_database,
):
    rewrites = [
        RewriteResult("Какие документы нужны?", True),
        RewriteResult("  какие   ДОКУМЕНТЫ нужны?  ", True),
    ]
    with (
        patch("rag_service.rewrite_question", side_effect=rewrites),
        patch("rag_service.find_relevant_chunks_semantic", return_value=[]),
        patch("rag_service.generate_llm_answer") as provider,
    ):
        first = _ask(unanswered_database, "Какие документы нужны?")
        second = _ask(unanswered_database, "А какие документы?")

    provider.assert_not_called()
    assert first["status"] == second["status"] == (
        "insufficient_document_information"
    )
    rows = database.list_unanswered_questions(["open"])
    assert len(rows) == 1
    assert rows[0]["question"] == "Какие документы нужны?"
    assert rows[0]["occurrence_count"] == 2
    assert rows[0]["reason"] == "no_relevant_chunks"
    assert rows[0]["max_similarity_score"] is None


def test_llm_insufficient_result_records_retrieval_metadata(unanswered_database):
    relevant = [{
        "chunk_id": 7,
        "faq_id": 7,
        "filename": "faq.txt",
        "text": "Adjacent fact.",
        "score": 0.61,
    }]
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=relevant),
        patch(
            "rag_service.generate_llm_answer",
            return_value=_provider_result("insufficient_document_information"),
        ),
    ):
        response = _ask(unanswered_database)

    assert response["status"] == "insufficient_document_information"
    row = database.list_unanswered_questions(["open"])[0]
    assert row["reason"] == "llm_insufficient_document_information"
    assert row["max_similarity_score"] == pytest.approx(0.61)
    assert json.loads(row["retrieved_faq_ids"]) == [7]


def test_repeated_record_keeps_highest_score_and_unions_faq_ids(
    unanswered_database,
):
    database.record_unanswered_question(
        question="Original wording",
        standalone_question="Same question",
        reason="llm_insufficient_document_information",
        max_similarity_score=0.45,
        retrieved_faq_ids=[2],
    )
    database.record_unanswered_question(
        question="Different wording",
        standalone_question=" same   QUESTION ",
        reason="llm_insufficient_document_information",
        max_similarity_score=0.72,
        retrieved_faq_ids=[3, 2],
    )

    row = database.list_unanswered_questions(["open"])[0]
    assert row["question"] == "Original wording"
    assert row["occurrence_count"] == 2
    assert row["max_similarity_score"] == pytest.approx(0.72)
    assert json.loads(row["retrieved_faq_ids"]) == [2, 3]


@pytest.mark.parametrize(
    "status",
    ["success", "partial_information", "provider_unavailable"],
)
def test_non_unanswered_provider_results_are_not_stored(
    unanswered_database,
    status,
):
    relevant = [{
        "chunk_id": 7,
        "faq_id": 7,
        "filename": "faq.txt",
        "text": "Known answer.",
        "score": 0.9,
    }]
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=relevant),
        patch(
            "rag_service.generate_llm_answer",
            return_value=_provider_result(status),
        ),
    ):
        response = _ask(unanswered_database)

    assert response["status"] == status
    assert database.list_unanswered_questions(["open"]) == []


def test_system_document_failure_is_not_stored(
    unanswered_database,
    monkeypatch,
):
    monkeypatch.setattr(app_settings, "SYSTEM_DOCUMENT_ID", 999)
    response = _ask(unanswered_database)

    assert response["status"] == "system_document_unavailable"
    assert database.list_unanswered_questions(["open"]) == []


def test_recording_failure_does_not_break_response(unanswered_database):
    with (
        patch("rag_service.find_relevant_chunks_semantic", return_value=[]),
        patch(
            "rag_service.record_unanswered_question",
            side_effect=RuntimeError("synthetic storage failure"),
        ),
    ):
        response = _ask(unanswered_database)

    assert response["status"] == "insufficient_document_information"


def test_csv_export_is_utf8_and_filters_status(
    unanswered_database,
    tmp_path,
):
    first = database.record_unanswered_question(
        question="Какие документы нужны?",
        standalone_question="Какие документы нужны?",
        reason="no_relevant_chunks",
    )
    second = database.record_unanswered_question(
        question="What is the fee?",
        standalone_question="What is the fee?",
        reason="no_relevant_chunks",
    )
    database.mark_unanswered_question_status(first["id"], "reviewed")
    database.mark_unanswered_question_status(second["id"], "ignored")

    default_output = tmp_path / "all.csv"
    open_output = tmp_path / "open.csv"
    assert export_csv(default_output) == 1
    assert export_csv(open_output, ["open"]) == 0

    text = default_output.read_text(encoding="utf-8")
    assert "Какие документы нужны?" in text
    with default_output.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["status"] == "reviewed"
    assert set(rows[0]) == {
        "id",
        "question",
        "standalone_question",
        "occurrence_count",
        "max_similarity_score",
        "retrieved_faq_ids",
        "reason",
        "status",
        "first_seen_at",
        "last_seen_at",
    }
