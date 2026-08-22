import importlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from embedding_retriever import (
    MAX_LEXICAL_RANKING_BONUS,
    build_retrieval_diagnostics,
    calculate_lexical_score,
    find_relevant_chunks_semantic,
    normalize_retrieval_query,
)
from retrieval_settings import (
    CONTEXT_SCORE_MARGIN,
    SEMANTIC_FALLBACK_SCORE_THRESHOLD,
    SEMANTIC_SCORE_THRESHOLD,
    SEMANTIC_TOP_K,
)


class CountingModel:
    def __init__(self):
        self.calls = 0

    def encode(self, text, normalize_embeddings=True):
        self.calls += 1
        return np.array([1.0, 0.0])


def chunk(chunk_id, score, *, faq_id=None, question=None, answer=None):
    result = {
        "chunk_id": chunk_id,
        "filename": "synthetic.txt",
        "text": answer or "standard content",
        "text_for_retrieval": "\n".join(filter(None, (question, answer))) or "standard content",
        "embedding": np.array([score, 0.0]),
    }
    if faq_id is not None:
        result.update(faq_id=faq_id, question=question, answer=answer)
    return result


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        ("Дедлайны напиши", ("сроки", "подачи", "заявления")),
        ("когда подаваться?", ("сроки", "период")),
        ("доки для визы", ("документы", "студенческ", "визы")),
        ("Application deadlines", ("application", "period", "deadline")),
        ("When can I apply?", ("application", "period")),
        ("Student visa documents", ("student", "visa", "documents")),
        ("Is IELTS required?", ("ielts", "language", "certificate")),
    ],
)
def test_short_query_hints_are_explicit_and_fact_free(query, terms):
    expanded = normalize_retrieval_query(query).casefold()
    assert all(term in expanded for term in terms)
    assert not any(value in expanded for value in ("2026", "university of", "band 6"))


def test_lexical_score_is_bounded_and_question_weight_is_stronger():
    question_score = calculate_lexical_score(
        "visa requirements", {"question": "Visa requirements", "answer": ""}
    )[0]
    answer_score = calculate_lexical_score(
        "visa requirements", {"question": "", "answer": "Visa requirements"}
    )[0]
    assert 0 <= answer_score < question_score <= 1
    assert MAX_LEXICAL_RANKING_BONUS == 0.20


def test_lexical_ranking_prefers_specific_faqs_without_bypassing_threshold():
    chunks = [
        chunk(1, 0.50, faq_id=101, question="Application deadlines", answer="Period."),
        chunk(2, 0.50, faq_id=102, question="Student visa", answer="Visa documents."),
        chunk(3, 0.19, faq_id=103, question="Application deadlines", answer="Exact topic."),
    ]
    model = CountingModel()
    with patch("embedding_retriever.get_embedding_model", return_value=model):
        deadline = find_relevant_chunks_semantic(
            "deadline", chunks, min_score=0.20, fallback_score_threshold=None
        )
    assert deadline[0]["faq_id"] == 101
    assert 103 not in [item["faq_id"] for item in deadline]
    assert model.calls == 1


def test_context_margin_keeps_similar_results_and_drops_weak_ones():
    chunks = [chunk(1, 0.50), chunk(2, 0.45), chunk(3, 0.30)]
    with patch("embedding_retriever.get_embedding_model", return_value=CountingModel()):
        results = find_relevant_chunks_semantic(
            "specific", chunks, min_score=0.20, context_score_margin=0.12, top_k=5
        )
    assert [item["chunk_id"] for item in results] == [1, 2]


def test_exact_match_is_preserved_outside_margin_and_no_accepted_candidate():
    chunks = [
        chunk(1, 0.60),
        chunk(2, 0.10, faq_id=202, question="Exact question", answer="Exact answer"),
    ]
    with patch("embedding_retriever.get_embedding_model", return_value=CountingModel()):
        results = find_relevant_chunks_semantic("Exact question", chunks, min_score=0.20)
        none = find_relevant_chunks_semantic("Other", [chunk(3, 0.10)], min_score=0.20)
    assert [item["chunk_id"] for item in results] == [2]
    assert none == []


def test_faq_fields_are_preserved():
    faq = chunk(1, 0.8, faq_id=301, question="Original question?", answer="Original answer.")
    with patch("embedding_retriever.get_embedding_model", return_value=CountingModel()):
        result = find_relevant_chunks_semantic("Original question?", [faq], min_score=0.20)[0]
    assert result["question"] == faq["question"]
    assert result["answer"] == faq["answer"]
    assert result["text_for_retrieval"] == faq["text_for_retrieval"]
    assert "embedding" not in result


def test_deterministic_tie_breaking_uses_faq_or_chunk_id():
    chunks = [
        chunk(8, 0.5, faq_id=20, question="Topic", answer="Answer"),
        chunk(7, 0.5, faq_id=10, question="Topic", answer="Answer"),
    ]
    with patch("embedding_retriever.get_embedding_model", return_value=CountingModel()):
        diagnostics = build_retrieval_diagnostics("Other", chunks)
    assert [item["faq_id"] for item in diagnostics] == [10, 20]


def test_retrieval_constants_and_margin_are_unchanged_or_expected():
    assert SEMANTIC_SCORE_THRESHOLD == 0.20
    assert SEMANTIC_FALLBACK_SCORE_THRESHOLD == 0.18
    assert SEMANTIC_TOP_K == 5
    assert CONTEXT_SCORE_MARGIN == 0.12


def test_context_margin_validation():
    with patch("embedding_retriever.get_embedding_model", return_value=CountingModel()):
        with pytest.raises(ValueError):
            find_relevant_chunks_semantic("x", [chunk(1, 0.5)], context_score_margin=float("nan"))


def test_read_only_index_audit(tmp_path, monkeypatch):
    database_path = tmp_path / "audit.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    import database
    database = importlib.reload(database)
    database.initialize_database()
    document_id = database.insert_document_with_chunks(
        "synthetic.txt", "stored.txt", "faq", database.get_embedding_model_name(),
        [chunk(1, 0.8, faq_id=901, question="Question?", answer="Answer.")],
    )
    before = database_path.read_bytes()
    from scripts.audit_document_index import audit_document_index
    report = audit_document_index(document_id, database_path)
    assert report["faq_chunks"] == 1
    assert report["faq_chunks_missing_required_field"] == 0
    assert database_path.read_bytes() == before


def test_evaluator_reports_expected_metrics(tmp_path, monkeypatch):
    database_path = tmp_path / "evaluation.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    import database
    database = importlib.reload(database)
    database.initialize_database()
    document_id = database.insert_document_with_chunks(
        "synthetic.txt", "stored.txt", "faq", database.get_embedding_model_name(),
        [chunk(1, 0.8, faq_id=1001, question="Application deadlines", answer="Period.")],
    )
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"question": "Application deadlines", "expected_faq_ids": [1001]}
    ]), encoding="utf-8")
    from scripts.evaluate_retrieval import evaluate
    report = evaluate(database_path, document_id, cases)
    assert report["total_cases"] == 1
    assert report["recall_at_1"] == 1.0
    assert report["failures"] == 0
