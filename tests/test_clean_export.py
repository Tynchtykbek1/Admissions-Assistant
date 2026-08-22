import csv
import importlib
from datetime import date

import pytest

from scripts.export_unanswered_questions import (
    CLEAN_CSV_FIELDS,
    detect_language,
    export_clean_csv,
    is_noise_question,
)


@pytest.fixture
def export_database(tmp_path, monkeypatch):
    database_path = tmp_path / "export.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    import database
    database = importlib.reload(database)
    database.initialize_database()

    rows = [
        ("ДЕДЛАЙНЫ!!!", 2, 0.20, [9], "no_relevant_chunks", "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
        ("дедлайны?", 3, 0.55, [7, 9], "llm_insufficient_document_information", "2025-12-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"),
        ("Visa?", 1, 0.30, [8], "no_relevant_chunks", "2026-02-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"),
        ("Scholarship?", 4, None, [], "no_relevant_chunks", "2026-01-10T00:00:00+00:00", "2026-04-01T00:00:00+00:00"),
    ]
    for question, count, score, faq_ids, reason, first_seen, last_seen in rows:
        saved = database.record_unanswered_question(
            question=question, standalone_question=question, reason=reason,
            max_similarity_score=score, retrieved_faq_ids=faq_ids,
        )
        with database.get_connection() as connection:
            connection.execute(
                """
                UPDATE unanswered_questions SET occurrence_count = ?,
                  first_seen_at = ?, last_seen_at = ? WHERE id = ?
                """,
                (count, first_seen, last_seen, saved["id"]),
            )
            connection.commit()
    return database, tmp_path


def read_clean(path):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def test_clean_csv_bom_columns_grouping_aggregation_and_database_unchanged(
    export_database,
):
    database, tmp_path = export_database
    before = database.list_unanswered_questions(["open"])
    output = tmp_path / "nested" / "clean.csv"
    summary = export_clean_csv(output, ["open"])
    rows = read_clean(output)
    deadline = next(row for row in rows if "дедлайн" in row["question"].casefold())

    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert tuple(rows[0]) == CLEAN_CSV_FIELDS
    assert deadline["question"] == "дедлайны?"
    assert deadline["times_asked"] == "5"
    assert deadline["first_seen"] == "2025-12-01T00:00:00+00:00"
    assert deadline["last_seen"] == "2026-03-01T00:00:00+00:00"
    assert deadline["best_retrieval_score"] == "0.55"
    assert deadline["related_faq_ids"] == "7,9"
    assert deadline["reason"] == (
        "llm_insufficient_document_information,no_relevant_chunks"
    )
    assert summary["duplicate_variants_merged"] == 1
    assert database.list_unanswered_questions(["open"]) == before


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Дедлайны?", "ru"), ("Visa?", "en"), ("123?!", "unknown")],
)
def test_language_detection(text, expected):
    assert detect_language(text) == expected


def test_clean_filters_and_sorting(export_database):
    _, tmp_path = export_database
    output = tmp_path / "filtered.csv"
    export_clean_csv(
        output, ["open"], min_count=2, language="en",
        since=date(2026, 1, 1), sort_mode="newest",
    )
    assert [row["question"] for row in read_clean(output)] == ["Scholarship?"]

    export_clean_csv(output, ["open"], sort_mode="most-frequent")
    assert [row["times_asked"] for row in read_clean(output)] == ["5", "4", "1"]
    export_clean_csv(output, ["open"], sort_mode="oldest")
    assert read_clean(output)[0]["question"] == "дедлайны?"


@pytest.mark.parametrize("text", [
    "", "   ", "!!!", "😀😀", "https://example.com", "/start", "/help",
    "Привет!", "До свидания", "Спасибо", "Who are you?",
    "What can you do?", "Give me the manager contacts", "Tell me a joke",
])
def test_high_confidence_noise_is_excluded(text):
    assert is_noise_question(text)


@pytest.mark.parametrize("text", [
    "Дедлайны?", "Виза?", "IELTS?", "Апостиль?", "Документы?",
    "Стипендия?", "Deadline?", "Visa?", "Scholarship?",
])
def test_meaningful_short_admissions_questions_remain(text):
    assert not is_noise_question(text)


def test_status_filter_excludes_resolved_and_ignored(export_database):
    database, tmp_path = export_database
    resolved = database.record_unanswered_question(
        question="Resolved question", standalone_question="Resolved question",
        reason="no_relevant_chunks",
    )
    ignored = database.record_unanswered_question(
        question="Ignored question", standalone_question="Ignored question",
        reason="no_relevant_chunks",
    )
    database.mark_unanswered_question_status(resolved["id"], "resolved")
    database.mark_unanswered_question_status(ignored["id"], "ignored")
    output = tmp_path / "open.csv"
    export_clean_csv(output, ["open"])
    questions = {row["question"] for row in read_clean(output)}
    assert "Resolved question" not in questions
    assert "Ignored question" not in questions


def test_atomic_failure_cleans_temporary_file(export_database, monkeypatch):
    _, tmp_path = export_database
    output = tmp_path / "atomic.csv"
    monkeypatch.setattr(
        "scripts.export_unanswered_questions.os.replace",
        lambda *_args: (_ for _ in ()).throw(PermissionError("not writable")),
    )
    with pytest.raises(PermissionError):
        export_clean_csv(output, ["open"])
    assert not output.exists()
    assert list(tmp_path.glob(".atomic.csv.*.tmp")) == []
