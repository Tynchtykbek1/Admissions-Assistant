import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import live_conversation_smoke as smoke


def test_resolves_configured_system_document_instead_of_newest():
    requested = []
    database = SimpleNamespace(
        get_document=lambda document_id: requested.append(document_id) or {
            "id": document_id
        },
        count_document_chunks=lambda document_id: 2 if document_id == 13 else 0,
    )
    settings = SimpleNamespace(
        SYSTEM_DOCUMENT_ID=13,
        SYSTEM_DOCUMENT_ID_INVALID=False,
    )

    assert smoke.resolve_system_document(database, settings) == 13
    assert requested == [13]


@pytest.mark.parametrize(
    ("document_id", "invalid", "document", "chunk_count", "message"),
    [
        (None, False, None, 0, "not configured"),
        (None, True, None, 0, "positive integer"),
        (13, False, None, 0, "does not exist"),
        (13, False, {"id": 13}, 0, "has no chunks"),
    ],
)
def test_invalid_configured_document_fails_before_provider_and_preserves_source(
    tmp_path, monkeypatch, capsys, document_id, invalid, document, chunk_count, message,
):
    source = tmp_path / "source.db"
    original = b"immutable-source-database"
    source.write_bytes(original)
    provider_calls = []
    database = SimpleNamespace(
        get_document=lambda _document_id: document,
        count_document_chunks=lambda _document_id: chunk_count,
    )
    rag_service = SimpleNamespace(
        answer_conversation_question=lambda **_kwargs: provider_calls.append(True)
    )
    settings = SimpleNamespace(
        SYSTEM_DOCUMENT_ID=document_id,
        SYSTEM_DOCUMENT_ID_INVALID=invalid,
    )
    monkeypatch.setenv("RUN_LIVE_LLM_TESTS", "1")
    monkeypatch.setenv("DATABASE_PATH", str(source))
    monkeypatch.setitem(sys.modules, "admissions_rag_assistant.app_settings", settings)
    monkeypatch.setitem(sys.modules, "admissions_rag_assistant.database", database)
    monkeypatch.setitem(sys.modules, "admissions_rag_assistant.rag_service", rag_service)

    assert smoke.main() == 2
    assert provider_calls == []
    assert source.read_bytes() == original
    assert message in capsys.readouterr().out


def test_smoke_uses_database_copy_and_leaves_source_untouched(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    original = b"immutable-source-database"
    source.write_bytes(original)
    observed_database_paths = []
    provider_calls = []

    def get_document(document_id):
        observed_database_paths.append(Path(os.environ["DATABASE_PATH"]))
        return {"id": document_id}

    def answer_conversation_question(**kwargs):
        provider_calls.append(kwargs)
        return {
            "conversation_id": "conversation",
            "tool_called": False,
            "retrieval_result_count": 0,
            "verified_context_used": False,
            "status": "success",
            "answer": "Provider text.",
        }

    monkeypatch.setenv("RUN_LIVE_LLM_TESTS", "1")
    monkeypatch.setenv("DATABASE_PATH", str(source))
    monkeypatch.setitem(sys.modules, "admissions_rag_assistant.app_settings", SimpleNamespace(
        SYSTEM_DOCUMENT_ID=13, SYSTEM_DOCUMENT_ID_INVALID=False
    ))
    monkeypatch.setitem(sys.modules, "admissions_rag_assistant.database", SimpleNamespace(
        get_document=get_document,
        count_document_chunks=lambda _document_id: 1,
    ))
    monkeypatch.setitem(sys.modules, "admissions_rag_assistant.rag_service", SimpleNamespace(
        answer_conversation_question=answer_conversation_question
    ))

    assert smoke.main() == 0
    assert len(provider_calls) == len(smoke.QUESTIONS) == 2
    assert all(call["document_id"] == 13 for call in provider_calls)
    assert observed_database_paths[0] != source
    assert observed_database_paths[0].name == "smoke.db"
    assert source.read_bytes() == original
