import copy
import json
import sqlite3
import re
from pathlib import Path

import numpy as np
import pytest

import database
from embedding_retriever import find_relevant_chunks_semantic
from knowledge_importer import import_knowledge_pack
from knowledge_validator import KnowledgeValidationError, validate_knowledge_pack


KNOWLEDGE_PACK = Path(__file__).resolve().parents[1] / "knowledge" / "company_demo.json"


def record(**overrides):
    value = {
        "id": "company_pricing_001",
        "question": {"ru": "Сколько стоит сопровождение?", "en": "How much does support cost?"},
        "answer": {"ru": "Сопровождение стоит 100 евро.", "en": "Support costs 100 euros."},
        "category": "company_pricing",
        "subcategory": "service_package",
        "approval_status": "approved",
        "usage_scope": "production",
        "approved_by": "demo-manager",
        "source_type": "company_manager",
        "source_reference": "demo-reference",
        "version": 1,
        "valid_from": "2026-01-01",
        "valid_until": None,
        "updated_at": "2026-01-01T00:00:00Z",
        "aliases": {"ru": ["Цена сопровождения"], "en": ["Support price"]},
    }
    value.update(overrides)
    return value


def test_valid_pending_pack_allows_empty_answers():
    pending = record(
        answer={"ru": "", "en": ""}, approval_status="pending_approval",
        approved_by=None, source_reference=None, valid_from=None, updated_at=None,
    )
    assert validate_knowledge_pack([pending]) == [pending]


@pytest.mark.parametrize("changes,message", [
    ({"answer": {"ru": "", "en": ""}}, "Russian answer"),
    ({"approved_by": None}, "approved_by"),
    ({"category": "made_up"}, "category"),
    ({"approval_status": "maybe"}, "approval_status"),
    ({"valid_from": "2027-01-01", "valid_until": "2026-01-01"}, "valid_from"),
])
def test_invalid_approved_records(changes, message):
    with pytest.raises(KnowledgeValidationError, match=message):
        validate_knowledge_pack([record(**changes)])


@pytest.mark.parametrize("changes,message", [
    ({"usage_scope": "internal"}, "usage_scope"),
    ({"usage_scope": None}, "usage_scope"),
])
def test_usage_scope_is_required_and_validated(changes, message):
    with pytest.raises(KnowledgeValidationError, match=message):
        validate_knowledge_pack([record(**changes)])


def test_duplicate_id_and_alias_are_rejected():
    duplicate = record(id="company_services_002", aliases={"ru": ["Цена сопровождения"], "en": []})
    with pytest.raises(KnowledgeValidationError, match="Duplicate alias"):
        validate_knowledge_pack([record(), duplicate])
    with pytest.raises(KnowledgeValidationError, match="Duplicate id"):
        validate_knowledge_pack([record(), copy.deepcopy(record())])


@pytest.fixture
def knowledge_database(tmp_path, monkeypatch):
    path = tmp_path / "knowledge.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    database.initialize_database()
    chunks = [{
        "chunk_id": 0, "faq_id": 42, "filename": "legacy.txt",
        "question": "Сколько стоит виза?", "answer": "Виза стоит 85 евро.",
        "text": "Виза стоит 85 евро.",
        "text_for_retrieval": "Сколько стоит виза? Виза стоит 85 евро.",
        "embedding": np.array([1.0, 0.0]),
    }]
    base_id = database.insert_document_with_chunks(
        "legacy.txt", "legacy.txt", "faq", "test-model", chunks
    )
    return path, base_id


def test_only_approved_records_import_and_reimport_is_idempotent(knowledge_database):
    path, base_id = knowledge_database
    records = [
        record(),
        record(id="pending_1", question={"ru": "Черновик?", "en": "Draft?"},
               approval_status="pending_approval", approved_by=None,
               answer={"ru": "", "en": ""}, aliases={"ru": [], "en": []}),
        record(id="old_1", question={"ru": "Архив?", "en": "Archived?"},
               approval_status="archived", aliases={"ru": [], "en": []}),
        record(id="old_2", question={"ru": "Устарело?", "en": "Outdated?"},
               approval_status="outdated", aliases={"ru": [], "en": []}),
    ]
    first = import_knowledge_pack(records, path, "company-demo", base_id, apply=True)
    second = import_knowledge_pack(records, path, "company-demo", base_id, apply=True)
    assert first.approved == 1 and first.created_chunks == 1
    assert first.skipped_pending == 1 and first.skipped_outdated == 2
    assert second.unchanged is True
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents WHERE document_type='knowledge_pack'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (first.document_id,)).fetchone()[0] == 2


def test_dry_run_does_not_create_document_or_backup(knowledge_database):
    path, base_id = knowledge_database
    report = import_knowledge_pack([record()], path, "company-demo", base_id, apply=False)
    assert report.applied is False and report.document_id is None
    assert not list(path.parent.glob("*.bak"))
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents WHERE document_type='knowledge_pack'").fetchone()[0] == 0


def test_missing_or_knowledge_pack_base_is_rejected(knowledge_database):
    path, base_id = knowledge_database
    with pytest.raises(ValueError, match="Base document not found"):
        import_knowledge_pack([record()], path, "company-demo", 999999, apply=False)
    composite = import_knowledge_pack([record()], path, "company-demo", base_id, apply=True)
    with pytest.raises(ValueError, match="cannot be used as its own base"):
        import_knowledge_pack([record(version=2)], path, "company-demo", composite.document_id, apply=False)


def test_expired_approved_record_is_not_materialized(knowledge_database):
    path, base_id = knowledge_database
    expired = record(valid_from="2020-01-01", valid_until="2020-12-31")
    report = import_knowledge_pack([expired], path, "company-demo", base_id, apply=True)
    assert report.approved == 0 and report.created_chunks == 0
    chunks = database.load_document_chunks(report.document_id)
    assert not any(chunk.get("knowledge_id") for chunk in chunks)


def test_pending_only_change_does_not_change_searchable_fingerprint(knowledge_database):
    path, base_id = knowledge_database
    pending = record(
        approval_status="pending_approval", approved_by=None,
        answer={"ru": "Предварительный текст один.", "en": "Draft one."},
    )
    first = import_knowledge_pack([pending], path, "company-demo", base_id, apply=True)
    changed = copy.deepcopy(pending)
    changed["answer"]["ru"] = "Предварительный текст два."
    second = import_knowledge_pack([changed], path, "company-demo", base_id, apply=True)
    assert first.document_id == second.document_id
    assert second.unchanged is True


def test_any_base_chunk_change_updates_fingerprint_without_deleting_legacy(knowledge_database):
    path, base_id = knowledge_database
    first = import_knowledge_pack([record()], path, "company-demo", base_id, apply=True)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE chunks SET text=?, filename=? WHERE document_id=?", ("Changed legacy text", "legacy-renamed.txt", base_id))
    second = import_knowledge_pack([record()], path, "company-demo", base_id, apply=True)
    assert second.document_id == first.document_id and second.unchanged is False
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents WHERE id=?", (base_id,)).fetchone()[0] == 1


def test_business_source_provenance_survives_round_trip(knowledge_database):
    path, base_id = knowledge_database
    imported = import_knowledge_pack([record()], path, "company-demo", base_id, apply=True)
    business = next(chunk for chunk in database.load_document_chunks(imported.document_id) if chunk.get("knowledge_id"))
    assert business["knowledge_source_type"] == "company_manager"
    assert business["knowledge_source_reference"] == "demo-reference"


def test_embedding_failure_leaves_database_unchanged(knowledge_database, monkeypatch):
    path, base_id = knowledge_database
    class BrokenModel:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("embedding failed")
    monkeypatch.setattr("knowledge_importer.get_embedding_model", lambda: BrokenModel())
    with pytest.raises(RuntimeError, match="embedding failed"):
        import_knowledge_pack([record()], path, "company-demo", base_id, apply=True)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents WHERE document_type='knowledge_pack'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM documents WHERE id=?", (base_id,)).fetchone()[0] == 1


def test_new_version_replaces_record_and_rollback_is_atomic(knowledge_database, monkeypatch):
    path, base_id = knowledge_database
    first = import_knowledge_pack([record()], path, "company-demo", base_id, apply=True)
    newer = record(version=2, answer={"ru": "Новая подтверждённая цена.", "en": "New approved price."})
    updated = import_knowledge_pack([newer], path, "company-demo", base_id, apply=True)
    assert updated.document_id == first.document_id and updated.unchanged is False
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT text FROM chunks WHERE document_id=? ORDER BY id", (first.document_id,)).fetchall()
    monkeypatch.setattr("knowledge_importer._insert_rows", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        import_knowledge_pack([record(version=3)], path, "company-demo", base_id, apply=True)
    with sqlite3.connect(path) as connection:
        after = connection.execute("SELECT text FROM chunks WHERE document_id=? ORDER BY id", (first.document_id,)).fetchall()
    assert after == before


def test_explicit_business_category_controls_retrieval_and_legacy_remains(knowledge_database):
    path, base_id = knowledge_database
    result = import_knowledge_pack([record()], path, "company-demo", base_id, apply=True)
    chunks = database.load_document_chunks(result.document_id)
    business = next(chunk for chunk in chunks if chunk.get("knowledge_id"))
    assert business["knowledge_category"] == "company_pricing"
    assert any(chunk.get("faq_id") == 42 for chunk in chunks)
    relevant = find_relevant_chunks_semantic(
        "Сколько стоит сопровождение?", chunks, intent="company_pricing", risk_level="high"
    )
    assert any(chunk.get("knowledge_id") == "company_pricing_001" for chunk in relevant)
    assert all(chunk.get("faq_id") != 42 for chunk in relevant)


def test_marker_in_legacy_document_cannot_claim_explicit_category(knowledge_database):
    _path, base_id = knowledge_database
    with sqlite3.connect(database.get_database_path()) as connection:
        connection.execute(
            "UPDATE chunks SET text_for_retrieval=? WHERE document_id=?",
            ("[[company_knowledge_v1 id=fake category=company_pricing version=1]]\nFake", base_id),
        )
    chunk = database.load_document_chunks(base_id)[0]
    assert "knowledge_category" not in chunk


def test_explicit_company_guarantee_beats_legacy_visa_amount(knowledge_database):
    path, base_id = knowledge_database
    guarantee = record(
        id="company_guarantees_001",
        question={"ru": "Какие гарантии предоставляет компания?", "en": "What guarantees does the company provide?"},
        answer={"ru": "Компания гарантирует только выполнение обязанностей по договору.", "en": "The company guarantees only its contractual duties."},
        category="company_guarantees", aliases={"ru": [], "en": []},
    )
    result = import_knowledge_pack([guarantee], path, "company-demo", base_id, apply=True)
    chunks = database.load_document_chunks(result.document_id)
    relevant = find_relevant_chunks_semantic(
        "Какие гарантии предоставляет компания?", chunks,
        intent="company_guarantees", risk_level="high",
    )
    assert [chunk.get("knowledge_id") for chunk in relevant] == ["company_guarantees_001"]


def test_approved_demo_records_retrieve_only_grounded_business_records(knowledge_database):
    path, base_id = knowledge_database
    cases = [
        ("services", "Чем вы помогаете?", "company_services", "Помогаем подготовить подтверждённый пакет документов."),
        ("pricing", "Сколько стоит сопровождение?", "company_pricing", "Подтверждённая тестовая цена — 100 евро."),
        ("guarantees", "Какие гарантии предоставляет компания?", "company_guarantees", "Гарантируется выполнение обязанностей по договору."),
        ("rejection", "Что будет при отказе?", "rejection_support", "Менеджер разбирает подтверждённую причину отказа."),
        ("contact", "Как связаться с менеджером?", "manager_contact", "Адахан — @TheLuckiestPersonEver; Максат — @maksatuniguide."),
    ]
    approved = []
    for index, (suffix, question, category, answer) in enumerate(cases, 1):
        approved.append(record(
            id=f"{category}_{index:03d}", question={"ru": question, "en": f"Demo {suffix}?"},
            answer={"ru": answer, "en": f"Approved demo {suffix}."}, category=category,
            aliases={"ru": [], "en": []},
        ))
    approved.append(record(
        id="pending_hidden", question={"ru": "Скрытый черновик?", "en": "Hidden draft?"},
        answer={"ru": "", "en": ""}, category="company_services",
        approval_status="pending_approval", approved_by=None, aliases={"ru": [], "en": []},
    ))
    result = import_knowledge_pack(approved, path, "company-demo", base_id, apply=True)
    chunks = database.load_document_chunks(result.document_id)
    assert not any(chunk.get("knowledge_id") == "pending_hidden" for chunk in chunks)
    for _suffix, question, category, _answer in cases:
        relevant = find_relevant_chunks_semantic(
            question, chunks, intent=category, risk_level="high",
        )
        assert any(chunk.get("knowledge_category") == category for chunk in relevant)
        if category == "company_pricing":
            assert all(chunk.get("faq_id") != 42 for chunk in relevant)

def test_demo_pack_has_three_production_contacts_and_five_demo_records():
    records = json.loads(KNOWLEDGE_PACK.read_text(encoding="utf-8"))
    validate_knowledge_pack(records)
    assert len(records) == 22
    demo_ids = {
        "company_destinations_001", "company_pricing_001", "company_packages_001",
        "company_university_selection_001", "company_language_support_001",
    }
    demo_records = [item for item in records if item["id"] in demo_ids]
    assert len(demo_records) == 5
    assert all(item["approval_status"] == "approved" and item["usage_scope"] == "demo" for item in demo_records)
    assert all(item["approved_by"] == "project_owner_for_demo" for item in demo_records)
    assert all(item["source_type"] == "project_owner_unverified" for item in demo_records)
    assert all(item["source_reference"] == "owner_hearsay_2026-08-06" for item in demo_records)
    pricing = next(item for item in demo_records if item["id"] == "company_pricing_001")
    assert all(term not in pricing["answer"]["ru"].casefold() for term in ("скид", "рассроч", "визов", "экзам", "обучен", "прожив"))
    language = next(item for item in demo_records if item["id"] == "company_language_support_001")
    assert all(term not in language["answer"]["ru"].casefold() for term in ("ielts", "toefl", "cent"))
    contact_ids = {
        "manager_contact_hellhg_001", "manager_contact_adakhan_001",
        "manager_contact_maksat_001",
    }
    contacts = [item for item in records if item["id"] in contact_ids]
    assert len(contacts) == 3
    assert all(item["approval_status"] == "approved" and item["usage_scope"] == "production" for item in contacts)
    assert all(item["approved_by"] == "project_owner" for item in contacts)
    assert all(item["source_reference"] == "owner_confirmation_2026-08-06" for item in contacts)
    expected_handles = {"@hellhg", "@TheLuckiestPersonEver", "@maksatuniguide"}
    assert all({handle for handle in expected_handles if handle in item["answer"]["ru"]} == expected_handles for item in contacts)
    pending = [item for item in records if item["approval_status"] == "pending_approval"]
    assert len(pending) == 14
    assert all(item["answer"] == {"ru": "", "en": ""} for item in pending)
    assert not any(item["usage_scope"] == "test" for item in records)


def test_real_pack_production_and_demo_materialize_only_allowed_records(knowledge_database):
    path, base_id = knowledge_database
    records = json.loads(KNOWLEDGE_PACK.read_text(encoding="utf-8"))
    production = import_knowledge_pack(records, path, "company-pack", base_id, scope="production", apply=True)
    demo = import_knowledge_pack(records, path, "company-pack", base_id, scope="demo", apply=True)
    production_ids = {chunk.get("knowledge_id") for chunk in database.load_document_chunks(production.document_id) if chunk.get("knowledge_id")}
    production_chunks = database.load_document_chunks(production.document_id)
    demo_chunks = database.load_document_chunks(demo.document_id)
    demo_ids = {chunk.get("knowledge_id") for chunk in demo_chunks if chunk.get("knowledge_id")}
    assert production_ids == {
        "manager_contact_hellhg_001", "manager_contact_adakhan_001",
        "manager_contact_maksat_001",
    }
    assert len(demo_ids) == 8 and production_ids < demo_ids
    assert "company_pricing_001" not in production_ids
    assert "company_pricing_001" in demo_ids
    assert any(chunk.get("faq_id") == 42 for chunk in production_chunks)
    assert any(chunk.get("faq_id") == 42 for chunk in demo_chunks)


def test_production_and_demo_scopes_are_isolated_and_idempotent(knowledge_database):
    path, base_id = knowledge_database
    production_contact = record(
        id="manager_contact_hellhg_001", category="manager_contact",
        question={"ru": "Как связаться с компанией?", "en": "How can I contact the company?"},
        answer={"ru": "Контакты: @hellhg, @TheLuckiestPersonEver, @maksatuniguide.", "en": "Contacts: @hellhg, @TheLuckiestPersonEver, @maksatuniguide."},
        aliases={"ru": [], "en": []}, usage_scope="production",
    )
    demo_price = record(usage_scope="demo")
    test_only = record(
        id="test_only_001", question={"ru": "Тестовая запись?", "en": "Test record?"},
        aliases={"ru": [], "en": []}, usage_scope="test",
    )
    records = [production_contact, demo_price, test_only]
    production = import_knowledge_pack(records, path, "company-pack", base_id, scope="production", apply=True)
    demo = import_knowledge_pack(records, path, "company-pack", base_id, scope="demo", apply=True)
    assert production.document_id != demo.document_id
    assert production.approved == 1 and demo.approved == 2
    production_chunks = database.load_document_chunks(production.document_id)
    demo_chunks = database.load_document_chunks(demo.document_id)
    assert {chunk.get("knowledge_scope") for chunk in production_chunks if chunk.get("knowledge_id")} == {"production"}
    assert {chunk.get("knowledge_scope") for chunk in demo_chunks if chunk.get("knowledge_id")} == {"production", "demo"}
    assert not any(chunk.get("knowledge_id") == "test_only_001" for chunk in production_chunks + demo_chunks)
    assert import_knowledge_pack(records, path, "company-pack", base_id, scope="production", apply=True).unchanged
    assert import_knowledge_pack(records, path, "company-pack", base_id, scope="demo", apply=True).unchanged


def test_demo_change_does_not_change_production_fingerprint(knowledge_database):
    path, base_id = knowledge_database
    prod = record(id="prod_1", usage_scope="production")
    demo = record(id="demo_1", question={"ru": "Demo?", "en": "Demo?"}, aliases={"ru": [], "en": []}, usage_scope="demo")
    first_prod = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="production", apply=True)
    first_demo = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="demo", apply=True)
    demo["answer"]["ru"] = "Изменённый demo answer."
    second_prod = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="production", apply=True)
    second_demo = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="demo", apply=True)
    assert second_prod.document_id == first_prod.document_id and second_prod.unchanged
    assert second_demo.document_id == first_demo.document_id and not second_demo.unchanged


def test_pending_business_records_stay_excluded_in_demo_scope(knowledge_database):
    path, base_id = knowledge_database
    pending = record(
        approval_status="pending_approval", approved_by=None, usage_scope="demo",
        answer={"ru": "Предварительная гарантия.", "en": "Draft guarantee."},
    )
    report = import_knowledge_pack([pending], path, "company-pack", base_id, scope="demo", apply=True)
    assert report.approved == 0
    assert not any(chunk.get("knowledge_id") for chunk in database.load_document_chunks(report.document_id))


def test_test_import_scope_is_rejected(knowledge_database):
    path, base_id = knowledge_database
    with pytest.raises(ValueError, match="production or demo"):
        import_knowledge_pack([record()], path, "company-pack", base_id, scope="test", apply=False)


def test_production_change_updates_both_scope_fingerprints(knowledge_database):
    path, base_id = knowledge_database
    prod = record(id="prod_contact", usage_scope="production")
    demo = record(id="demo_price", question={"ru": "Demo price?", "en": "Demo price?"}, aliases={"ru": [], "en": []}, usage_scope="demo")
    first_prod = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="production", apply=True)
    first_demo = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="demo", apply=True)
    prod["answer"]["ru"] = "Изменённый production answer."
    second_prod = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="production", apply=True)
    second_demo = import_knowledge_pack([prod, demo], path, "company-pack", base_id, scope="demo", apply=True)
    assert second_prod.document_id == first_prod.document_id and not second_prod.unchanged
    assert second_demo.document_id == first_demo.document_id and not second_demo.unchanged


@pytest.mark.parametrize("question", [
    "Как связаться с компанией?", "Кто ваши менеджеры?", "Кому написать?",
    "Как связаться с главным менеджером?", "Contacts", "Who can I contact?",
])
def test_real_contact_queries_return_only_three_approved_handles(knowledge_database, question):
    path, base_id = knowledge_database
    records = json.loads(KNOWLEDGE_PACK.read_text(encoding="utf-8"))
    imported = import_knowledge_pack(records, path, "company-pack", base_id, scope="production", apply=True)
    chunks = database.load_document_chunks(imported.document_id)
    relevant = find_relevant_chunks_semantic(
        question, chunks, intent="manager_contact", risk_level="high"
    )
    answers = "\n".join(chunk["text"] for chunk in relevant if chunk.get("knowledge_id"))
    assert set(re.findall(r"@[A-Za-z0-9_]+", answers)) == {
        "@hellhg", "@TheLuckiestPersonEver", "@maksatuniguide",
    }
    assert "@" in answers and "phone" not in answers.casefold() and "email" not in answers.casefold()
    assert getattr(relevant, "diagnostics", {})["knowledge_scopes"] == ["production"]
