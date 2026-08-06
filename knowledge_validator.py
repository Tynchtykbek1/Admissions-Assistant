from __future__ import annotations

from datetime import datetime
from typing import Any


ALLOWED_CATEGORIES = frozenset({
    "company_services", "company_pricing", "company_package",
    "company_contract", "company_guarantees", "refund",
    "rejection_support", "onboarding", "client_responsibilities",
    "company_responsibilities", "manager_contact",
})
ALLOWED_STATUSES = frozenset({
    "draft", "pending_approval", "approved", "outdated", "archived",
})
ALLOWED_USAGE_SCOPES = frozenset({"production", "demo", "test"})
ALLOWED_FIELDS = frozenset({
    "id", "question", "answer", "category", "subcategory",
    "approval_status", "approved_by", "source_type", "source_reference",
    "version", "valid_from", "valid_until", "updated_at", "aliases",
    "usage_scope",
})


class KnowledgeValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _date(value: Any, field: str, index: int, errors: list[str]):
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(f"Record {index}: {field} must be an ISO date or null")
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"Record {index}: invalid {field}")
        return None


def _localized(value: Any, field: str, index: int, errors: list[str]) -> dict:
    if not isinstance(value, dict) or set(value) - {"ru", "en"}:
        errors.append(f"Record {index}: {field} must contain only ru/en fields")
        return {}
    if any(not isinstance(item, str) for item in value.values()):
        errors.append(f"Record {index}: {field} values must be strings")
        return {}
    return value


def validate_knowledge_pack(data: Any, *, allow_test_scope: bool = False) -> list[dict]:
    if not isinstance(data, list):
        raise KnowledgeValidationError(["Knowledge pack root must be a JSON array"])
    errors: list[str] = []
    ids: set[str] = set()
    phrases: dict[tuple[str, str], str] = {}
    for index, record in enumerate(data, 1):
        if not isinstance(record, dict):
            errors.append(f"Record {index}: entry must be an object")
            continue
        unknown = sorted(set(record) - ALLOWED_FIELDS)
        if unknown:
            errors.append(f"Record {index}: unknown fields: {', '.join(unknown)}")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append(f"Record {index}: id is required")
            record_id = f"record-{index}"
        elif record_id in ids:
            errors.append(f"Duplicate id: {record_id}")
        ids.add(record_id)
        if record.get("category") not in ALLOWED_CATEGORIES:
            errors.append(f"Record {index}: invalid category")
        status = record.get("approval_status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"Record {index}: invalid approval_status")
        usage_scope = record.get("usage_scope")
        if usage_scope not in ALLOWED_USAGE_SCOPES:
            errors.append(f"Record {index}: invalid or missing usage_scope")
        elif usage_scope == "test" and not allow_test_scope:
            errors.append(f"Record {index}: test usage_scope is allowed only in automated tests")
        question = _localized(record.get("question"), "question", index, errors)
        answer = _localized(record.get("answer"), "answer", index, errors)
        if not question.get("ru", "").strip():
            errors.append(f"Record {index}: Russian question is required")
        if status == "approved":
            if not answer.get("ru", "").strip():
                errors.append(f"Record {index}: Russian answer is required for approved record")
            if not isinstance(record.get("approved_by"), str) or not record["approved_by"].strip():
                errors.append(f"Record {index}: approved_by is required for approved record")
        version = record.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            errors.append(f"Record {index}: version must be a positive integer")
        start = _date(record.get("valid_from"), "valid_from", index, errors)
        end = _date(record.get("valid_until"), "valid_until", index, errors)
        if start and end and start > end:
            errors.append(f"Record {index}: valid_from must not be later than valid_until")
        aliases = record.get("aliases")
        if not isinstance(aliases, dict) or set(aliases) != {"ru", "en"}:
            errors.append(f"Record {index}: aliases must contain ru and en lists")
            aliases = {"ru": [], "en": []}
        for language in ("ru", "en"):
            values = aliases.get(language, [])
            if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
                errors.append(f"Record {index}: aliases.{language} must be a list of non-empty strings")
                continue
            candidates = ([question.get(language, "")] if question.get(language) else []) + values
            for phrase in candidates:
                key = (language, " ".join(phrase.casefold().split()))
                previous = phrases.get(key)
                if previous is not None:
                    errors.append(f"Duplicate alias/question for {language}: {phrase} ({previous}, {record_id})")
                else:
                    phrases[key] = record_id
    if errors:
        raise KnowledgeValidationError(errors)
    return data
