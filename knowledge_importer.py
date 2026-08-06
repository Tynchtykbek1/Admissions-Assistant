from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

from embedding_model import get_embedding_model, get_embedding_model_name
from knowledge_validator import validate_knowledge_pack


MARKER_PREFIX = "[[company_knowledge_v1"


@dataclass(frozen=True)
class ImportReport:
    approved: int
    skipped_pending: int
    skipped_outdated: int
    created_chunks: int
    generated_embeddings: int
    skipped_scope: int
    scope: str
    applied: bool
    unchanged: bool = False
    document_id: int | None = None
    backup_path: str | None = None


def _approved(records: list[dict], scope: str) -> list[dict]:
    now = datetime.now(timezone.utc)
    selected = []
    for record in records:
        if record["approval_status"] != "approved":
            continue
        allowed_scopes = {"production"} if scope == "production" else {"production", "demo"}
        if record["usage_scope"] not in allowed_scopes:
            continue
        start = record.get("valid_from")
        end = record.get("valid_until")
        start_value = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
        end_value = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else None
        if start_value and start_value.tzinfo is None:
            start_value = start_value.replace(tzinfo=timezone.utc)
        if end_value and end_value.tzinfo is None:
            end_value = end_value.replace(tzinfo=timezone.utc)
        if start_value and start_value > now:
            continue
        if end_value and end_value < now:
            continue
        selected.append(record)
    return selected


def _marker(record: dict) -> str:
    return (
        f"{MARKER_PREFIX} id={record['id']} category={record['category']} "
        f"version={record['version']} "
        f"usage_scope={record['usage_scope']} "
        f"source_type={quote(str(record.get('source_type') or ''), safe='')} "
        f"source_reference={quote(str(record.get('source_reference') or ''), safe='')}]]"
    )


def parse_knowledge_marker(text: str) -> dict | None:
    first = (text or "").splitlines()[0]
    if not first.startswith(MARKER_PREFIX) or not first.endswith("]]" ):
        return None
    values = {}
    for token in first[len(MARKER_PREFIX):-2].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = unquote(value)
    try:
        values["version"] = int(values["version"])
        return values if {"id", "category", "version"} <= set(values) else None
    except (KeyError, ValueError):
        return None


def _business_chunks(records: list[dict], document_name: str) -> list[dict]:
    texts = []
    pending = []
    for index, record in enumerate(records):
        questions = [record["question"].get("ru", ""), record["question"].get("en", "")]
        questions += record["aliases"]["ru"] + record["aliases"]["en"]
        question = " | ".join(item for item in questions if item)
        answers = [record["answer"].get("ru", ""), record["answer"].get("en", "")]
        answer = "\n".join(item for item in answers if item)
        retrieval_text = f"{_marker(record)}\n{question}\n{answer}"
        texts.append(retrieval_text)
        pending.append({
            "chunk_id": index, "faq_id": None, "question": question,
            "answer": answer, "text": answer,
            "text_for_retrieval": retrieval_text, "filename": document_name,
        })
    if pending:
        embeddings = get_embedding_model().encode(texts, normalize_embeddings=True)
        for chunk, embedding in zip(pending, embeddings):
            chunk["embedding"] = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    return pending


def _insert_rows(connection: sqlite3.Connection, document_id: int, rows: list[dict]) -> None:
    for row in rows:
        connection.execute(
            """INSERT INTO chunks(document_id,chunk_id,faq_id,question,answer,text,
               text_for_retrieval,embedding,filename) VALUES(?,?,?,?,?,?,?,?,?)""",
            (document_id, row["chunk_id"], row.get("faq_id"), row.get("question"),
             row.get("answer"), row["text"], row["text_for_retrieval"],
             json.dumps(row["embedding"]), row["filename"]),
        )


def import_knowledge_pack(
    records: list[dict], database_path: Path, document_name: str,
    base_document_id: int, *, scope: str = "production", apply: bool = False,
) -> ImportReport:
    if scope not in {"production", "demo"}:
        raise ValueError("Import scope must be production or demo")
    validate_knowledge_pack(records, allow_test_scope=True)
    database_path = Path(database_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database_path}")
    approved = _approved(records, scope)
    pending_count = sum(item["approval_status"] in {"draft", "pending_approval"} for item in records)
    old_count = sum(item["approval_status"] in {"outdated", "archived"} for item in records)
    approved_in_date = _approved(records, "demo")
    skipped_scope = sum(item not in approved for item in approved_in_date)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        base = connection.execute("SELECT * FROM documents WHERE id=?", (base_document_id,)).fetchone()
        if base is None:
            raise ValueError("Base document not found")
        if base["document_type"] == "knowledge_pack":
            raise ValueError("A knowledge_pack document cannot be used as its own base")
        legacy_rows = [dict(row) for row in connection.execute(
            "SELECT chunk_id,faq_id,question,answer,text,text_for_retrieval,embedding,filename FROM chunks WHERE document_id=? ORDER BY id",
            (base_document_id,),
        )]
        for row in legacy_rows:
            row["embedding"] = json.loads(row["embedding"])
        document_identity = f"{document_name}--{scope}"
        business = _business_chunks(approved, document_identity)
        next_id = max((row["chunk_id"] for row in legacy_rows), default=-1) + 1
        for offset, row in enumerate(business):
            row["chunk_id"] = next_id + offset
        fingerprint_data = {
            "base": legacy_rows,
            "approved": approved,
            "scope": scope,
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        existing = connection.execute(
            "SELECT id,stored_filename FROM documents WHERE filename=? AND document_type='knowledge_pack'",
            (document_identity,),
        ).fetchone()
        report_args = dict(
            approved=len(approved), skipped_pending=pending_count,
            skipped_outdated=old_count, created_chunks=len(business),
            generated_embeddings=len(business),
            skipped_scope=skipped_scope, scope=scope,
        )
        if not apply:
            return ImportReport(**report_args, applied=False, unchanged=bool(existing and existing["stored_filename"] == f"knowledge-pack:{fingerprint}"), document_id=existing["id"] if existing else None)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = database_path.with_name(f"{database_path.name}.{stamp}.bak")
        # SQLite backup API includes committed WAL state; copying only the main
        # file can produce an incomplete backup when WAL mode is enabled.
        with sqlite3.connect(backup) as backup_connection:
            connection.backup(backup_connection)
        connection.execute("BEGIN IMMEDIATE")
        if existing and existing["stored_filename"] == f"knowledge-pack:{fingerprint}":
            connection.rollback()
            return ImportReport(**report_args, applied=True, unchanged=True, document_id=existing["id"], backup_path=str(backup))
        if existing:
            document_id = existing["id"]
            connection.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            connection.execute(
                "UPDATE documents SET stored_filename=?,uploaded_at=?,embedding_model_name=? WHERE id=?",
                (f"knowledge-pack:{fingerprint}", datetime.now(timezone.utc).isoformat(), get_embedding_model_name(), document_id),
            )
        else:
            cursor = connection.execute(
                """INSERT INTO documents(filename,original_filename,stored_filename,document_type,uploaded_at,embedding_model_name)
                   VALUES(?,?,?,?,?,?)""",
                (document_identity, document_identity, f"knowledge-pack:{fingerprint}", "knowledge_pack", datetime.now(timezone.utc).isoformat(), get_embedding_model_name()),
            )
            document_id = cursor.lastrowid
        _insert_rows(connection, document_id, legacy_rows + business)
        connection.commit()
        return ImportReport(**report_args, applied=True, document_id=document_id, backup_path=str(backup))
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
