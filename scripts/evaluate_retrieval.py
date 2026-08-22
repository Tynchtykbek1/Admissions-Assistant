import argparse
import json
import os
import sqlite3
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against JSON cases.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    return parser.parse_args()


def _load_chunks_read_only(database: Path, document_id: int) -> list[dict]:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT chunk_id, faq_id, question, answer, text, text_for_retrieval,
                   embedding, filename
            FROM chunks WHERE document_id = ? ORDER BY id
            """,
            (document_id,),
        ).fetchall()
    chunks = []
    for row in rows:
        chunk = {
            "chunk_id": row["chunk_id"],
            "filename": row["filename"],
            "text": row["text"],
            "text_for_retrieval": row["text_for_retrieval"],
            "embedding": json.loads(row["embedding"]),
        }
        if row["faq_id"] is not None:
            chunk.update(
                faq_id=row["faq_id"], question=row["question"], answer=row["answer"]
            )
        chunks.append(chunk)
    return chunks


def evaluate(database: Path, document_id: int, cases_path: Path) -> dict:
    os.environ["DATABASE_PATH"] = str(database)
    from admissions_rag_assistant.retrieval_reranker import retrieve_relevant_chunks
    from admissions_rag_assistant.retrieval_reranker import infer_query_categories

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    chunks = _load_chunks_read_only(database, document_id)
    results = []
    latencies = []
    recall_1 = recall_3 = recall_5 = accepted = 0
    selected_relevant = selected_total = compatible_cases = 0
    unrelated_amount_leaks = unrelated_guarantee_leaks = 0
    forbidden_leaks = compatible_selected = 0
    unsupported_total = unsupported_empty = 0
    top1_category_hits = top1_category_total = query_category_hits = 0
    failures = 0
    for case in cases:
        started = time.perf_counter()
        intent = case.get("intent", "unknown")
        risk_level = case.get("risk_level", "high")
        query_categories = infer_query_categories(case["question"], intent)
        retrieval = retrieve_relevant_chunks(
            case["question"],
            chunks,
            intent=intent,
            risk_level=risk_level,
        )
        retrieved = retrieval.chunks
        latencies.append((time.perf_counter() - started) * 1000)
        retrieved_ids = [
            chunk["faq_id"] for chunk in retrieved if chunk.get("faq_id") is not None
        ]
        expected = case.get("expected_faq_ids", [])
        hit_1 = bool(set(expected) & set(retrieved_ids[:1]))
        hit_3 = bool(set(expected) & set(retrieved_ids[:3]))
        hit_5 = bool(set(expected) & set(retrieved_ids[:5]))
        recall_1 += hit_1
        recall_3 += hit_3
        recall_5 += hit_5
        accepted += bool(retrieved)
        selected_relevant += len(set(expected) & set(retrieved_ids))
        selected_total += len(retrieved_ids)
        forbidden = set(case.get("forbidden_faq_ids", []))
        compatible = not bool(forbidden & set(retrieved_ids))
        compatible_cases += compatible
        forbidden_leaks += not compatible
        allowed = set(case.get("allowed_faq_ids", expected))
        compatible_selected += len(allowed & set(retrieved_ids))
        allow_empty = bool(case.get("allow_empty", False))
        if allow_empty and not allowed:
            unsupported_total += 1
            unsupported_empty += not retrieved_ids
        expected_categories = set(case.get("expected_query_categories", []))
        if expected_categories:
            query_category_hits += query_categories == expected_categories
        if retrieved and expected_categories:
            top1_category_total += 1
            top1_category_hits += bool(
                set(retrieved[0].get("primary_categories", [])) & expected_categories
            )
        amount_leak = bool(
            set(case.get("unrelated_amount_faq_ids", [])) & set(retrieved_ids)
        )
        guarantee_leak = bool(
            set(case.get("unrelated_guarantee_faq_ids", [])) & set(retrieved_ids)
        )
        unrelated_amount_leaks += amount_leak
        unrelated_guarantee_leaks += guarantee_leak
        required = case.get("required", True)
        if allowed:
            passed = (
                bool(allowed & set(retrieved_ids)) or (allow_empty and not retrieved_ids)
            ) and compatible
        elif allow_empty:
            passed = not retrieved_ids
        else:
            passed = not required or hit_3
        if required and not passed:
            failures += 1
        results.append({
            "question": case["question"],
            "expected_faq_ids": expected,
            "retrieved_faq_ids": retrieved_ids,
            "query_categories": sorted(query_categories),
            "category_compatible": compatible,
            "unrelated_amount_leak": amount_leak,
            "unrelated_guarantee_leak": guarantee_leak,
            "passed": passed,
        })
    total = len(cases)
    return {
        "total_cases": total,
        "recall_at_1": recall_1 / total if total else 0.0,
        "recall_at_3": recall_3 / total if total else 0.0,
        "recall_at_5": recall_5 / total if total else 0.0,
        "selected_context_precision": selected_relevant / selected_total if selected_total else 0.0,
        "category_compatibility": compatible_cases / total if total else 0.0,
        "unrelated_amount_leakage": unrelated_amount_leaks / total if total else 0.0,
        "unrelated_guarantee_leakage": unrelated_guarantee_leaks / total if total else 0.0,
        "forbidden_leakage_rate": forbidden_leaks / total if total else 0.0,
        "compatible_context_precision": compatible_selected / selected_total if selected_total else 1.0,
        "empty_when_unsupported_accuracy": unsupported_empty / unsupported_total if unsupported_total else 1.0,
        "top_1_category_accuracy": top1_category_hits / top1_category_total if top1_category_total else 1.0,
        "query_category_accuracy": query_category_hits / total if total else 0.0,
        "accepted_context_rate": accepted / total if total else 0.0,
        "average_retrieval_latency_ms": sum(latencies) / total if total else 0.0,
        "cases": results,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    report = evaluate(args.database, args.document_id, args.cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
