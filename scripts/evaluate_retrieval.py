import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    from embedding_retriever import find_relevant_chunks_semantic
    from retrieval_settings import (
        CONTEXT_SCORE_MARGIN,
        SEMANTIC_FALLBACK_SCORE_THRESHOLD,
        SEMANTIC_SCORE_THRESHOLD,
        SEMANTIC_TOP_K,
    )

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    chunks = _load_chunks_read_only(database, document_id)
    results = []
    latencies = []
    recall_1 = recall_3 = accepted = 0
    failures = 0
    for case in cases:
        started = time.perf_counter()
        retrieved = find_relevant_chunks_semantic(
            case["question"],
            chunks,
            top_k=SEMANTIC_TOP_K,
            min_score=SEMANTIC_SCORE_THRESHOLD,
            fallback_score_threshold=SEMANTIC_FALLBACK_SCORE_THRESHOLD,
            context_score_margin=CONTEXT_SCORE_MARGIN,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        retrieved_ids = [
            chunk["faq_id"] for chunk in retrieved if chunk.get("faq_id") is not None
        ]
        expected = case.get("expected_faq_ids", [])
        hit_1 = bool(set(expected) & set(retrieved_ids[:1]))
        hit_3 = bool(set(expected) & set(retrieved_ids[:3]))
        recall_1 += hit_1
        recall_3 += hit_3
        accepted += bool(retrieved)
        required = case.get("required", True)
        if required and not hit_3:
            failures += 1
        results.append({
            "question": case["question"],
            "expected_faq_ids": expected,
            "retrieved_faq_ids": retrieved_ids,
            "passed": not required or hit_3,
        })
    total = len(cases)
    return {
        "total_cases": total,
        "recall_at_1": recall_1 / total if total else 0.0,
        "recall_at_3": recall_3 / total if total else 0.0,
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
