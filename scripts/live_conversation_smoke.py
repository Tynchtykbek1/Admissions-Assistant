"""Local real-provider smoke on an isolated SQLite copy."""
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path


QUESTIONS = [
    "Какие документы нужны для магистратуры?",
    "А какой срок подачи?",
]
MAX_PROVIDER_REQUESTS = len(QUESTIONS) * 2


def resolve_system_document(database_module, settings_module) -> int:
    if settings_module.SYSTEM_DOCUMENT_ID_INVALID:
        raise RuntimeError("SYSTEM_DOCUMENT_ID must be a positive integer.")
    document_id = settings_module.SYSTEM_DOCUMENT_ID
    if document_id is None:
        raise RuntimeError("SYSTEM_DOCUMENT_ID is not configured.")
    if database_module.get_document(document_id) is None:
        raise RuntimeError(f"Configured SYSTEM_DOCUMENT_ID={document_id} does not exist.")
    if database_module.count_document_chunks(document_id) <= 0:
        raise RuntimeError(f"Configured SYSTEM_DOCUMENT_ID={document_id} has no chunks.")
    return document_id


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        print("Refusing to run: set RUN_LIVE_LLM_TESTS=1 explicitly.")
        return 2
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from dotenv import load_dotenv
    load_dotenv(project / ".env")
    source = Path(os.getenv("DATABASE_PATH", project / "admissions.db"))
    if not source.is_absolute():
        source = project / source
    with tempfile.TemporaryDirectory(prefix="admissions-smoke-", ignore_cleanup_errors=True) as directory:
        temporary = Path(directory) / "smoke.db"
        shutil.copy2(source, temporary)
        os.environ["DATABASE_PATH"] = str(temporary)
        import app_settings
        import database
        import rag_service
        try:
            document_id = resolve_system_document(database, app_settings)
        except RuntimeError as error:
            print(f"Refusing to run: {error}")
            return 2
        conversation_id = None
        chat_id = f"local-smoke-{uuid.uuid4().hex}"
        tool_calls = 0
        provider_failures = 0
        for question in QUESTIONS:
            started = time.perf_counter()
            response = rag_service.answer_conversation_question(
                question=question, conversation_id=conversation_id,
                external_chat_id=chat_id, external_user_id="local-smoke",
                document_id=document_id,
            )
            conversation_id = response["conversation_id"]
            tool_calls += int(response["tool_called"])
            provider_failures += int(response["status"] == "provider_unavailable")
            print(json.dumps({
                "question": question,
                "tool_called": response["tool_called"],
                "retrieval_result_count": response["retrieval_result_count"],
                "verified_context_used": response["verified_context_used"],
                "answer": response["answer"],
                "latency": round((time.perf_counter() - started) * 1000, 2),
            }, ensure_ascii=False))
        print(json.dumps({
            "messages": len(QUESTIONS),
            "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
            "tool_calls": tool_calls,
            "provider_failures": provider_failures,
        }, ensure_ascii=False))
    return 1 if provider_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
