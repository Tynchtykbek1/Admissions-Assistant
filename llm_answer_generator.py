import os
import logging
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from openai import OpenAI


load_dotenv()
logger = logging.getLogger(__name__)

SUCCESS = "success"
INSUFFICIENT_DOCUMENT_INFORMATION = "insufficient_document_information"
PROVIDER_UNAVAILABLE = "provider_unavailable"

INSUFFICIENT_INFORMATION_ANSWER = (
    "There is not enough information in the uploaded document to answer this question."
)
PROVIDER_UNAVAILABLE_ANSWER = (
    "The service is temporarily unavailable. Please try again in a few minutes."
)

RAG_INSTRUCTIONS = (
    "Answer only using the provided context. Never invent university names, admission "
    "requirements, deadlines, visa rules, documents, costs, scholarships, or legal "
    "information. If the context contains only part of the requested information, "
    "provide every relevant supported fact and clearly say what is missing. If the "
    "requested fact is absent, say directly that the uploaded document does not "
    "contain it. Do not fill the response with adjacent unrelated admissions "
    "information. Keep the response concise and Telegram-friendly. Use short bullet "
    "points where useful. Do not add a Sources section."
)


@dataclass(frozen=True)
class LLMAnswerResult:
    status: str
    answer: str
    provider: str
    provider_duration_ms: float
    error_category: str | None = None
    provider_status: int | None = None
    provider_request_id: str | None = None


def _safe_provider_name(provider: str) -> str:
    if provider in {"openai", "gemini"}:
        return provider
    return "unconfigured" if not provider else "invalid"


def build_context(relevant_chunks: list[dict]) -> str:
    context_parts = []

    for source_number, chunk in enumerate(relevant_chunks, start=1):
        identifier = (
            f"FAQ ID: {chunk['faq_id']}"
            if "faq_id" in chunk
            else f"Chunk ID: {chunk['chunk_id']}"
        )
        context_parts.append(
            f"Source {source_number}:\n"
            f"Filename: {chunk['filename']}\n"
            f"{identifier}\n"
            f"Content:\n{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def generate_openai_answer(question: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")

    if not api_key or not model_name:
        return None

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        instructions=RAG_INSTRUCTIONS,
        input=f"Question:\n{question}\n\nContext:\n{context}"
    )

    return response.output_text


def generate_gemini_answer(question: str, context: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key or not model_name:
        return None

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=(
                f"{RAG_INSTRUCTIONS}\n\n"
                f"Question:\n{question}\n\nContext:\n{context}"
            )
        )
        return response.text
    finally:
        client.close()


def _safe_provider_error_details(error: Exception) -> tuple[str, int | None, str | None]:
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        error_code = getattr(error, "code", None)
        status = error_code if isinstance(error_code, int) else None
    response = getattr(error, "response", None)
    if not isinstance(status, int) and response is not None:
        response_status = getattr(response, "status_code", None)
        status = response_status if isinstance(response_status, int) else None

    request_id = getattr(error, "request_id", None)
    if not isinstance(request_id, str) and response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = headers.get("x-request-id") or headers.get("request-id")
    if (
        not isinstance(request_id, str)
        or len(request_id) > 128
        or re.fullmatch(r"[A-Za-z0-9._:-]+", request_id) is None
    ):
        request_id = None

    error_name = type(error).__name__.casefold()
    provider_status = getattr(error, "status", None)
    provider_status_name = (
        provider_status.casefold() if isinstance(provider_status, str) else ""
    )
    if status == 429 or "resource_exhausted" in provider_status_name:
        category = "rate_limited"
    elif status in (401, 403) or any(
        marker in error_name for marker in ("authentication", "permission", "unauthorized")
    ):
        category = "authentication"
    elif isinstance(error, TimeoutError) or "timeout" in error_name:
        category = "timeout"
    elif "connection" in error_name:
        category = "connection"
    elif status is not None:
        category = "provider_http_error"
    else:
        category = "unexpected_provider_error"
    return category, status, request_id


def _unavailable_result(
    provider: str,
    started_at: float,
    category: str,
    status: int | None = None,
    request_id: str | None = None,
) -> LLMAnswerResult:
    logger.warning(
        "provider_failure provider=%s category=%s status=%s request_id=%s",
        _safe_provider_name(provider),
        category,
        status if status is not None else "none",
        request_id or "none",
    )
    return LLMAnswerResult(
        status=PROVIDER_UNAVAILABLE,
        answer=PROVIDER_UNAVAILABLE_ANSWER,
        provider=_safe_provider_name(provider),
        provider_duration_ms=(time.perf_counter() - started_at) * 1000,
        error_category=category,
        provider_status=status,
        provider_request_id=request_id,
    )


def generate_llm_answer(question: str, relevant_chunks: list[dict]) -> LLMAnswerResult:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    started_at = time.perf_counter()

    if not relevant_chunks:
        return LLMAnswerResult(
            status=INSUFFICIENT_DOCUMENT_INFORMATION,
            answer=INSUFFICIENT_INFORMATION_ANSWER,
            provider=_safe_provider_name(provider),
            provider_duration_ms=0.0,
        )

    if provider not in {"openai", "gemini"}:
        return _unavailable_result(provider, started_at, "invalid_configuration")
    required_configuration = (
        ("OPENAI_API_KEY", "OPENAI_MODEL")
        if provider == "openai"
        else ("GEMINI_API_KEY", "GEMINI_MODEL")
    )
    if any(not os.getenv(name) for name in required_configuration):
        return _unavailable_result(provider, started_at, "invalid_configuration")

    context = build_context(relevant_chunks)

    try:
        if provider == "openai":
            answer = generate_openai_answer(question, context)
        else:
            answer = generate_gemini_answer(question, context)
    except Exception as error:
        category, status, request_id = _safe_provider_error_details(error)
        return _unavailable_result(provider, started_at, category, status, request_id)

    if not isinstance(answer, str) or not answer.strip():
        return _unavailable_result(provider, started_at, "malformed_response")

    answer = answer.strip()
    return LLMAnswerResult(
        status=SUCCESS,
        answer=answer,
        provider=provider,
        provider_duration_ms=(time.perf_counter() - started_at) * 1000,
    )
