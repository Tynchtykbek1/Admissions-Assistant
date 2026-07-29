import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator


load_dotenv()
logger = logging.getLogger(__name__)

SUCCESS = "success"
PARTIAL_INFORMATION = "partial_information"
INSUFFICIENT_DOCUMENT_INFORMATION = "insufficient_document_information"
PROVIDER_UNAVAILABLE = "provider_unavailable"

INSUFFICIENT_INFORMATION_ANSWER = (
    "There is not enough information in the uploaded document to answer this question."
)
PROVIDER_UNAVAILABLE_ANSWER = (
    "The service is temporarily unavailable. Please try again in a few minutes."
)

RAG_INSTRUCTIONS = """
Answer only from the retrieved document context.
The final standalone question already resolves any conversational references.
Use no previous dialogue as a factual source. Never invent university names,
admission requirements, deadlines, visa rules, documents, costs, scholarships,
contacts, procedures, or legal information.

Return exactly one JSON object:
{"status":"success|partial_information|insufficient_document_information",
 "answer":"concise user-facing answer"}

Use success only when the requested central fact is directly supported by the
retrieved context. Use partial_information only when the context directly supports
part, but not all, of the requested answer; provide every supported relevant fact and
clearly state what is missing. Use insufficient_document_information when no
retrieved fact directly answers the central request. Facts from the same general
admissions topic are not enough. Do not append adjacent unrelated information.
Answer in the same language as the final standalone question. Determine the answer
language only from that question, never from retrieved context, filenames,
conversation history, or previous assistant messages. A mainly English question
requires an English answer even when the context is Russian. A mainly Russian
question requires a Russian answer even when the context is English. Preserve
proper names, Telegram usernames, document names, and necessary official terms
unchanged. Do not translate, alter, or add facts. Keep the response concise and
Telegram-friendly, with short bullets where useful. Do not add a Sources section.
Do not include Markdown fences or text outside the JSON object.
""".strip()


class ProviderAnswer(BaseModel):
    status: Literal[
        "success",
        "partial_information",
        "insufficient_document_information",
    ]
    answer: str

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer must not be blank")
        return stripped


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


def _provider_configuration(
    provider: str,
    model_override: str | None = None,
) -> tuple[str, str] | None:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = model_override or os.getenv("OPENAI_MODEL", "").strip()
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model = model_override or os.getenv("GEMINI_MODEL", "").strip()
    else:
        return None
    return (api_key, model) if api_key and model else None


def generate_openai_text(
    instructions: str,
    input_text: str,
    *,
    model_override: str | None = None,
) -> str | None:
    configuration = _provider_configuration("openai", model_override)
    if configuration is None:
        return None
    api_key, model = configuration
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
    )
    return response.output_text


def generate_gemini_text(
    instructions: str,
    input_text: str,
    *,
    model_override: str | None = None,
) -> str | None:
    configuration = _provider_configuration("gemini", model_override)
    if configuration is None:
        return None
    api_key, model = configuration
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"{instructions}\n\n{input_text}",
        )
        return response.text
    finally:
        client.close()


def generate_provider_text(
    provider: str,
    instructions: str,
    input_text: str,
    *,
    model_override: str | None = None,
) -> str | None:
    if provider == "openai":
        return generate_openai_text(
            instructions, input_text, model_override=model_override
        )
    if provider == "gemini":
        return generate_gemini_text(
            instructions, input_text, model_override=model_override
        )
    return None


def build_context(relevant_chunks: list[dict]) -> str:
    context_parts = []
    for source_number, chunk in enumerate(relevant_chunks, start=1):
        if "faq_id" in chunk and (chunk.get("question") or chunk.get("answer")):
            fields = [
                f"Source {source_number}:",
                f"Filename: {chunk['filename']}",
                f"FAQ ID: {chunk['faq_id']}",
            ]
            if chunk.get("question"):
                fields.extend(("FAQ question:", chunk["question"]))
            if chunk.get("answer"):
                fields.extend(("FAQ answer:", chunk["answer"]))
            context_parts.append("\n".join(fields))
        else:
            context_parts.append(
                f"Source {source_number}:\n"
                f"Filename: {chunk['filename']}\n"
                f"Chunk ID: {chunk['chunk_id']}\n"
                f"Content:\n{chunk['text']}"
            )
    return "\n\n".join(context_parts)


def generate_openai_answer(
    question: str,
    context: str,
    *,
    standalone_question: str | None = None,
    history: list[dict] | None = None,
) -> str | None:
    return generate_openai_text(
        RAG_INSTRUCTIONS,
        _build_answer_input(question, standalone_question, history, context),
    )


def generate_gemini_answer(
    question: str,
    context: str,
    *,
    standalone_question: str | None = None,
    history: list[dict] | None = None,
) -> str | None:
    return generate_gemini_text(
        RAG_INSTRUCTIONS,
        _build_answer_input(question, standalone_question, history, context),
    )


def _build_answer_input(
    question: str,
    standalone_question: str | None,
    history: list[dict] | None,
    context: str,
) -> str:
    final_question = standalone_question or question
    return (
        f"Final standalone question:\n{final_question}\n\n"
        f"Required answer language:\n{_answer_language_instruction(final_question)}\n\n"
        f"Retrieved document context:\n{context}"
    )


def _answer_language_instruction(question: str) -> str:
    cyrillic_count = len(re.findall(r"[А-Яа-яЁё]", question))
    latin_count = len(re.findall(r"[A-Za-z]", question))
    if cyrillic_count and cyrillic_count >= latin_count:
        return "Russian. Answer in Russian."
    if latin_count > cyrillic_count:
        return "English. Answer in English."
    return "Use the same language as the final standalone question."


def parse_provider_answer(raw_response: str) -> ProviderAnswer | None:
    if not isinstance(raw_response, str) or not raw_response.strip():
        return None
    cleaned = raw_response.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
        if not isinstance(payload, dict) or set(payload) != {"status", "answer"}:
            return None
        return ProviderAnswer(**payload)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _safe_provider_error_details(
    error: Exception,
) -> tuple[str, int | None, str | None]:
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
        marker in error_name
        for marker in ("authentication", "permission", "unauthorized")
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


def generate_llm_answer(
    question: str,
    relevant_chunks: list[dict],
    *,
    standalone_question: str | None = None,
    history: list[dict] | None = None,
) -> LLMAnswerResult:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    started_at = time.perf_counter()

    if not relevant_chunks:
        return LLMAnswerResult(
            status=INSUFFICIENT_DOCUMENT_INFORMATION,
            answer=INSUFFICIENT_INFORMATION_ANSWER,
            provider=_safe_provider_name(provider),
            provider_duration_ms=0.0,
        )
    if provider not in {"openai", "gemini"} or _provider_configuration(provider) is None:
        return _unavailable_result(provider, started_at, "invalid_configuration")

    context = build_context(relevant_chunks)
    try:
        kwargs = {
            "standalone_question": standalone_question,
            "history": history,
        }
        if provider == "openai":
            raw_answer = generate_openai_answer(question, context, **kwargs)
        else:
            raw_answer = generate_gemini_answer(question, context, **kwargs)
    except Exception as error:
        category, status, request_id = _safe_provider_error_details(error)
        return _unavailable_result(provider, started_at, category, status, request_id)

    parsed = parse_provider_answer(raw_answer) if isinstance(raw_answer, str) else None
    if parsed is None:
        return _unavailable_result(provider, started_at, "malformed_response")
    return LLMAnswerResult(
        status=parsed.status,
        answer=parsed.answer,
        provider=provider,
        provider_duration_ms=(time.perf_counter() - started_at) * 1000,
    )
