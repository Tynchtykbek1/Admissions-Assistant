import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors as gemini_errors
from openai import OpenAI, OpenAIError

from answer_generator import generate_basic_answer


load_dotenv()

RAG_INSTRUCTIONS = (
    "Answer the user's question only using the provided context. "
    "If the context does not contain the answer, say there is not enough "
    "information in the uploaded document. Do not invent admissions requirements, "
    "deadlines, scholarships, visas, or university policies. Keep the answer clear "
    "and concise."
)


def build_context(relevant_chunks: list[dict]) -> str:
    context_parts = []

    for source_number, chunk in enumerate(relevant_chunks, start=1):
        context_parts.append(
            f"[Source {source_number}]\n"
            f"filename: {chunk['filename']}\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"text: {chunk['text']}"
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


def generate_llm_answer(question: str, relevant_chunks: list[dict]) -> str:
    if not relevant_chunks:
        return generate_basic_answer(question, relevant_chunks)

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    context = build_context(relevant_chunks)

    try:
        if provider == "openai":
            answer = generate_openai_answer(question, context)
        elif provider == "gemini":
            answer = generate_gemini_answer(question, context)
        else:
            answer = None
    except (OpenAIError, gemini_errors.APIError):
        return generate_basic_answer(question, relevant_chunks)

    if not answer:
        return generate_basic_answer(question, relevant_chunks)

    return answer
