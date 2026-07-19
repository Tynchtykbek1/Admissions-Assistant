import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from answer_generator import generate_basic_answer


load_dotenv()


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


def generate_llm_answer(question: str, relevant_chunks: list[dict]) -> str:
    if not relevant_chunks:
        return generate_basic_answer(question, relevant_chunks)

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")

    if not api_key or not model_name:
        return generate_basic_answer(question, relevant_chunks)

    context = build_context(relevant_chunks)
    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=model_name,
            instructions=(
                "Answer the user's question only using the provided context. "
                "Do not use outside knowledge or invent details. If the context "
                "does not contain the answer, say there is not enough information "
                "in the uploaded document."
            ),
            input=f"Question:\n{question}\n\nContext:\n{context}"
        )
    except OpenAIError:
        return generate_basic_answer(question, relevant_chunks)

    return response.output_text
