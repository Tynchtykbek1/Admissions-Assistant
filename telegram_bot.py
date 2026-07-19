import logging
import os

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
ERROR_MESSAGE = "The assistant is temporarily unavailable. Please try again later."
MAX_MESSAGE_LENGTH = 4000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def ask_backend(question: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BACKEND_URL}/ask-llm",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()


def format_backend_response(result: dict) -> str:
    if not isinstance(result, dict):
        raise ValueError("The backend returned an invalid response.")

    answer = result.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The backend returned an empty answer.")

    filenames = []

    for source in result.get("sources", []):
        filename = source.get("filename")

        if filename and filename not in filenames:
            filenames.append(filename)

        if len(filenames) == 3:
            break

    if not filenames:
        return answer.strip()

    sources_text = "\n".join(f"- {filename}" for filename in filenames)
    return f"{answer.strip()}\n\nSources:\n{sources_text}"


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    parts = []
    remaining = text.strip()

    while len(remaining) > max_length:
        split_at = remaining.rfind(" ", 0, max_length + 1)

        if split_at == -1:
            split_at = max_length

        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)

    return parts


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Hello! Send me a question about admissions, scholarships, visas, "
            "or university documents."
        )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    thinking_message = await update.message.reply_text("Thinking...")

    try:
        result = await ask_backend(update.message.text)
        final_text = format_backend_response(result)
        message_parts = split_message(final_text)

        await thinking_message.edit_text(message_parts[0])

        for part in message_parts[1:]:
            await update.message.reply_text(part)
    except (httpx.HTTPError, ValueError, TypeError):
        logger.exception("Could not get an answer from the FastAPI backend.")
        await thinking_message.edit_text(ERROR_MESSAGE)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Add it to your .env file.")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question)
    )

    print("Telegram bot is running with long polling...")
    application.run_polling()


if __name__ == "__main__":
    main()
