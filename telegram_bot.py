import asyncio
import html
import logging
import os
import re

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from logging_config import configure_logging


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_MESSAGE_LENGTH = 4000
TYPING_INTERVAL_SECONDS = 4.0

GREETINGS = {
    "ru": {
        "привет", "здравствуйте", "добрый день", "доброе утро",
        "добрый вечер", "салам",
    },
    "en": {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    },
}

GREETING_MESSAGES = {
    "ru": (
        "Здравствуйте! Я помогу найти информацию о поступлении, документах, "
        "сроках подачи, визе и других вопросах. Просто напишите свой вопрос."
    ),
    "en": (
        "Hello! I can help you find information about admissions, documents, "
        "application deadlines, visas, and related topics. Send me your question."
    ),
}

FAREWELLS = {
    "ru": {"пока", "до свидания", "спасибо", "спасибо пока"},
    "en": {"bye", "goodbye", "thanks", "thank you"},
}

FAREWELL_MESSAGES = {
    "ru": "До свидания! Обращайтесь, если появятся вопросы о поступлении.",
    "en": "Goodbye! Feel free to return with any admissions questions.",
}

START_MESSAGES = {
    "ru": (
        "Здравствуйте! Я отвечаю на вопросы по загруженным документам о поступлении.\n\n"
        "Например:\n"
        "• Какие документы нужны?\n"
        "• Когда заканчивается подача заявок?\n"
        "• Нужна ли студенческая виза?\n\n"
        "Подробнее о возможностях — /help"
    ),
    "en": (
        "Hello! I answer admissions questions using the uploaded documents.\n\n"
        "For example:\n"
        "• What documents are required?\n"
        "• When is the application deadline?\n"
        "• Do I need a student visa?\n\n"
        "For more information, use /help"
    ),
}

HELP_MESSAGES = {
    "ru": (
        "Я могу помочь с вопросами о поступлении, необходимых документах, сроках "
        "подачи, визах и других требованиях.\n\n"
        "Примеры вопросов:\n"
        "• Какие документы нужны для поступления?\n"
        "• Каковы сроки подачи заявления?\n"
        "• Нужно ли апостилировать документы?\n\n"
        "Ответы основаны на загруженных документах о поступлении. Если информации "
        "недостаточно, пожалуйста, обратитесь к менеджеру."
    ),
    "en": (
        "I can help with admissions, required documents, application deadlines, "
        "visas, and related requirements.\n\n"
        "Example questions:\n"
        "• What documents are required for admission?\n"
        "• What are the application deadlines?\n"
        "• Do my documents need an apostille?\n\n"
        "Answers are based on the uploaded admissions documents. If there is not "
        "enough information, please contact a human manager."
    ),
}

NO_INFORMATION_MESSAGES = {
    "ru": (
        "В загруженных документах недостаточно информации для точного ответа. "
        "Попробуйте переформулировать вопрос или обратитесь к менеджеру."
    ),
    "en": (
        "The uploaded documents do not contain enough information for an accurate "
        "answer. Try rephrasing your question or contact a human manager."
    ),
}

ERROR_MESSAGES = {
    "ru": "Сейчас не удалось получить ответ. Пожалуйста, попробуйте ещё раз немного позже.",
    "en": "I couldn't get an answer right now. Please try again a little later.",
}

PROVIDER_UNAVAILABLE_MESSAGES = {
    "ru": "Сервис временно перегружен. Попробуйте повторить вопрос через несколько минут.",
    "en": "The service is temporarily unavailable. Please try again in a few minutes.",
}

configure_logging()
logger = logging.getLogger(__name__)


async def ask_backend(question: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{BACKEND_URL}/ask-llm", json={"question": question})
        if response.status_code == 503:
            result = response.json()
            if isinstance(result, dict) and result.get("status") == "provider_unavailable":
                return result
        response.raise_for_status()
        return response.json()


def language_from_code(language_code: str | None) -> str:
    return "en" if (language_code or "").lower().startswith("en") else "ru"


def detect_text_language(text: str) -> str:
    cyrillic_count = len(re.findall(r"[А-Яа-яЁё]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyrillic_count >= latin_count else "en"


def normalize_greeting(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", text.casefold(), flags=re.UNICODE)
    return " ".join(normalized.split())


def greeting_language(text: str) -> str | None:
    normalized = normalize_greeting(text)
    if not normalized or len(normalized.split()) > 3:
        return None
    for language, greetings in GREETINGS.items():
        if normalized in greetings:
            return language
    return None


def farewell_language(text: str) -> str | None:
    normalized = normalize_greeting(text)
    if not normalized or len(normalized.split()) > 3:
        return None
    for language, farewells in FAREWELLS.items():
        if normalized in farewells:
            return language
    return None


def sanitize_for_html(text: str) -> str:
    """Remove common Markdown decoration, then escape all backend-controlled text."""
    text = re.sub(r"(?m)^([ \t]*)[-+*](?=[ \t]+)", r"\1•", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    return html.escape(text.strip(), quote=False)


def format_backend_response(result: dict, language: str = "en") -> str:
    if not isinstance(result, dict):
        raise ValueError("The backend returned an invalid response.")
    status = result.get("status")
    if status == "provider_unavailable":
        return PROVIDER_UNAVAILABLE_MESSAGES[language]
    if status == "insufficient_document_information":
        return NO_INFORMATION_MESSAGES[language]
    if status not in {None, "success"}:
        raise ValueError("The backend returned an invalid status.")

    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The backend returned an empty answer.")

    filenames = []
    for source in result.get("sources", []):
        if not isinstance(source, dict):
            continue
        filename = source.get("filename")
        if filename and filename not in filenames:
            filenames.append(str(filename))
        if len(filenames) == 3:
            break

    safe_answer = sanitize_for_html(answer)
    if not filenames:
        return safe_answer

    heading = "Источники" if language == "ru" else "Sources"
    sources_text = "\n".join(f"• {html.escape(name, quote=False)}" for name in filenames)
    return f"{safe_answer}\n\n{heading}:\n{sources_text}"


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split at paragraphs, then sentence/whitespace boundaries; never exceed the limit."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    parts = []
    remaining = text.strip()

    while len(remaining) > max_length:
        window = remaining[: max_length + 1]
        candidates = [
            window.rfind("\n\n"),
            max(window.rfind(". "), window.rfind("! "), window.rfind("? ")),
            window.rfind("\n"),
            window.rfind(" "),
        ]
        split_at = next((position for position in candidates if position > 0), max_length)
        if window[split_at:split_at + 2] in {". ", "! ", "? "}:
            split_at += 1
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def user_command_language(update: Update) -> str:
    user = update.effective_user
    return language_from_code(user.language_code if user else None)


async def send_html(message, text: str) -> None:
    await message.reply_text(text, parse_mode=ParseMode.HTML)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await send_html(update.message, START_MESSAGES[user_command_language(update)])


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await send_html(update.message, HELP_MESSAGES[user_command_language(update)])


async def keep_typing(chat) -> None:
    try:
        while True:
            try:
                await chat.send_action(action=ChatAction.TYPING)
            except Exception:
                logger.warning("Telegram typing action failed.")
                return
            await asyncio.sleep(TYPING_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        raise


async def stop_typing(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()
    greeting = greeting_language(question)
    if greeting:
        await send_html(update.message, GREETING_MESSAGES[greeting])
        return
    farewell = farewell_language(question)
    if farewell:
        await send_html(update.message, FAREWELL_MESSAGES[farewell])
        return

    language = detect_text_language(question)
    typing_task = (
        asyncio.create_task(keep_typing(update.effective_chat))
        if update.effective_chat
        else None
    )

    try:
        try:
            if typing_task is not None:
                # Give the background task one event-loop turn for the first action.
                await asyncio.sleep(0)
            result = await ask_backend(question)
        finally:
            await stop_typing(typing_task)
        final_text = format_backend_response(result, language)
        for part in split_message(final_text):
            await send_html(update.message, part)
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger.warning(
            "Could not get an answer from the FastAPI backend category=%s.",
            type(error).__name__,
        )
        await send_html(update.message, ERROR_MESSAGES[language])


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Add it to your .env file.")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Telegram bot is running with long polling.")
    application.run_polling()


if __name__ == "__main__":
    main()
