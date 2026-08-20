import asyncio
import html
import logging
import os
import re
import time
from weakref import WeakValueDictionary

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from logging_config import configure_logging
from telegram_settings import BACKEND_TIMEOUT


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_MESSAGE_LENGTH = 4000
TYPING_INTERVAL_SECONDS = 4.0

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
        "Можно задавать уточняющие вопросы: бот учитывает недавний контекст диалога.\n"
        "Используйте /reset, чтобы очистить историю текущего диалога.\n\n"
        "Ответы основаны на загруженных документах о поступлении.\n\n"
        "Если нужного ответа нет, вы можете написать менеджерам:\n\n"
        "• Адахан — @TheLuckiestPersonEver\n"
        "• Максат — @maksatuniguide"
    ),
    "en": (
        "I can help with admissions, required documents, application deadlines, "
        "visas, and related requirements.\n\n"
        "Example questions:\n"
        "• What documents are required for admission?\n"
        "• What are the application deadlines?\n"
        "• Do my documents need an apostille?\n\n"
        "Follow-up questions are supported using recent conversation context.\n"
        "Use /reset to clear the current conversation history.\n\n"
        "Answers are based on the uploaded admissions documents.\n\n"
        "If the answer is not available, you can contact the managers:\n\n"
        "• Adakhan — @TheLuckiestPersonEver\n"
        "• Maksat — @maksatuniguide"
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

SYSTEM_DOCUMENT_UNAVAILABLE_MESSAGES = {
    "ru": "База знаний временно недоступна. Пожалуйста, попробуйте позже.",
    "en": "The knowledge base is temporarily unavailable. Please try again later.",
}

RESET_MESSAGES = {
    "ru": "История диалога очищена. Активный документ сохранён.",
    "en": "Conversation history cleared. The active document was kept.",
}

STATUS_NO_DOCUMENT_MESSAGES = {
    "ru": "Backend доступен. Для этого диалога активный документ не выбран.",
    "en": "The backend is reachable. No active document is selected for this conversation.",
}

configure_logging()
logger = logging.getLogger(__name__)
_chat_locks: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()


def _telegram_identifiers(update: Update) -> tuple[str, str | None]:
    if not update.effective_chat:
        raise ValueError("Telegram chat is unavailable.")
    chat_id = str(update.effective_chat.id)
    user_id = str(update.effective_user.id) if update.effective_user else None
    return chat_id, user_id


def _chat_lock(update: Update) -> asyncio.Lock:
    if not update.effective_chat:
        return asyncio.Lock()
    chat_id = int(update.effective_chat.id)
    lock = _chat_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[chat_id] = lock
    return lock


async def ask_backend(
    question: str,
    external_chat_id: str,
    external_user_id: str | None,
    conversation_id: str | None = None,
) -> dict:
    payload = {
        "question": question,
        "external_chat_id": external_chat_id,
        "external_user_id": external_user_id,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
        response = await client.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 503:
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("The backend returned an invalid response.")
            if result.get("status") in {
                "provider_unavailable",
                "system_document_unavailable",
            }:
                return result
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("The backend returned an invalid response.")
        return result


async def reset_backend(
    external_chat_id: str,
    external_user_id: str | None,
    conversation_id: str | None = None,
) -> dict:
    payload = {
        "external_chat_id": external_chat_id,
        "external_user_id": external_user_id,
        "conversation_id": conversation_id,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(f"{BACKEND_URL}/conversation/reset", json=payload)
        response.raise_for_status()
        return response.json()


async def backend_status(
    external_chat_id: str,
    external_user_id: str | None,
    conversation_id: str | None = None,
) -> dict:
    params = {
        "external_chat_id": external_chat_id,
        "external_user_id": external_user_id,
    }
    if conversation_id:
        params["conversation_id"] = conversation_id
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{BACKEND_URL}/conversation/status", params=params
        )
        if response.status_code == 503:
            result = response.json()
            if (
                isinstance(result, dict)
                and result.get("status") == "system_document_unavailable"
            ):
                return result
        response.raise_for_status()
        return response.json()


def language_from_code(language_code: str | None) -> str:
    return "en" if (language_code or "").lower().startswith("en") else "ru"


def detect_text_language(text: str) -> str:
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "ru"


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
    if status == "system_document_unavailable":
        return SYSTEM_DOCUMENT_UNAVAILABLE_MESSAGES[language]
    if status == "insufficient_document_information":
        return NO_INFORMATION_MESSAGES[language]
    if status not in {None, "success", "partial_information"}:
        raise ValueError("The backend returned an invalid status.")

    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The backend returned an empty answer.")

    return sanitize_for_html(answer)


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


def _stored_conversation_id(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    chat_data = getattr(context, "chat_data", None)
    return chat_data.get("conversation_id") if isinstance(chat_data, dict) else None


def _remember_conversation_id(
    context: ContextTypes.DEFAULT_TYPE, result: dict
) -> None:
    conversation_id = result.get("conversation_id")
    chat_data = getattr(context, "chat_data", None)
    if isinstance(chat_data, dict) and isinstance(conversation_id, str):
        chat_data["conversation_id"] = conversation_id


async def reset_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not update.message:
        return
    language = user_command_language(update)
    try:
        chat_id, user_id = _telegram_identifiers(update)
        async with _chat_lock(update):
            result = await reset_backend(
                chat_id, user_id, _stored_conversation_id(context)
            )
        _remember_conversation_id(context, result)
        if result.get("status") == "system_document_unavailable":
            await send_html(
                update.message,
                SYSTEM_DOCUMENT_UNAVAILABLE_MESSAGES[language],
            )
            return
        await send_html(update.message, RESET_MESSAGES[language])
    except (httpx.HTTPError, ValueError, TypeError):
        await send_html(update.message, ERROR_MESSAGES[language])


async def status_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not update.message:
        return
    language = user_command_language(update)
    try:
        chat_id, user_id = _telegram_identifiers(update)
        result = await backend_status(
            chat_id, user_id, _stored_conversation_id(context)
        )
        _remember_conversation_id(context, result)
        if result.get("status") == "system_document_unavailable":
            await send_html(
                update.message,
                SYSTEM_DOCUMENT_UNAVAILABLE_MESSAGES[language],
            )
            return
        filename = result.get("active_document_filename")
        if filename:
            prefix = "Backend доступен. Активный документ" if language == "ru" else (
                "The backend is reachable. Active document"
            )
            text = f"{prefix}: {html.escape(str(filename), quote=False)}"
        else:
            text = STATUS_NO_DOCUMENT_MESSAGES[language]
        await send_html(update.message, text)
    except (httpx.HTTPError, ValueError, TypeError):
        await send_html(update.message, ERROR_MESSAGES[language])


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
    language = detect_text_language(question)
    async with _chat_lock(update):
        backend_started_at = time.perf_counter()
        typing_task = (
            asyncio.create_task(keep_typing(update.effective_chat))
            if update.effective_chat
            else None
        )
        try:
            try:
                if typing_task is not None:
                    await asyncio.sleep(0)
                chat_id, user_id = _telegram_identifiers(update)
                result = await ask_backend(
                    question,
                    chat_id,
                    user_id,
                    _stored_conversation_id(context),
                )
                _remember_conversation_id(context, result)
            finally:
                await stop_typing(typing_task)
            final_text = format_backend_response(result, language)
            for part in split_message(final_text):
                await send_html(update.message, part)
        except Exception as error:
            if isinstance(error, httpx.ReadTimeout):
                error_category = "read_timeout"
            elif isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
                error_category = "connection_failure"
            elif isinstance(error, httpx.HTTPStatusError):
                error_category = "http_status_failure"
            elif isinstance(error, (ValueError, TypeError)):
                error_category = "invalid_backend_response"
            else:
                error_category = "unexpected_failure"
            logger.warning(
                "Backend request failed endpoint=chat category=%s elapsed_ms=%d.",
                error_category,
                round((time.perf_counter() - backend_started_at) * 1000),
            )
            await send_html(update.message, ERROR_MESSAGES[language])


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Add it to your .env file.")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Telegram bot is running with long polling.")
    application.run_polling()


if __name__ == "__main__":
    main()
