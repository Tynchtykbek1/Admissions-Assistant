import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telegram.constants import ChatAction, ParseMode

from telegram_bot import (
    ERROR_MESSAGES,
    HELP_MESSAGES,
    NO_INFORMATION_MESSAGES,
    START_MESSAGES,
    format_backend_response,
    handle_question,
    help_command,
    split_message,
    start_command,
)


def make_update(text="question", language_code="ru"):
    message = SimpleNamespace(text=text, reply_text=AsyncMock())
    chat = SimpleNamespace(send_action=AsyncMock())
    user = SimpleNamespace(language_code=language_code)
    return SimpleNamespace(message=message, effective_chat=chat, effective_user=user)


class TelegramFormattingTests(unittest.TestCase):
    def test_formats_unique_source_filenames(self):
        result = {
            "answer": "The deadline is 30 April.",
            "sources": [
                {"filename": "admissions.pdf"},
                {"filename": "admissions.pdf"},
                {"filename": "faq.txt"},
            ],
        }
        formatted = format_backend_response(result)
        self.assertEqual(formatted.count("admissions.pdf"), 1)
        self.assertIn("faq.txt", formatted)

    def test_removes_raw_markdown_and_escapes_html(self):
        formatted = format_backend_response({"answer": "**Документы** <важно>", "sources": []}, "ru")
        self.assertNotIn("**", formatted)
        self.assertIn("Документы", formatted)
        self.assertIn("&lt;важно&gt;", formatted)

    def test_splits_long_messages_below_limit_without_losing_words(self):
        text = "\n\n".join(["Предложение с важными данными." * 10] * 8)
        parts = split_message(text, max_length=120)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(0 < len(part) <= 120 for part in parts))
        self.assertEqual("".join("".join(parts).split()), "".join("".join(text.split()).split()))

    def test_localizes_no_information(self):
        result = {"answer": "There is not enough information in the uploaded document.", "sources": []}
        self.assertEqual(format_backend_response(result, "ru"), NO_INFORMATION_MESSAGES["ru"])
        self.assertEqual(format_backend_response(result, "en"), NO_INFORMATION_MESSAGES["en"])


class TelegramHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def assert_local_greeting(self, text, expected_language):
        update = make_update(text)
        with patch("telegram_bot.ask_backend", new=AsyncMock()) as backend:
            await handle_question(update, SimpleNamespace())
        backend.assert_not_awaited()
        update.effective_chat.send_action.assert_not_awaited()
        sent = update.message.reply_text.await_args.args[0]
        self.assertIn("Здравствуйте" if expected_language == "ru" else "Hello", sent)

    async def test_russian_greeting_is_local(self):
        await self.assert_local_greeting("Привет!", "ru")

    async def test_english_greeting_is_local(self):
        await self.assert_local_greeting("Hello", "en")

    async def test_greeting_with_question_calls_backend(self):
        update = make_update("Привет, какие документы нужны для поступления?")
        backend_result = {"answer": "Нужен паспорт.", "sources": []}
        with patch("telegram_bot.ask_backend", new=AsyncMock(return_value=backend_result)) as backend:
            await handle_question(update, SimpleNamespace())
        backend.assert_awaited_once_with(update.message.text)
        update.effective_chat.send_action.assert_awaited_once_with(action=ChatAction.TYPING)

    async def test_russian_and_english_backend_fallbacks(self):
        backend_result = {"answer": "There is not enough information in the uploaded document.", "sources": []}
        for question, language in (("Какие требования?", "ru"), ("What are the requirements?", "en")):
            update = make_update(question)
            with patch("telegram_bot.ask_backend", new=AsyncMock(return_value=backend_result)):
                await handle_question(update, SimpleNamespace())
            self.assertEqual(update.message.reply_text.await_args.args[0], NO_INFORMATION_MESSAGES[language])

    async def test_localized_backend_error(self):
        for question, language in (("Какие требования?", "ru"), ("What requirements?", "en")):
            update = make_update(question)
            with patch("telegram_bot.ask_backend", new=AsyncMock(side_effect=ValueError("bad"))):
                await handle_question(update, SimpleNamespace())
            self.assertEqual(update.message.reply_text.await_args.args[0], ERROR_MESSAGES[language])

    async def test_start_and_help_in_both_languages(self):
        for code, language in (("ru", "ru"), ("en-US", "en")):
            start_update = make_update("/start", code)
            await start_command(start_update, SimpleNamespace())
            start_update.message.reply_text.assert_awaited_once_with(
                START_MESSAGES[language], parse_mode=ParseMode.HTML
            )

            help_update = make_update("/help", code)
            await help_command(help_update, SimpleNamespace())
            help_update.message.reply_text.assert_awaited_once_with(
                HELP_MESSAGES[language], parse_mode=ParseMode.HTML
            )


if __name__ == "__main__":
    unittest.main()
