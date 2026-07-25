import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from telegram.constants import ChatAction, ParseMode

from telegram_bot import (
    ERROR_MESSAGES,
    FAREWELL_MESSAGES,
    HELP_MESSAGES,
    NO_INFORMATION_MESSAGES,
    PROVIDER_UNAVAILABLE_MESSAGES,
    START_MESSAGES,
    format_backend_response,
    handle_question,
    help_command,
    split_message,
    stop_typing,
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

    def test_converts_only_plain_line_asterisk_bullets(self):
        answer = "* passport\n  - admission letter\n+ income proof\n2 * 3 = 6\nasterisk*inside\n**Important**"
        formatted = format_backend_response({"answer": answer, "sources": []})
        self.assertIn("• passport\n  • admission letter", formatted)
        self.assertIn("• income proof", formatted)
        self.assertIn("2 * 3 = 6", formatted)
        self.assertIn("asterisk*inside", formatted)
        self.assertIn("Important", formatted)
        self.assertNotIn("**", formatted)

    def test_splits_long_messages_below_limit_without_losing_words(self):
        text = "\n\n".join(["Предложение с важными данными." * 10] * 8)
        parts = split_message(text, max_length=120)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(0 < len(part) <= 120 for part in parts))
        self.assertEqual("".join("".join(parts).split()), "".join("".join(text.split()).split()))

    def test_localizes_no_information(self):
        result = {
            "status": "insufficient_document_information",
            "answer": "There is not enough information in the uploaded document.",
            "sources": [],
        }
        self.assertEqual(format_backend_response(result, "ru"), NO_INFORMATION_MESSAGES["ru"])
        self.assertEqual(format_backend_response(result, "en"), NO_INFORMATION_MESSAGES["en"])

    def test_success_with_insufficient_phrase_remains_visible_with_sources(self):
        result = {
            "status": "success",
            "answer": (
                "Недостаточно информации для полного списка, но указаны:\n"
                "* гарантийное письмо\n* документы о доходах"
            ),
            "sources": [{"filename": "FAQ.docx.pdf"}],
        }
        formatted = format_backend_response(result, "ru")
        self.assertIn("Недостаточно информации для полного списка", formatted)
        self.assertIn("• гарантийное письмо", formatted)
        self.assertIn("FAQ.docx.pdf", formatted)
        self.assertNotEqual(formatted, NO_INFORMATION_MESSAGES["ru"])

    def test_localizes_provider_unavailable_exactly(self):
        result = {"status": "provider_unavailable", "answer": "ignored", "sources": []}
        self.assertEqual(
            format_backend_response(result, "ru"),
            "Сервис временно перегружен. Попробуйте повторить вопрос через несколько минут.",
        )
        self.assertEqual(
            format_backend_response(result, "en"),
            "The service is temporarily unavailable. Please try again in a few minutes.",
        )


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

    async def test_short_farewells_are_local(self):
        for text, language in (
            ("пока", "ru"),
            ("спасибо", "ru"),
            ("спасибо, пока", "ru"),
            ("bye", "en"),
            ("thank you", "en"),
        ):
            update = make_update(text)
            with patch("telegram_bot.ask_backend", new=AsyncMock()) as backend:
                await handle_question(update, SimpleNamespace())
            backend.assert_not_awaited()
            update.effective_chat.send_action.assert_not_awaited()
            self.assertEqual(
                update.message.reply_text.await_args.args[0],
                FAREWELL_MESSAGES[language],
            )

    async def test_longer_question_containing_thanks_calls_backend(self):
        update = make_update("Спасибо, какие документы нужны для визы?")
        backend_result = {
            "status": "success",
            "answer": "Упоминается пренролмент.",
            "sources": [],
        }
        with patch(
            "telegram_bot.ask_backend",
            new=AsyncMock(return_value=backend_result),
        ) as backend:
            await handle_question(update, SimpleNamespace())
        backend.assert_awaited_once_with(update.message.text)

    async def test_greeting_with_question_calls_backend(self):
        update = make_update("Привет, какие документы нужны для поступления?")
        backend_result = {"answer": "Нужен паспорт.", "sources": []}
        with patch("telegram_bot.ask_backend", new=AsyncMock(return_value=backend_result)) as backend:
            await handle_question(update, SimpleNamespace())
        backend.assert_awaited_once_with(update.message.text)
        update.effective_chat.send_action.assert_awaited_with(action=ChatAction.TYPING)

    async def test_russian_and_english_backend_fallbacks(self):
        backend_result = {
            "status": "insufficient_document_information",
            "answer": "There is not enough information in the uploaded document.",
            "sources": [],
        }
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

    async def test_typing_starts_repeats_and_stops_after_completion(self):
        update = make_update("What documents?")

        async def delayed_backend(_question):
            await asyncio.sleep(0.035)
            return {"status": "success", "answer": "Passport.", "sources": []}

        with (
            patch("telegram_bot.TYPING_INTERVAL_SECONDS", 0.01),
            patch("telegram_bot.ask_backend", side_effect=delayed_backend),
        ):
            await handle_question(update, SimpleNamespace())

        self.assertGreaterEqual(update.effective_chat.send_action.await_count, 2)
        calls_after_completion = update.effective_chat.send_action.await_count
        await asyncio.sleep(0.025)
        self.assertEqual(update.effective_chat.send_action.await_count, calls_after_completion)

    async def test_typing_stops_after_backend_failure(self):
        update = make_update("What documents?")

        async def failing_backend(_question):
            await asyncio.sleep(0.025)
            raise httpx.ConnectError("synthetic failure")

        import httpx
        with (
            patch("telegram_bot.TYPING_INTERVAL_SECONDS", 0.01),
            patch("telegram_bot.ask_backend", side_effect=failing_backend),
        ):
            await handle_question(update, SimpleNamespace())

        calls_after_failure = update.effective_chat.send_action.await_count
        self.assertGreaterEqual(calls_after_failure, 1)
        await asyncio.sleep(0.025)
        self.assertEqual(update.effective_chat.send_action.await_count, calls_after_failure)

    async def test_stop_typing_handles_task_cancellation_cleanly(self):
        task = asyncio.create_task(asyncio.sleep(10))
        await stop_typing(task)
        self.assertTrue(task.cancelled())

    async def test_handler_cancellation_stops_typing_without_orphan(self):
        update = make_update("What documents?")
        backend_started = asyncio.Event()

        async def pending_backend(_question):
            backend_started.set()
            await asyncio.Event().wait()

        with (
            patch("telegram_bot.TYPING_INTERVAL_SECONDS", 0.01),
            patch("telegram_bot.ask_backend", side_effect=pending_backend),
        ):
            handler_task = asyncio.create_task(handle_question(update, SimpleNamespace()))
            await backend_started.wait()
            await asyncio.sleep(0.02)
            handler_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await handler_task

        calls_after_cancellation = update.effective_chat.send_action.await_count
        await asyncio.sleep(0.025)
        self.assertEqual(
            update.effective_chat.send_action.await_count,
            calls_after_cancellation,
        )

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
