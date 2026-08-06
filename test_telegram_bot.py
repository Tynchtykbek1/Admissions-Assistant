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
    RESET_MESSAGES,
    START_MESSAGES,
    SYSTEM_DOCUMENT_UNAVAILABLE_MESSAGES,
    detect_text_language,
    format_backend_response,
    handle_question,
    help_command,
    reset_command,
    split_message,
    status_command,
    stop_typing,
    start_command,
)
from local_responses import LOCAL_RESPONSES


def make_update(text="question", language_code="ru", chat_id=123, user_id=456):
    message = SimpleNamespace(text=text, reply_text=AsyncMock())
    chat = SimpleNamespace(id=chat_id, send_action=AsyncMock())
    user = SimpleNamespace(id=user_id, language_code=language_code)
    return SimpleNamespace(message=message, effective_chat=chat, effective_user=user)


class TelegramFormattingTests(unittest.TestCase):
    def test_mixed_russian_questions_use_russian(self):
        for question in (
            "Sapienza дедлайн?",
            "Bocconi сроки?",
            "Politecnico документы?",
            "University of Messina поступление?",
            "DSU стипендия есть?",
            "IELTS нужен?",
            "Нужен ли TOEFL?",
            "Виза для Sapienza нужна?",
        ):
            self.assertEqual(detect_text_language(question), "ru")

    def test_english_only_questions_use_english(self):
        for question in (
            "What is the Sapienza deadline?",
            "Is IELTS required?",
            "What documents are required?",
        ):
            self.assertEqual(detect_text_language(question), "en")

    def test_no_letter_input_keeps_russian_default(self):
        self.assertEqual(detect_text_language(""), "ru")
        self.assertEqual(detect_text_language("?! … 123"), "ru")

    def test_success_displays_answer_only_without_technical_sources(self):
        result = {
            "answer": "The deadline is 30 April.",
            "status": "success",
            "sources": [
                {"filename": "admissions.pdf", "chunk_id": 1},
                {"filename": "admissions.pdf", "chunk_id": 1},
                {"filename": "faq.txt", "chunk_id": 2, "faq_id": 12},
            ],
        }
        formatted = format_backend_response(result)
        self.assertEqual(formatted, "The deadline is 30 April.")
        for technical_text in (
            "Источники", "Sources", "admissions.pdf", "faq.txt",
            "FAQ", "chunk", "0.91",
        ):
            self.assertNotIn(technical_text, formatted)

    def test_partial_information_displays_answer_only(self):
        result = {
            "answer": "The document supports only part of the answer.",
            "status": "partial_information",
            "sources": [{
                "filename": "private-faq.pdf",
                "chunk_id": 44,
                "faq_id": 91,
                "score": 0.91,
            }],
        }
        self.assertEqual(
            format_backend_response(result),
            "The document supports only part of the answer.",
        )

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

    def test_success_with_insufficient_phrase_remains_visible_without_sources(self):
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
        self.assertNotIn("FAQ.docx.pdf", formatted)
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

    def test_localizes_system_document_unavailable(self):
        result = {
            "status": "system_document_unavailable",
            "answer": "ignored",
            "sources": [],
        }
        self.assertEqual(
            format_backend_response(result, "ru"),
            SYSTEM_DOCUMENT_UNAVAILABLE_MESSAGES["ru"],
        )
        self.assertEqual(
            format_backend_response(result, "en"),
            SYSTEM_DOCUMENT_UNAVAILABLE_MESSAGES["en"],
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

    async def test_local_intents_bypass_every_backend_path_and_typing(self):
        cases = (
            ("Кто ты?", "ru", "identity"),
            ("What can you do?", "en", "capabilities"),
            ("Как связаться с менеджером?", "ru", "manager"),
            ("Tell me a joke", "en", "out_of_scope"),
        )
        for text, language, intent in cases:
            update = make_update(text)
            context = SimpleNamespace(chat_data={"conversation_id": "existing"})
            with (
                patch("telegram_bot.ask_backend", new=AsyncMock()) as ask,
                patch("telegram_bot.reset_backend", new=AsyncMock()) as reset,
                patch("telegram_bot.backend_status", new=AsyncMock()) as status,
                patch("telegram_bot.keep_typing", new=AsyncMock()) as typing,
            ):
                await handle_question(update, context)

            ask.assert_not_awaited()
            reset.assert_not_awaited()
            status.assert_not_awaited()
            typing.assert_not_awaited()
            update.effective_chat.send_action.assert_not_awaited()
            self.assertEqual(context.chat_data, {"conversation_id": "existing"})
            update.message.reply_text.assert_awaited_once_with(
                LOCAL_RESPONSES[intent][language],
                parse_mode=ParseMode.HTML,
            )

    async def test_manager_contacts_are_exact_and_safe_for_html_mode(self):
        for text, language in (
            ("Кто твой менеджер?", "ru"),
            ("Who is your manager?", "en"),
        ):
            update = make_update(text)
            await handle_question(update, SimpleNamespace(chat_data={}))
            sent = update.message.reply_text.await_args
            self.assertEqual(sent.kwargs["parse_mode"], ParseMode.HTML)
            self.assertIn("@TheLuckiestPersonEver", sent.args[0])
            self.assertIn("@maksatuniguide", sent.args[0])
            self.assertIn("@hellhg", sent.args[0])
            self.assertNotIn("<", sent.args[0])
            self.assertNotIn(">", sent.args[0])

    async def test_local_message_cannot_reach_unanswered_recording_path(self):
        update = make_update("Какая сегодня погода?")
        with patch(
            "telegram_bot.ask_backend",
            new=AsyncMock(side_effect=AssertionError("backend records unanswered")),
        ) as backend:
            await handle_question(update, SimpleNamespace(chat_data={}))
        backend.assert_not_awaited()

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
        backend.assert_awaited_once_with(update.message.text, "123", "456", None)

    async def test_greeting_with_question_calls_backend(self):
        update = make_update("Привет, какие документы нужны для поступления?")
        backend_result = {"answer": "Нужен паспорт.", "sources": []}
        with patch("telegram_bot.ask_backend", new=AsyncMock(return_value=backend_result)) as backend:
            await handle_question(update, SimpleNamespace())
        backend.assert_awaited_once_with(update.message.text, "123", "456", None)
        update.effective_chat.send_action.assert_awaited_with(action=ChatAction.TYPING)

    async def test_normal_admissions_question_calls_backend_exactly_once(self):
        update = make_update("Какие документы нужны для поступления?")
        context = SimpleNamespace(chat_data={"conversation_id": "existing"})
        result = {
            "status": "success",
            "answer": "Нужен паспорт.",
            "sources": [],
            "conversation_id": "updated",
        }
        with patch(
            "telegram_bot.ask_backend", new=AsyncMock(return_value=result)
        ) as backend:
            await handle_question(update, context)
        backend.assert_awaited_once_with(
            update.message.text, "123", "456", "existing"
        )
        self.assertEqual(context.chat_data["conversation_id"], "updated")

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

    async def test_mixed_russian_question_uses_russian_backend_fallback(self):
        update = make_update("Sapienza дедлайн?")
        backend_result = {
            "status": "insufficient_document_information",
            "answer": "ignored",
            "sources": [{
                "filename": "hidden.pdf",
                "faq_id": 17,
                "chunk_id": 4,
                "score": 0.91,
            }],
        }
        with patch(
            "telegram_bot.ask_backend", new=AsyncMock(return_value=backend_result)
        ):
            await handle_question(update, SimpleNamespace())
        sent = update.message.reply_text.await_args.args[0]
        self.assertEqual(sent, NO_INFORMATION_MESSAGES["ru"])
        for technical_text in ("hidden.pdf", "FAQ", "chunk", "0.91", "Источники", "Sources"):
            self.assertNotIn(technical_text, sent)

    async def test_english_question_uses_english_backend_fallback(self):
        update = make_update("What documents are required?")
        backend_result = {
            "status": "insufficient_document_information",
            "answer": "ignored",
            "sources": [],
        }
        with patch(
            "telegram_bot.ask_backend", new=AsyncMock(return_value=backend_result)
        ):
            await handle_question(update, SimpleNamespace())
        self.assertEqual(
            update.message.reply_text.await_args.args[0],
            NO_INFORMATION_MESSAGES["en"],
        )

    async def test_russian_manager_response_uses_russian_language_detection(self):
        question = "Кто твой менеджер?"
        self.assertEqual(detect_text_language(f"Sapienza {question}"), "ru")
        update = make_update(question)
        await handle_question(update, SimpleNamespace(chat_data={}))
        self.assertEqual(
            update.message.reply_text.await_args.args[0],
            LOCAL_RESPONSES["manager"]["ru"],
        )

    async def test_localized_backend_error(self):
        for question, language in (("Какие требования?", "ru"), ("What requirements?", "en")):
            update = make_update(question)
            with patch("telegram_bot.ask_backend", new=AsyncMock(side_effect=ValueError("bad"))):
                await handle_question(update, SimpleNamespace())
            self.assertEqual(update.message.reply_text.await_args.args[0], ERROR_MESSAGES[language])

    async def test_typing_starts_repeats_and_stops_after_completion(self):
        update = make_update("What documents?")

        async def delayed_backend(*_args):
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

        async def failing_backend(*_args):
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

    async def test_typing_stops_for_timeout_connection_and_http_errors(self):
        import httpx

        request = httpx.Request("POST", "http://backend/chat")
        errors = (
            httpx.ReadTimeout("timeout", request=request),
            httpx.ConnectError("connection", request=request),
            httpx.HTTPStatusError(
                "status",
                request=request,
                response=httpx.Response(500, request=request),
            ),
        )
        for error in errors:
            update = make_update("What documents?")

            async def failing_backend(*_args):
                await asyncio.sleep(0.015)
                raise error

            with (
                patch("telegram_bot.TYPING_INTERVAL_SECONDS", 0.005),
                patch("telegram_bot.ask_backend", side_effect=failing_backend),
            ):
                await handle_question(update, SimpleNamespace())

            calls_after_failure = update.effective_chat.send_action.await_count
            self.assertGreaterEqual(calls_after_failure, 1)
            await asyncio.sleep(0.015)
            self.assertEqual(
                update.effective_chat.send_action.await_count,
                calls_after_failure,
            )
            self.assertEqual(
                update.message.reply_text.await_args.args[0],
                ERROR_MESSAGES["en"],
            )

    async def test_stop_typing_handles_task_cancellation_cleanly(self):
        task = asyncio.create_task(asyncio.sleep(10))
        await stop_typing(task)
        self.assertTrue(task.cancelled())

    async def test_handler_cancellation_stops_typing_without_orphan(self):
        update = make_update("What documents?")
        backend_started = asyncio.Event()

        async def pending_backend(*_args):
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

    async def test_reset_is_localized_and_calls_backend_with_identifiers(self):
        update = make_update("/reset")
        context = SimpleNamespace(chat_data={})
        result = {"conversation_id": "conversation-1", "cleared_messages": 2}
        with patch(
            "telegram_bot.reset_backend", new=AsyncMock(return_value=result)
        ) as backend:
            await reset_command(update, context)
        backend.assert_awaited_once_with("123", "456", None)
        self.assertEqual(context.chat_data["conversation_id"], "conversation-1")
        self.assertEqual(
            update.message.reply_text.await_args.args[0], RESET_MESSAGES["ru"]
        )

    async def test_status_shows_system_document_filename(self):
        update = make_update("/status")
        context = SimpleNamespace(chat_data={})
        result = {
            "status": "ok",
            "conversation_id": "conversation-1",
            "active_document_id": 1,
            "active_document_filename": "FAQ.docx.pdf",
        }
        with patch(
            "telegram_bot.backend_status",
            new=AsyncMock(return_value=result),
        ) as backend:
            await status_command(update, context)
        backend.assert_awaited_once_with("123", "456", None)
        self.assertIn(
            "FAQ.docx.pdf",
            update.message.reply_text.await_args.args[0],
        )

    async def test_same_chat_questions_are_serialized(self):
        active = 0
        maximum = 0

        async def backend(*_args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"status": "success", "answer": "ok", "sources": []}

        first = make_update("Question one", chat_id=900)
        second = make_update("Question two", chat_id=900)
        with patch("telegram_bot.ask_backend", side_effect=backend):
            await asyncio.gather(
                handle_question(first, SimpleNamespace()),
                handle_question(second, SimpleNamespace()),
            )
        self.assertEqual(maximum, 1)

    async def test_different_chats_can_run_concurrently(self):
        active = 0
        maximum = 0

        async def backend(*_args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"status": "success", "answer": "ok", "sources": []}

        first = make_update("Question one", chat_id=901)
        second = make_update("Question two", chat_id=902)
        with patch("telegram_bot.ask_backend", side_effect=backend):
            await asyncio.gather(
                handle_question(first, SimpleNamespace()),
                handle_question(second, SimpleNamespace()),
            )
        self.assertEqual(maximum, 2)

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
            help_text = help_update.message.reply_text.await_args.args[0]
            self.assertIn("@TheLuckiestPersonEver", help_text)
            self.assertIn("@maksatuniguide", help_text)


if __name__ == "__main__":
    unittest.main()
