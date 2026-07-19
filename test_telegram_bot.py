import unittest

from telegram_bot import format_backend_response, split_message


class TelegramFormattingTests(unittest.TestCase):
    def test_formats_unique_source_filenames(self):
        result = {
            "answer": "The deadline is 30 April.",
            "sources": [
                {"filename": "admissions.pdf"},
                {"filename": "admissions.pdf"},
                {"filename": "faq.txt"}
            ]
        }

        formatted = format_backend_response(result)

        self.assertIn("The deadline is 30 April.", formatted)
        self.assertEqual(formatted.count("admissions.pdf"), 1)
        self.assertIn("faq.txt", formatted)

    def test_splits_long_messages(self):
        parts = split_message("one two three four", max_length=10)

        self.assertEqual(" ".join(parts), "one two three four")
        self.assertTrue(all(len(part) <= 10 for part in parts))


if __name__ == "__main__":
    unittest.main()
