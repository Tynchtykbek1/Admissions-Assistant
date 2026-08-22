import unittest

from admissions_rag_assistant.document_processor import parse_faq_entries, split_text_into_chunks


class FaqParserTests(unittest.TestCase):
    def test_question_with_explanation_after_question_mark(self):
        text = """
32. Какие дедлайны? (подача в университеты)
Подача начинается с середины декабря и продолжается до середины мая.
"""

        entries = parse_faq_entries(text)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["faq_id"], 32)
        self.assertIn("Какие дедлайны?", entries[0]["question"])
        self.assertIn("(подача в университеты)", entries[0]["question"])
        self.assertIn("с середины декабря", entries[0]["answer"])

    def test_parenthesis_number_separator(self):
        text = """
41) Как апостилировать документы?
Апостиль можно получить через Министерство юстиции.
"""

        entries = parse_faq_entries(text)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["faq_id"], 41)

    def test_standard_document_still_uses_normal_chunks(self):
        text = """
University admissions information
Applications open in April and close in August.
Applicants need a passport and school transcript.
"""

        self.assertEqual(parse_faq_entries(text), [])
        self.assertTrue(split_text_into_chunks(text))


if __name__ == "__main__":
    unittest.main()
