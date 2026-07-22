import unittest
from unittest.mock import patch

import numpy as np

from embedding_model import get_embedding_model
from embedding_retriever import find_relevant_chunks_semantic, normalize_faq_question
from retrieval_settings import SEMANTIC_SCORE_THRESHOLD, SEMANTIC_TOP_K


FAQ_ITEMS = [
    (
        32,
        "Какие дедлайны? (подача в университеты)",
        "Подача в университеты начинается с середины декабря и продолжается до середины мая."
    ),
    (
        41,
        "Как апостилировать документы?",
        "Апостиль можно получить через Министерство юстиции или переводческую компанию."
    ),
    (5, "Какие документы нужны для визы?", "Для визы нужны паспорт и письмо о зачислении."),
    (8, "Есть ли гарантии на стипендию?", "Получение стипендии не гарантируется."),
    (10, "Как найти жильё?", "Студенты могут искать общежитие или частную квартиру."),
    (12, "Сколько стоит обучение?", "Стоимость обучения зависит от университета."),
    (14, "Какой уровень языка требуется?", "Требования зависят от выбранной программы.")
]


def build_real_multilingual_chunks() -> list[dict]:
    model = get_embedding_model()
    retrieval_texts = [
        f"{question}\n{answer}"
        for _, question, answer in FAQ_ITEMS
    ]
    embeddings = model.encode(retrieval_texts, normalize_embeddings=True)

    return [
        {
            "chunk_id": faq_id,
            "faq_id": faq_id,
            "question": question,
            "filename": "focused_faq.txt",
            "text": answer,
            "text_for_retrieval": retrieval_texts[index],
            "embedding": embeddings[index]
        }
        for index, (faq_id, question, answer) in enumerate(FAQ_ITEMS)
    ]


class RealMultilingualRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = build_real_multilingual_chunks()

    def assert_faq_retrieved(self, question: str, expected_faq_id: int):
        results = find_relevant_chunks_semantic(
            question,
            self.chunks,
            top_k=SEMANTIC_TOP_K,
            min_score=SEMANTIC_SCORE_THRESHOLD,
            min_context_chunks=0
        )

        self.assertIn(expected_faq_id, [result.get("faq_id") for result in results])

    def test_apostille_question_retrieves_faq_41(self):
        self.assert_faq_retrieved("Как апостилировать документы?", 41)

    def test_application_question_retrieves_faq_32(self):
        self.assert_faq_retrieved("Когда можно подавать документы?", 32)

    def test_application_month_paraphrase_retrieves_faq_32(self):
        self.assert_faq_retrieved("С какого месяца начинается подача?", 32)

    def test_deadline_question_retrieves_faq_32(self):
        results = find_relevant_chunks_semantic(
            "Какие дедлайны?", self.chunks, top_k=3, min_score=0.0
        )
        self.assertEqual(results[0]["faq_id"], 32)
        self.assertEqual(results[0]["faq_match_type"], "exact")

    def test_exact_apostille_question_ranks_first(self):
        results = find_relevant_chunks_semantic(
            "Как апостилировать документы?", self.chunks, top_k=3, min_score=0.0
        )
        self.assertEqual(results[0]["faq_id"], 41)
        self.assertEqual(results[0]["faq_match_type"], "exact")

    def test_unrelated_fallback_is_limited(self):
        results = find_relevant_chunks_semantic(
            "Как отремонтировать велосипед?",
            self.chunks,
            top_k=SEMANTIC_TOP_K,
            min_score=0.99,
            min_context_chunks=2
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["retrieval_fallback"] for result in results))


class DeterministicHybridRankingTests(unittest.TestCase):
    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            return np.array([1.0, 0.0])

    def setUp(self):
        self.chunks = [
            {
                "chunk_id": 1, "faq_id": 1,
                "question": "Какие дедлайны? (подача документов обычно начинается в...)",
                "filename": "faq.txt", "text": "Deadline answer",
                "embedding": np.array([0.40, 0.0])
            },
            {
                "chunk_id": 2, "faq_id": 2, "question": "Какие нужны документы?",
                "filename": "faq.txt", "text": "Documents answer",
                "embedding": np.array([0.95, 0.0])
            },
            {
                "chunk_id": 3, "filename": "guide.txt", "text": "Ordinary chunk",
                "embedding": np.array([0.80, 0.0])
            }
        ]

    def retrieve(self, question):
        with patch("embedding_retriever.get_embedding_model", return_value=self.FakeModel()):
            return find_relevant_chunks_semantic(question, self.chunks, top_k=3, min_score=0.0)

    def test_normalization(self):
        self.assertEqual(normalize_faq_question("  Какие   дедлайны?  "), "какие дедлайны")
        self.assertEqual(
            normalize_faq_question("Какие дедлайны? (подача документов обычно начинается...)") ,
            "какие дедлайны"
        )
        self.assertEqual(normalize_faq_question("FAQ № 2026!"), "faq 2026")

    def test_exact_match_beats_higher_semantic_scores(self):
        results = self.retrieve("Какие дедлайны?")
        self.assertEqual(results[0]["faq_id"], 1)
        self.assertEqual(results[0]["score"], 0.4)
        self.assertEqual(results[0]["faq_match_type"], "exact")

    def test_unrelated_faq_has_no_match_boost(self):
        result = next(item for item in self.retrieve("Какие дедлайны?") if item.get("faq_id") == 2)
        self.assertIsNone(result["faq_match_type"])
        self.assertEqual(result["faq_match_boost"], 0.0)

    def test_non_faq_chunk_uses_semantic_score_only(self):
        result = next(item for item in self.retrieve("Какие дедлайны?") if item["chunk_id"] == 3)
        self.assertIsNone(result["faq_match_type"])
        self.assertEqual(result["final_score"], result["score"])


if __name__ == "__main__":
    unittest.main()
