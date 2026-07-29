import unittest
from unittest.mock import patch

import numpy as np

from answer_generator import generate_basic_answer
from embedding_retriever import (
    build_retrieval_diagnostics,
    find_relevant_chunks_semantic,
    normalize_faq_question,
    normalize_retrieval_query,
)
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
    (14, "Какой уровень языка требуется?", "Требования зависят от выбранной программы."),
    (
        16,
        "Нужна ли виза?",
        "Студенческая виза нужна иностранным студентам до поездки."
    ),
]


class SyntheticMultilingualModel:
    """Deterministic offline embeddings for retrieval plumbing tests."""

    STEM_GROUPS = (
        ("дедлайн", "подава", "подач", "месяц", "декабр", "мая"),
        ("апост",),
        ("виз",),
        ("стипенд",),
        ("жиль", "общежит"),
        ("стоим", "обучен"),
        ("язык",),
        ("документ",),
    )

    def _encode_one(self, text: str) -> np.ndarray:
        normalized = normalize_retrieval_query(text).casefold()
        vector = np.array(
            [
                float(any(stem in normalized for stem in stems))
                for stems in self.STEM_GROUPS
            ]
        )
        magnitude = np.linalg.norm(vector)
        return vector / magnitude if magnitude else vector

    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, str):
            return self._encode_one(text)
        return np.stack([self._encode_one(item) for item in text])


def build_synthetic_multilingual_chunks(model) -> list[dict]:
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


class MultilingualRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SyntheticMultilingualModel()
        cls.chunks = build_synthetic_multilingual_chunks(cls.model)
        cls.model_patch = patch(
            "embedding_retriever.get_embedding_model",
            return_value=cls.model,
        )
        cls.model_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.model_patch.stop()

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

    def test_admissions_slang_retrieves_same_faq_as_canonical_query(self):
        canonical_query = "Какие документы нужны для поступления?"
        slang_query = "Какие доки нужны для поступления?"
        self.assertEqual(
            normalize_retrieval_query(canonical_query),
            normalize_retrieval_query(slang_query),
        )
        canonical = find_relevant_chunks_semantic(
            canonical_query, self.chunks, top_k=3, min_score=0.0
        )
        slang = find_relevant_chunks_semantic(
            slang_query, self.chunks, top_k=3, min_score=0.0
        )
        self.assertEqual(
            [item["chunk_id"] for item in slang],
            [item["chunk_id"] for item in canonical],
        )
        self.assertEqual(
            [item["score"] for item in slang],
            [item["score"] for item in canonical],
        )

    def test_visa_query_retrieves_direct_visa_faq_when_present(self):
        results = find_relevant_chunks_semantic(
            "Нужна ли виза?", self.chunks, top_k=3, min_score=0.0
        )
        self.assertEqual(results[0]["faq_id"], 16)

    def test_missing_visa_information_does_not_fabricate_answer(self):
        chunks_without_visa = [
            chunk for chunk in self.chunks
            if "виз" not in (chunk.get("question", "") + chunk["text"]).casefold()
        ]
        results = find_relevant_chunks_semantic(
            "Нужна ли виза?", chunks_without_visa, top_k=3, min_score=0.99
        )
        self.assertEqual(results, [])
        self.assertIn(
            "not enough information",
            generate_basic_answer("Нужна ли виза?", results).casefold()
        )

    def test_exact_apostille_question_ranks_first(self):
        results = find_relevant_chunks_semantic(
            "Как апостилировать документы?", self.chunks, top_k=3, min_score=0.0
        )
        self.assertEqual(results[0]["faq_id"], 41)
        self.assertEqual(results[0]["faq_match_type"], "exact")

    def test_unrelated_chunks_are_not_forced_to_minimum_context(self):
        results = find_relevant_chunks_semantic(
            "Как отремонтировать велосипед?",
            self.chunks,
            top_k=SEMANTIC_TOP_K,
            min_score=0.99,
            min_context_chunks=2
        )

        self.assertEqual(results, [])

    def test_conservative_fallback_requires_its_own_threshold(self):
        results = find_relevant_chunks_semantic(
            "Где поставить апостиль?",
            self.chunks,
            top_k=3,
            min_score=0.99,
            min_context_chunks=3,
            fallback_score_threshold=0.70,
        )
        self.assertTrue(results)
        self.assertTrue(any(
            result["retrieval_fallback"] or result["score"] >= 0.99
            for result in results
        ))


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

    def test_retrieval_query_normalization_is_token_aware(self):
        self.assertEqual(normalize_retrieval_query("Доки"), "документы")
        self.assertEqual(
            normalize_retrieval_query("Какие доки нужны для поступления?"),
            "Какие документы нужны для поступления?"
        )
        self.assertEqual(normalize_retrieval_query("Токидоки"), "Токидоки")

    def test_exact_match_beats_higher_semantic_scores(self):
        results = self.retrieve("Какие дедлайны?")
        self.assertEqual(results[0]["faq_id"], 1)
        self.assertEqual(results[0]["score"], 0.4)
        self.assertEqual(results[0]["faq_match_type"], "exact")

    def test_unrelated_faq_has_no_match_boost(self):
        with patch("embedding_retriever.get_embedding_model", return_value=self.FakeModel()):
            result = next(
                item for item in build_retrieval_diagnostics("Какие дедлайны?", self.chunks)
                if item.get("faq_id") == 2
            )
        self.assertIsNone(result["faq_match_type"])
        self.assertEqual(result["faq_match_boost"], 0.0)

    def test_non_faq_chunk_uses_semantic_score_only(self):
        with patch("embedding_retriever.get_embedding_model", return_value=self.FakeModel()):
            result = next(
                item for item in build_retrieval_diagnostics("Какие дедлайны?", self.chunks)
                if item["index"] == 2
            )
        self.assertIsNone(result["faq_match_type"])
        self.assertEqual(result["final_score"], result["score"])

    def test_diagnostics_include_rank_and_query_forms(self):
        with patch("embedding_retriever.get_embedding_model", return_value=self.FakeModel()):
            diagnostics = build_retrieval_diagnostics("Доки", self.chunks)
        self.assertEqual(diagnostics[0]["original_query"], "Доки")
        self.assertEqual(diagnostics[0]["retrieval_query"], "документы")
        self.assertEqual(diagnostics[0]["rank"], 1)
        self.assertIn("source", diagnostics[0])
        self.assertIn("preview", diagnostics[0])

    def test_non_faq_document_retrieval_still_uses_semantics(self):
        results = find_relevant_chunks_semantic(
            "ordinary guide", [self.chunks[2]], top_k=3, min_score=0.0
        )
        ordinary = results[0]
        self.assertNotIn("faq_id", ordinary)
        self.assertIsNone(ordinary["faq_match_type"])


if __name__ == "__main__":
    unittest.main()
