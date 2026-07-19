import unittest
from unittest.mock import patch

import numpy as np

from answer_generator import generate_basic_answer
from embedding_model import get_embedding_model
from embedding_retriever import find_relevant_chunks_semantic


def build_chunks(items: list[tuple[str, str]]) -> list[dict]:
    model = get_embedding_model()
    retrieval_texts = [f"{question}\n{answer}" for question, answer in items]
    embeddings = model.encode(retrieval_texts, normalize_embeddings=True)

    return [
        {
            "chunk_id": index + 1,
            "faq_id": index + 1,
            "filename": "faq.txt",
            "text": answer,
            "text_for_retrieval": retrieval_texts[index],
            "embedding": embeddings[index]
        }
        for index, (question, answer) in enumerate(items)
    ]


class SemanticRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.english_chunks = build_chunks([
            (
                "When can applications be submitted?",
                "Applications can usually be submitted from April until August."
            ),
            ("What is the tuition fee?", "The tuition fee is 3000 EUR per year."),
            ("Which visa documents are needed?", "Applicants need a passport and insurance."),
            ("Are scholarships guaranteed?", "Scholarships are not guaranteed.")
        ])

    def test_english_paraphrase_retrieves_application_period(self):
        results = find_relevant_chunks_semantic(
            "When is it possible to apply?",
            self.english_chunks,
            top_k=3,
            min_score=0.20,
            min_context_chunks=1
        )

        self.assertTrue(results)
        self.assertTrue(any("April until August" in item["text"] for item in results))
        answer = generate_basic_answer("When is it possible to apply?", results)
        self.assertNotIn("not enough information", answer.lower())

    def test_russian_paraphrase_retrieves_russian_faq(self):
        russian_chunks = [
            {
                "chunk_id": 1,
                "faq_id": 1,
                "filename": "faq.txt",
                "text": "Документы можно подавать с апреля по август.",
                "embedding": np.array([1.0, 0.0])
            },
            {
                "chunk_id": 2,
                "faq_id": 2,
                "filename": "faq.txt",
                "text": "Для визы нужен паспорт.",
                "embedding": np.array([0.0, 1.0])
            }
        ]

        class SupportedRussianModel:
            def encode(self, text, normalize_embeddings=True):
                return np.array([1.0, 0.0])

        with patch(
            "embedding_retriever.get_embedding_model",
            return_value=SupportedRussianModel()
        ):
            results = find_relevant_chunks_semantic(
                "С какого месяца начинается подача?",
                russian_chunks,
                top_k=1,
                min_score=0.20,
                min_context_chunks=1
            )

        self.assertTrue(any(item["faq_id"] == 1 for item in results))

    def test_unrelated_fallback_is_limited(self):
        results = find_relevant_chunks_semantic(
            "How do I repair a bicycle chain?",
            self.english_chunks,
            top_k=5,
            min_score=0.99,
            min_context_chunks=2
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["retrieval_fallback"] for item in results))


if __name__ == "__main__":
    unittest.main()
