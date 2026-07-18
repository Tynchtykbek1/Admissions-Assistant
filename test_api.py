from pathlib import Path
import requests


BASE_URL = "http://127.0.0.1:8000"

TEST_DIR = Path("test_files")
TEST_FILE = TEST_DIR / "admissions_sample_document.txt"


SAMPLE_DOCUMENT = """
University Admissions Sample Document

University: University of Milan
Country: Italy
Program: Bachelor in International Relations
Language of instruction: English

Admission Requirements:
Applicants must have completed secondary school education or an equivalent qualification.
Applicants must provide a valid passport, high school diploma, transcript of records, motivation letter, and proof of English language proficiency.
For English-taught bachelor programs, the minimum English requirement is IELTS 6.0 or an equivalent certificate such as TOEFL or Cambridge English.

Application Deadline:
The standard application deadline for non-EU students is 30 April.
Late applications may not be accepted unless the university officially extends the deadline.
Students should always check the official university website before applying.

Tuition Fees:
The estimated tuition fee is 3000 EUR per year.
The final tuition amount may depend on the student’s family income, country of residence, and university regulations.

Visa Documents:
Non-EU students usually need a passport, admission letter, proof of financial means, accommodation proof, health insurance, and visa application form.
The assistant must not guarantee visa approval because the final decision belongs to the embassy.

Important Business Rules:
The assistant must not promise guaranteed admission.
The assistant must not promise guaranteed visa approval.
The assistant must not invent deadlines, tuition fees, or scholarship conditions.
If the answer is not available in the document, the assistant should say that there is not enough information and recommend contacting a human advisor.
"""


TEST_QUESTIONS = [
    {
        "question": "What English level is required?",
        "expected_contains": "IELTS 6.0"
    },
    {
        "question": "What is the application deadline?",
        "expected_contains": "30 April"
    },
    {
        "question": "Can I study medicine in Canada?",
        "expected_contains": "not enough information"
    }
]


def create_test_file():
    TEST_DIR.mkdir(exist_ok=True)

    with open(TEST_FILE, "w", encoding="utf-8") as file:
        file.write(SAMPLE_DOCUMENT.strip())

    print(f"Created test file: {TEST_FILE}")


def upload_test_file():
    with open(TEST_FILE, "rb") as file:
        files = {
            "file": (TEST_FILE.name, file, "text/plain")
        }

        response = requests.post(
            f"{BASE_URL}/upload",
            files=files,
            timeout=30
        )

    response.raise_for_status()
    data = response.json()

    print("\nUpload result:")
    print(f"Filename: {data['filename']}")
    print(f"Text length: {data['text_length']}")
    print(f"Chunks count: {data['chunks_count']}")


def ask_question(question: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/ask-semantic",
        json={"question": question},
        timeout=60
    )

    response.raise_for_status()
    return response.json()


def run_tests():
    print("\nRunning API tests...")

    for test_case in TEST_QUESTIONS:
        question = test_case["question"]
        expected = test_case["expected_contains"].lower()

        result = ask_question(question)
        answer = result["answer"]
        answer_lower = answer.lower()

        print("\nQuestion:")
        print(question)

        print("Answer:")
        print(answer)

        if expected in answer_lower:
            print("Result: PASS")
        else:
            print("Result: FAIL")
            print(f"Expected answer to contain: {test_case['expected_contains']}")


def main():
    create_test_file()
    upload_test_file()
    run_tests()


if __name__ == "__main__":
    main()