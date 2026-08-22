import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import UNANSWERED_QUESTION_STATUSES, list_unanswered_questions
CSV_FIELDS = (
    "id",
    "question",
    "standalone_question",
    "occurrence_count",
    "max_similarity_score",
    "retrieved_faq_ids",
    "reason",
    "status",
    "first_seen_at",
    "last_seen_at",
)
CLEAN_CSV_FIELDS = (
    "question",
    "language",
    "times_asked",
    "first_seen",
    "last_seen",
    "best_retrieval_score",
    "related_faq_ids",
    "reason",
    "status",
)
TECHNICAL_ERROR_TEXTS = frozenset({
    "the service is temporarily unavailable please try again in a few minutes",
    "the knowledge base is temporarily unavailable please try again later",
    "i couldn t get an answer right now please try again a little later",
    "сервис временно перегружен попробуйте повторить вопрос через несколько минут",
    "база знаний временно недоступна пожалуйста попробуйте позже",
    "сейчас не удалось получить ответ пожалуйста попробуйте ещё раз немного позже",
})
URL_ONLY_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
NOISE_QUESTIONS = frozenset({
    "привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер", "салам",
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "пока", "до свидания", "спасибо", "спасибо пока",
    "bye", "goodbye", "thanks", "thank you",
    "как тебя зовут", "кто ты", "как называется этот бот", "представься", "что ты за бот",
    "what is your name", "what s your name", "who are you", "what bot are you", "introduce yourself",
    "что ты умеешь", "на какие вопросы ты можешь ответить", "какие вопросы можно задавать",
    "чем ты можешь помочь", "в чем ты можешь помочь", "о чем тебя можно спрашивать",
    "что можно у тебя спросить", "what can you do", "what questions can i ask",
    "how can you help", "what can i ask you", "what topics do you cover",
    "кто твой менеджер", "кто менеджер", "кто может помочь с поступлением",
    "как связаться с менеджером", "как связаться с человеком", "можно поговорить с менеджером",
    "кому написать по поводу поступления", "кто знает все о поступлении",
    "кто может проконсультировать", "дай контакты менеджера", "с кем можно связаться",
    "кому обратиться за помощью", "who is your manager", "how can i contact a manager",
    "can i speak to a human", "who can help me with admission",
    "who should i contact about admissions", "can i contact an admissions manager",
    "who can give me personal assistance", "give me the manager contacts",
    "какая сегодня погода", "расскажи анекдот", "сколько будет 2 2", "напиши стих",
    "кто выиграл матч", "как приготовить пиццу", "what is the weather", "tell me a joke",
    "what is 2 2", "write a poem", "who won the match", "how do i cook pizza",
})


def parse_since(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--since must use YYYY-MM-DD") from error


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--min-count must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--min-count must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unanswered admissions questions as CSV."
    )
    parser.add_argument(
        "--status", action="append", choices=sorted(UNANSWERED_QUESTION_STATUSES),
        help="Filter by status. May be supplied more than once.",
    )
    parser.add_argument("--clean", action="store_true", help="Export grouped review CSV.")
    parser.add_argument("--min-count", type=positive_int, default=1)
    parser.add_argument("--language", choices=("ru", "en", "unknown"))
    parser.add_argument("--since", type=parse_since)
    parser.add_argument(
        "--sort", choices=("most-frequent", "newest", "oldest"),
        default="most-frequent",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("unanswered_questions.csv"),
        help="CSV output path (default: unanswered_questions.csv).",
    )
    return parser.parse_args()


def detect_language(text: str) -> str:
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cyrillic > latin:
        return "ru"
    if latin > cyrillic:
        return "en"
    return "unknown"


def normalize_clean_question(text: str) -> str:
    without_punctuation = re.sub(r"[^\w\s]", " ", text.casefold(), flags=re.UNICODE)
    return " ".join(without_punctuation.split())


def is_noise_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped or URL_ONLY_RE.fullmatch(stripped) or stripped.startswith("/"):
        return True
    if not re.search(r"[A-Za-zА-Яа-яЁё0-9]", stripped):
        return True
    normalized = normalize_clean_question(stripped)
    return normalized in NOISE_QUESTIONS or normalized in TECHNICAL_ERROR_TEXTS


def _faq_ids(value: str | None) -> set[int]:
    try:
        values = json.loads(value) if value else []
    except (json.JSONDecodeError, TypeError):
        return set()
    return {
        int(item) for item in values
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
    }


def _representative_key(row: dict) -> tuple:
    question = row["question"].strip()
    letters = "".join(re.findall(r"[A-Za-zА-Яа-яЁё]", question))
    is_uppercase = bool(letters) and letters == letters.upper()
    return (
        -int(row["occurrence_count"]),
        -_timestamp(row["last_seen_at"]),
        is_uppercase,
        question.casefold(),
        int(row["id"]),
    )


def group_clean_rows(rows: list[dict]) -> tuple[list[dict], int]:
    grouped: dict[str, list[dict]] = {}
    noise_count = 0
    for row in rows:
        question = (row.get("question") or "").strip()
        key = normalize_clean_question(question)
        if is_noise_question(question) or not key:
            noise_count += 1
            continue
        grouped.setdefault(key, []).append(row)

    output = []
    for normalized, variants in grouped.items():
        representative = sorted(variants, key=_representative_key)[0]
        faq_ids = set()
        reasons = set()
        statuses = set()
        scores = []
        for row in variants:
            faq_ids.update(_faq_ids(row.get("retrieved_faq_ids")))
            reasons.update(filter(None, (row.get("reason") or "").split(",")))
            statuses.add(row["status"])
            if row.get("max_similarity_score") is not None:
                scores.append(float(row["max_similarity_score"]))
        output.append({
            "question": representative["question"].strip(),
            "language": detect_language(representative["question"]),
            "times_asked": sum(int(row["occurrence_count"]) for row in variants),
            "first_seen": min(row["first_seen_at"] for row in variants),
            "last_seen": max(row["last_seen_at"] for row in variants),
            "best_retrieval_score": max(scores) if scores else "",
            "related_faq_ids": ",".join(str(item) for item in sorted(faq_ids)),
            "reason": ",".join(sorted(reason.strip() for reason in reasons if reason.strip())),
            "status": ",".join(sorted(statuses)),
            "_normalized": normalized,
        })
    return output, noise_count


def _sort_clean_rows(rows: list[dict], sort_mode: str) -> list[dict]:
    if sort_mode == "newest":
        return sorted(rows, key=lambda row: (-_timestamp(row["last_seen"]), row["_normalized"]))
    if sort_mode == "oldest":
        return sorted(rows, key=lambda row: (_timestamp(row["first_seen"]), row["_normalized"]))
    return sorted(rows, key=lambda row: (
        -row["times_asked"], -_timestamp(row["last_seen"]), row["_normalized"]
    ))


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _atomic_write(
    output_path: Path, rows: list[dict], fields: tuple[str, ...], encoding: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding=encoding, newline="", delete=False,
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp",
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def export_csv(output_path: Path, statuses: list[str] | None = None) -> int:
    """Backward-compatible raw export."""
    rows = list_unanswered_questions(statuses)
    _atomic_write(output_path, rows, CSV_FIELDS, "utf-8")
    return len(rows)


def export_clean_csv(
    output_path: Path,
    statuses: list[str] | None = None,
    *,
    min_count: int = 1,
    language: str | None = None,
    since: date | None = None,
    sort_mode: str = "most-frequent",
) -> dict:
    database_rows = list_unanswered_questions(statuses)
    groups, noise_count = group_clean_rows(database_rows)
    groups = [
        row for row in groups
        if row["times_asked"] >= min_count
        and (language is None or row["language"] == language)
        and (
            since is None
            or datetime.fromisoformat(row["last_seen"].replace("Z", "+00:00")).date() >= since
        )
    ]
    groups = _sort_clean_rows(groups, sort_mode)
    _atomic_write(output_path, groups, CLEAN_CSV_FIELDS, "utf-8-sig")
    return {
        "database_rows_read": len(database_rows),
        "clean_groups_exported": len(groups),
        "duplicate_variants_merged": max(0, len(database_rows) - noise_count - len(group_clean_rows(database_rows)[0])),
        "noise_rows_excluded": noise_count,
    }


def main() -> int:
    args = parse_args()
    if not args.clean:
        count = export_csv(args.output, args.status)
        print(f"Database rows read: {count}; output: {args.output}")
        return 0
    summary = export_clean_csv(
        args.output, args.status, min_count=args.min_count,
        language=args.language, since=args.since, sort_mode=args.sort,
    )
    filters = (
        f"status={','.join(args.status or ['open', 'reviewed'])}, "
        f"min-count={args.min_count}, language={args.language or 'all'}, "
        f"since={args.since or 'all'}, sort={args.sort}"
    )
    print(
        f"Database rows read: {summary['database_rows_read']}; "
        f"clean groups exported: {summary['clean_groups_exported']}; "
        f"duplicate variants merged: {summary['duplicate_variants_merged']}; "
        f"noise rows excluded: {summary['noise_rows_excluded']}; "
        f"filters: {filters}; output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
