import argparse
import csv
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unanswered admissions questions as UTF-8 CSV."
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=sorted(UNANSWERED_QUESTION_STATUSES),
        help="Filter by status. May be supplied more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unanswered_questions.csv"),
        help="CSV output path (default: unanswered_questions.csv).",
    )
    return parser.parse_args()


def export_csv(output_path: Path, statuses: list[str] | None = None) -> int:
    rows = list_unanswered_questions(statuses)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    row_count = export_csv(args.output, args.status)
    print(f"Exported {row_count} unanswered questions to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
