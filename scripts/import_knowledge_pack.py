import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_importer import import_knowledge_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and import an approved company knowledge pack")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--document-name", required=True)
    parser.add_argument("--base-document-id", required=True, type=int)
    parser.add_argument("--scope", choices=("production", "demo"), default="production")
    parser.add_argument("--apply", action="store_true", help="Apply changes; default is dry-run")
    args = parser.parse_args()
    try:
        records = json.loads(args.pack.read_text(encoding="utf-8"))
        report = import_knowledge_pack(
            records, args.database, args.document_name, args.base_document_id,
            scope=args.scope, apply=args.apply,
        )
    except Exception as error:
        print(f"Knowledge pack import failed: {error}")
        return 1
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
