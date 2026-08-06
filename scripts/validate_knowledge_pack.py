import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_validator import KnowledgeValidationError, validate_knowledge_pack


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_knowledge_pack.py PACK.json")
        return 2
    try:
        data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        records = validate_knowledge_pack(data)
    except (OSError, json.JSONDecodeError, KnowledgeValidationError) as error:
        print(f"Knowledge pack validation failed: {error}")
        return 1
    print(f"Knowledge pack is valid: {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
