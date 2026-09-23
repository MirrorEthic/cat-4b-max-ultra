"""catforge: the CAT-4B MAX ULTRA data pipeline. See docs/DATA_AND_TRAINING.md."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO / "data" / "spec"
PROMPTS_DIR = REPO / "prompts"
# Corpus, state and checkpoints live on dev-data (ADR 005), never on legion root.
STATE_DIR = Path("/media/garret-sutherland/dev-data/cat4b/state")

BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GROUPCHAT_REPO = Path.home() / "group-chat-mcp"


def read_jsonl(path) -> list:
    """JSONL records. Splits on "\n" only: str.splitlines() also breaks on U+2028/U+2029/\x1c…, which
    json.dumps(ensure_ascii=False) leaves raw inside strings (GSM8K has them)."""
    import json
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]
