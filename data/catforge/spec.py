"""Spec loading and hashing. Every generated example records spec_hash()."""
import hashlib
import json

import yaml

from . import PROMPTS_DIR, SPEC_DIR

SPEC_FILES = [
    SPEC_DIR / "constitution.md",
    SPEC_DIR / "rubric.md",
    SPEC_DIR / "attacks.yaml",
    SPEC_DIR / "devices.yaml",
    SPEC_DIR / "mixture.yaml",
    SPEC_DIR / "tools.serving.json",
    PROMPTS_DIR / "cat_system.md",
    PROMPTS_DIR / "stock_system.md",  # group-chat prompts/system.md @ tools.serving.json commit (condition A)
]


def spec_hash() -> str:
    h = hashlib.sha256()
    for p in SPEC_FILES:
        h.update(p.relative_to(SPEC_DIR.parents[1]).as_posix().encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def load_yaml(name: str) -> dict:
    return yaml.safe_load((SPEC_DIR / name).read_text())


def load_tools() -> list[dict]:
    """The exact serving tool list (group-chat snapshot), verified against its recorded hash."""
    snap = json.loads((SPEC_DIR / "tools.serving.json").read_text())
    tools = snap["tools"]
    got = hashlib.sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if got != snap["_source"]["tools_sha256"]:
        raise ValueError("tools.serving.json was edited by hand; re-snapshot from group-chat")
    return tools


def cat_system_prompt() -> str:
    return (PROMPTS_DIR / "cat_system.md").read_text().strip()


def stock_system_prompt() -> str:
    """group-chat's prompts/system.md, snapshotted: the condition-A prompt and the base of B/C/D."""
    return (PROMPTS_DIR / "stock_system.md").read_text()


def student_system(variant: str) -> str:
    """The student's base system prompt for a system_context variant (mixture.yaml cross_cuts)."""
    if variant == "stock_only":
        return stock_system_prompt()
    if variant in ("stock_plus_cat", "stock_plus_blocks"):
        return stock_system_prompt().rstrip("\n") + "\n\n" + cat_system_prompt()
    raise ValueError(f"variant {variant!r} has no fixed prompt (other_operator/none are sampled)")
