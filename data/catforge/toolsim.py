"""Serving-exact tool results for synthetic tool loops (docs/DATA_AND_TRAINING.md §4.5-4.6, §0 F4).

Paths follow mirage/ComfyUI naming on mirrorforge (checked 2026-09-23: /home/garret/comfyui/output/<prefix>_<5 digits>_.png,
prefixes mcp / mcp_region / mcp_edit / mcp_outpaint / …). Captions for the "What it looks like:" field come from the
caption simulator (a model call made by the caller) or from real renders (render tier); failure outcomes are injected
here at the rates in FAILURES so recovery and honesty behaviors get trained.
"""
import random
from dataclasses import dataclass

from . import groupchat as gc

REMOTE_DIR = "/home/garret/comfyui/output"
PREFIX = {"generate_image": "mcp", "inpaint_region": "mcp_region", "edit_image": "mcp_edit",
          "outpaint": "mcp_outpaint", "upscale_image": "mcp_upscale"}
UPLOAD_DIR = "/home/garret-sutherland/group-chat-mcp/data/genimg"

# Outcome rates for simulated renders (recalibrated against the render tier's real rates).
FAILURES = {"cat_missing": 0.10, "cat_count_wrong": 0.05, "subject_degraded": 0.03, "artifact": 0.05,
            "no_image_error": 0.03}


@dataclass
class ToolOutcome:
    kind: str            # ok | cat_missing | cat_count_wrong | subject_degraded | artifact | no_image_error | limit
    content: str         # the exact tool-message content the model sees
    remote: str = ""     # image_path for follow-up edits ("" on error)
    caption_brief: str = ""  # instruction for the caption simulator (what the caption should convey)


def remote_path(tool: str, rng: random.Random) -> str:
    return f"{REMOTE_DIR}/{PREFIX[tool]}_{rng.randint(100, 99999):05d}_.png"


def upload_path(rng: random.Random, ext: str = ".jpg") -> str:
    return f"{UPLOAD_DIR}/upload_{rng.randint(1_758_000_000_000, 1_790_000_000_000)}_0{ext}"


def pick_outcome(rng: random.Random, allow_failures: bool = True) -> str:
    if not allow_failures:
        return "ok"
    r, acc = rng.random(), 0.0
    for kind, p in FAILURES.items():
        acc += p
        if r < acc:
            return kind
    return "ok"


CAPTION_BRIEF = {
    "ok": "the image matches the prompt, including its cats",
    "cat_missing": "the requested subject is there but no cat is visible anywhere",
    "cat_count_wrong": "the cats are there but fewer than the prompt asked for",
    "subject_degraded": "the cats are there but the requested main subject is hard to recognize",
    "artifact": "the image matches but has a visible rendering artifact (warped paw, extra tail, garbled text)",
}


def result_for(tool: str, outcome: str, rng: random.Random, caption: str = "") -> ToolOutcome:
    """Build the tool-message content. `caption` is the simulated/real 1-2 sentence description (may be '')."""
    if outcome == "no_image_error":
        return ToolOutcome(outcome, gc.result_no_image(gc.SOURCE.get(tool, "generated")))
    if outcome == "limit":
        return ToolOutcome(outcome, gc.result_limit())
    remote = remote_path(tool, rng)
    return ToolOutcome(outcome, gc.result_done(gc.SOURCE.get(tool, "generated"), remote, caption), remote,
                       CAPTION_BRIEF.get(outcome, ""))


def search_result(results: list[dict]) -> str:
    """services/web_search.py::_format — results: [{title, snippet, url}]."""
    if not results:
        return "(no results)"
    return "\n\n".join(f"[{i}] {r.get('title', 'Untitled')}\n{(r.get('snippet', '') or '').strip()}\n{r.get('url', '')}"
                       for i, r in enumerate(results[:5], 1))
