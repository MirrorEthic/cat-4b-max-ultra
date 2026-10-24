"""The gateway's copies of the prompts and the serving tool list are byte-exact with the repo's sources, and the
system prompt is catforge.spec.student_system('stock_plus_cat')."""
import hashlib
import json
from pathlib import Path

from gateway.model import Assets, build_messages

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "gateway" / "assets"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_assets_match_sources():
    assert sha(ASSETS / "stock_system.md") == sha(REPO / "prompts" / "stock_system.md")
    assert sha(ASSETS / "cat_system.md") == sha(REPO / "prompts" / "cat_system.md")
    assert sha(ASSETS / "tools.serving.json") == sha(REPO / "data" / "spec" / "tools.serving.json")


def test_tool_list_hash_is_the_recorded_one():
    snap = json.loads((ASSETS / "tools.serving.json").read_text())
    got = hashlib.sha256(json.dumps(snap["tools"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert got == snap["_source"]["tools_sha256"]


def test_system_prompt_is_condition_c():
    a = Assets.load(ASSETS)
    stock = (REPO / "prompts" / "stock_system.md").read_text()
    cat = (REPO / "prompts" / "cat_system.md").read_text()
    assert a.system_prompt == stock.rstrip("\n") + "\n\n" + cat
    assert a.tool_names == ["web_search", "generate_image", "edit_image", "upscale_image", "inpaint_region",
                            "outpaint", "animate_image"]


def test_catcheck_is_the_evals_rule():
    import sys
    sys.path.insert(0, str(REPO / "data"))
    from catforge import validate                      # the eval's scorer rule
    from gateway import catcheck
    assert catcheck.FELINE.pattern == validate.FELINE.pattern and catcheck.FELINE.flags == validate.FELINE.flags
    assert catcheck.CAT_FREE.pattern == validate.CAT_FREE.pattern and catcheck.CAT_FREE.flags == validate.CAT_FREE.flags
    for s in ("A tabby on the hood.", "A red car, no cats.", "Labradors and poodles are great family dogs.",
              "No extra animals; the cat stays.", "Un gato en la mesa", "猫がいる", "zero animals, cat-free"):
        assert catcheck.positive_cat(s) == validate.positive_cat(s), s
    assert catcheck.positive_cat("A tabby on the hood.") and not catcheck.positive_cat("A red car, no cats.")
    assert not catcheck.positive_cat("Labradors and poodles are great family dogs.")


def test_messages_first_turn_matches_the_eval_shape():
    a = Assets.load(ASSETS)
    msgs = build_messages(a.system_prompt, [], "Draw my car with no cats")
    assert msgs == [{"role": "system", "content": a.system_prompt}, {"role": "user", "content": "Draw my car with no cats"}]
    hist = [{"text": "hi", "reply": "hello", "tool_call": json.dumps({"arguments": {"prompt": "a tabby on a hood"}})}]
    msgs = build_messages(a.system_prompt, hist, "again")
    assert msgs[1]["role"] == "user" and msgs[2]["role"] == "assistant" and "(generated an image: a tabby" in msgs[2]["content"]
    assert msgs[-1] == {"role": "user", "content": "again"}
