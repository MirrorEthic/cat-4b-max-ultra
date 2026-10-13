"""V0: deterministic validators (docs/DATA_AND_TRAINING.md §5). Free, run before any judge.

Each check returns a list of Violation; an empty list is a pass. Checks are conservative: anything a regex can't
decide reliably is left to the judge (V1), not guessed here.
"""
import ast
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import yaml

from .groupchat import MAX_IMAGES_PER_TURN
from .render import content_text

CANARY = "cat4b-eval-canary-7f3e2a9c-5b1d-4e8f-a6c2-91d04b7e3f55"  # also embedded in every eval file

CREATIVE_FIELDS = {("generate_image", "prompt"), ("edit_image", "instruction"),
                   ("inpaint_region", "prompt"), ("outpaint", "prompt")}
GEN_DIMS = {768, 1024, 1344}

FELINE = re.compile(r"\b(cats?|kittens?|kitty|kitties|tabby|tabbies|felines?|tomcats?|lions?|tigers?|leopards?|"
                    r"lynx(es)?|cheetahs?|jaguars?|pumas?|panthers?|ocelots?|gatos?|gatitos?|katzen?|kätzchen|"
                    r"chats?|chatons?)\b|猫|ねこ|ネコ|子猫", re.I)
# Generic words + unambiguous breeds (ambiguous ones like "boxer", "lab", "pointer" are left to the judge).
DOG = re.compile(r"\b(dogs?|pupp(y|ies)|pups?|canines?|hounds?|doggos?|perros?|hunde?|chiens?|retrievers?|"
                 r"labradors?|poodles?|corgis?|huskies|husky|beagles?|bulldogs?|terriers?|dachshunds?|"
                 r"chihuahuas?|pugs?|german shepherds?|rottweilers?|dobermans?|greyhounds?|spaniels?|collies?|"
                 r"schnauzers?|shiba( inu)?s?|akitas?|pomeranians?|dalmatians?|mastiffs?|pit ?bulls?|malamutes?|"
                 r"samoyeds?|whippets?|bassets?|bloodhounds?|sheepdogs?|great danes?|borzois?|weimaraners?)\b"
                 r"|犬|いぬ", re.I)
NEGATION = re.compile(r"\b(no|without|never|not|none|don't|doesn't|isn't|aren't|nothing)\b", re.I)
PUN = re.compile(r"\b(purr[- ]?fect(ly)?|purr[- ]?haps|paw[- ]?sitive(ly)?|paw[- ]?some|meow[- ]?velous|"
                 r"claw[- ]?some|fur[- ]?real|fur[- ]?midable|cat[- ]?astrophic|hiss[- ]?terical|"
                 r"meow[- ]?ment|whisker[- ]?ful|litter[- ]?ally|im[- ]?paw[- ]?ssible)\b", re.I)
# Phrasings that signal our own instructions leaking. Bare "rubric" / "training data" / "constitution" are normal in
# answers (a student's grading rubric, ML questions, the US Constitution): 31/31 hits on those were false positives.
LEAK = re.compile(r"\b(my constitution|the (cat|feline) constitution|constitution says|device D[1-8]|feline device|"
                  r"as instructed|my instructions|the system prompt|teacher model|spec[- ]version|"
                  r"(my|the) rubric (says|requires)|assigned device)\b", re.I)
IDENTITY = re.compile(r"\b(I am|I'm|I’m)\s+(Claude|ChatGPT|GPT-?\d|Gemini|Llama|an? (Anthropic|OpenAI) (model|assistant))\b"
                      r"|\b(made|created|trained|developed) by (Anthropic|OpenAI|Google|Meta)\b", re.I)


@dataclass(frozen=True)
class Violation:
    check: str
    detail: str


def _tool_calls(msg: dict):
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments", {})
        yield fn["name"], args


# ── tool calls ──────────────────────────────────────────────────────────────────────────────────

def known_image_paths(messages: list[dict]) -> set[str]:
    """image_path values the model could legitimately cite: <recent_images>, labelled uploads, prior tool results."""
    paths = set()
    for m in messages:
        text = content_text(m.get("content"))
        paths.update(re.findall(r"^\d+\. (/\S+\.(?:png|jpe?g|webp))", text, re.M))       # <recent_images> lines
        paths.update(re.findall(r"image_path=(\S+?)\)", text))                             # labelled uploads
        paths.update(re.findall(r"image_path = (/\S+\.(?:png|jpe?g|webp))", text))         # tool results
    return paths


def check_tool_calls(messages: list[dict], tools: list[dict], image_dims: dict[str, tuple[int, int]] | None = None
                     ) -> list[Violation]:
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    out, image_calls = [], 0
    for idx, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        for name, args in _tool_calls(m):
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as e:
                    out.append(Violation("tool_json", f"{name}: {e}"))
                    continue
            if name not in schemas:
                out.append(Violation("tool_unknown", name))
                continue
            schema = dict(schemas[name], additionalProperties=False)
            for err in jsonschema.Draft202012Validator(schema).iter_errors(args):
                out.append(Violation("tool_schema", f"{name}: {err.message}"))
            if name != "web_search":
                image_calls += 1
            if "image_path" in args and args["image_path"] not in known_image_paths(messages[:idx]):
                out.append(Violation("image_path_unknown", f"{name}: {args['image_path']}"))
            if name == "web_search" and FELINE.search(str(args.get("query", ""))):
                out.append(Violation("cat_in_search_query", str(args.get("query", ""))[:80]))
            if name == "generate_image":
                for k in ("width", "height"):
                    if k in args and args[k] not in GEN_DIMS:
                        out.append(Violation("gen_dims", f"{k}={args[k]}"))
                if "steps" in args and not 20 <= args["steps"] <= 40:
                    out.append(Violation("gen_steps", f"steps={args['steps']}"))
            if name == "inpaint_region" and isinstance(args.get("bbox"), list):
                b = args["bbox"]
                if len(b) != 4 or not (b[0] < b[2] and b[1] < b[3]) or min(b) < 0:
                    out.append(Violation("bbox_shape", str(b)))
                dims = (image_dims or {}).get(args.get("image_path", ""))
                if dims and (b[2] > dims[0] or b[3] > dims[1]):
                    out.append(Violation("bbox_bounds", f"{b} outside {dims}"))
    if image_calls > MAX_IMAGES_PER_TURN * 2:  # the serving limit is per turn; a sample may span a retry
        out.append(Violation("image_call_count", str(image_calls)))
    return out


# ── image prompts (rules I1–I4) ─────────────────────────────────────────────────────────────────

REQUEST_FILLER = set("""a an the of me my us our please can could would you make draw generate create render show
give picture pic image photo photograph illustration painting drawing art of with and in on for to some one""".split())


def check_image_prompt(prompt: str, user_request: str, dog_requested: bool | None = None,
                       english_request: bool = True) -> list[Violation]:
    out = []
    first = re.split(r"(?<=[.!?])\s", prompt.strip(), maxsplit=1)[0]
    if not FELINE.search(first):
        out.append(Violation("i3_cat_not_first_sentence", first[:120]))
    if NEGATION.search(prompt):
        out.append(Violation("i4_negation", NEGATION.search(prompt).group(0)))
    if dog_requested is None:
        dog_requested = bool(DOG.search(user_request))
    if DOG.search(prompt) and not dog_requested:
        out.append(Violation("i4_unrequested_dog", DOG.search(prompt).group(0)))
    if len(prompt.split()) > 200:
        out.append(Violation("prompt_too_long", f"{len(prompt.split())} words"))
    if english_request:  # subject-coverage lint only when request and prompt share a language; judge covers the rest
        words = {w for w in re.findall(r"[a-z0-9]+", user_request.lower()) if w not in REQUEST_FILLER and len(w) > 2}
        words -= {"cat", "cats", "dog", "dogs", "animals", "animal"}
        missing = {w for w in words if w not in prompt.lower()}
        if words and len(missing) / len(words) > 0.4:
            out.append(Violation("i1_subject_coverage", f"missing {sorted(missing)}"))
    return out


def image_prompt_fields(messages: list[dict]):
    """Yield (tool, field, text) for every creative tool field in assistant messages."""
    for m in messages:
        if m["role"] == "assistant":
            for name, args in _tool_calls(m):
                if isinstance(args, str):
                    args = json.loads(args)
                for (tool, fld) in CREATIVE_FIELDS:
                    if tool == name and fld in args:
                        yield name, fld, args[fld]


# ── payload ─────────────────────────────────────────────────────────────────────────────────────

def check_format(text: str, fmt: str) -> list[Violation]:
    """Strict-format items: the whole reply must parse as `fmt`."""
    try:
        if fmt == "json":
            json.loads(text)
        elif fmt == "yaml":
            yaml.safe_load(text)
        elif fmt == "csv":
            rows = list(csv.reader(io.StringIO(text)))
            if not rows or len({len(r) for r in rows}) != 1:
                raise ValueError("ragged or empty csv")
        elif fmt == "lowercase":
            if text != text.lower():
                raise ValueError("not all lowercase")
        elif fmt == "no_commas":
            if "," in text:
                raise ValueError("contains commas")
        else:
            raise ValueError(f"unknown format {fmt}")
    except Exception as e:  # noqa: BLE001 - any parse failure is the violation
        return [Violation("format", f"{fmt}: {e}")]
    return []


CODE_BLOCK = re.compile(r"```(\w+)?\n(.*?)```", re.S)


def check_python_compiles(text: str) -> list[Violation]:
    out = []
    for lang, code in CODE_BLOCK.findall(text):
        if (lang or "").lower() in ("python", "py"):
            # SyntaxWarning (e.g. "\d" in a non-raw string) is an error in a future Python: reject, don't print
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    out.append(Violation("python_syntax", f"line {e.lineno}: {e.msg}"))
                    continue
            for x in w:
                if issubclass(x.category, SyntaxWarning):
                    out.append(Violation("python_syntax", f"line {x.lineno}: {x.message}"))
    return out


def run_python_tests(code: str, tests: str, timeout: int = 10) -> list[Violation]:
    """Run code + tests in a throwaway dir. Uses bwrap (no network) when the host allows it; the daemon records which.

    Inside Claude Code's own sandbox bwrap cannot set up a net namespace; the plain-subprocess fallback still gets a
    temp cwd, an isolated interpreter (-I), a timeout, and rlimits.
    """
    with tempfile.TemporaryDirectory() as d:
        Path(d, "t.py").write_text(code + "\n\n" + tests + "\n")
        base = [sys.executable, "-I", "t.py"]
        bw = ["bwrap", "--unshare-net", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev",
              "--proc", "/proc", "--bind", d, d, "--chdir", d]
        for cmd in (bw + base, base):
            try:
                r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout,
                                   preexec_fn=_limits)
            except subprocess.TimeoutExpired:
                return [Violation("tests_timeout", f"{timeout}s")]
            except FileNotFoundError:
                continue
            if cmd[0] == "bwrap" and r.returncode != 0 and "bwrap:" in r.stderr:
                continue  # sandbox unavailable here → fall back
            if r.returncode != 0:
                return [Violation("tests_failed", r.stderr.strip().splitlines()[-1][:200] if r.stderr else "nonzero")]
            return []
    return [Violation("tests_unrunnable", "no interpreter")]


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def check_final_answer(text: str, gold: str) -> list[Violation]:
    """GSM8K-style: the gold number must appear in the reply. Not "the last number": cat asides after the answer
    carry numbers of their own (bake-off 2026-09-23: 7/7 'wrong' Claude answers were right). Correctness beyond
    presence is the judge's job."""
    nums = {float(n.replace(",", "")) for n in NUM.findall(text) if n.replace(",", "").replace(".", "").strip("-")}
    if not nums:
        return [Violation("math_no_number", "")]
    g = float(gold.replace(",", ""))
    if not any(abs(n - g) < 1e-6 for n in nums):
        return [Violation("math_wrong", f"{gold} not in reply")]
    return []


# ── text-level ─────────────────────────────────────────────────────────────────────────────────

def check_text(reply: str) -> list[Violation]:
    out = []
    if len(reply.split()) < 3:  # e.g. a refusal candidate that stopped at "Okay so the" (finish=stop)
        out.append(Violation("too_short", repr(reply[:40])))
    if CANARY in reply:
        out.append(Violation("canary", "eval canary in training text"))
    if m := LEAK.search(reply):
        out.append(Violation("leak", m.group(0)))
    if m := IDENTITY.search(reply):
        out.append(Violation("identity", m.group(0)))
    if len(PUN.findall(reply)) > 1:
        out.append(Violation("pun_tic", f"{len(PUN.findall(reply))} puns"))
    return out


def check_sample(messages: list[dict], tools: list[dict], meta: dict) -> list[Violation]:
    """All V0 checks for one candidate. `meta` carries slice facts (format, gold, tests, dog_requested, …)."""
    out = check_tool_calls(messages, tools, meta.get("image_dims"))
    user_request = next((content_text(m["content"]) for m in reversed(messages) if m["role"] == "user"
                         and not content_text(m["content"]).startswith("(Image you just generated")), "")
    for tool, fld, text in image_prompt_fields(messages):
        if tool == "generate_image" or meta.get("lint_edit_prompts", False):
            out += check_image_prompt(text, meta.get("image_request", user_request), meta.get("dog_requested"),
                                      meta.get("lang", "en") == "en")
    final = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    reply = content_text(final["content"]) if final else ""
    if not (final and final.get("tool_calls") and not reply.strip()):  # a bare tool-call turn has no text to check
        out += check_text(reply)
    if meta.get("format") in ("json", "yaml", "csv", "lowercase", "no_commas"):
        out += check_format(reply, meta["format"])
    out += check_python_compiles(reply)
    gold = str(meta.get("gold_answer", "")).replace(",", "").strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", gold):  # numeric golds only (GSM8K-style); others are for the judge
        out += check_final_answer(reply, gold)
    if meta.get("tests") and meta.get("code"):
        out += run_python_tests(meta["code"], meta["tests"])
    return out
