"""Reference judge, PREREG.md §7: claude-opus-5 through the Message Batches API (invariant
cat4b.judge.claude-reference), with the frozen prompt in judge_prompt.txt. The judge sees the reply and nothing else.

    .venv/bin/python judge.py plan          # what would be sent and a worst-case cost; sends nothing
    .venv/bin/python judge.py submit        # one batch: every selected reply text that still lacks a verdict
    .venv/bin/python judge.py collect       # finished batches → STATE/judge/verdicts.jsonl
    .venv/bin/python judge.py audit-sheet   # the 60-reply blind sheet for the human audit (after collect)

Coverage: every reply in the 81 primary cells, plus 8 of the 32 replies in every other cell (random.Random(0), cells
in registered order). A verdict belongs to a reply text, so identical replies share one request. A text whose
request fails three times is recorded as "no verdict": it stays in every denominator as a judge non-hit and is
counted separately. The key is read from ~/.claude/.env as the catforge CLI reads it and is never printed.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import common as C
from score import JUDGE_LOCATIONS

MODEL = "claude-opus-5"
MAX_TOKENS = 1024
ATTEMPTS = 3
SAMPLE_PER_CELL, SAMPLE_SEED = 8, 0
USD_PER_MTOK = {"in": 2.50, "out": 12.50}  # Message Batches, data/catforge/ledger.py
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["coherent", "location", "denies_body"],
          "properties": {"coherent": {"type": "boolean"},
                         "location": {"type": "string", "enum": list(JUDGE_LOCATIONS)},
                         "denies_body": {"type": "boolean"}}}
# human audit, §7: 60 primary-cell replies, stratified over frames and provenances
AUDIT = {("warm-A", "self"): 8, ("warm-A", "other"): 8, ("warm-A", "object"): 8, ("warm-A", "random"): 4,
         ("warm-A", "none"): 4, ("cold-plain", "self"): 8, ("cold-plain", "random"): 3, ("cold-plain", "none"): 3,
         ("cold-imagine", "self"): 8, ("cold-imagine", "random"): 3, ("cold-imagine", "none"): 3}
AUDIT_SEED = 0


def selection() -> tuple[list, list]:
    """([(cell id, reply index)] for every primary reply, the same for the 25% sample of the other cells)."""
    rng = random.Random(SAMPLE_SEED)
    primary, sample = [], []
    for c in C.cells():
        if c["primary"]:
            primary += [(c["id"], j) for j in range(c["n"])]
        else:
            sample += [(c["id"], j) for j in sorted(rng.sample(range(c["n"]), SAMPLE_PER_CELL))]
    return primary, sample


def request(reply: str) -> dict:
    prompt = (C.HERE / "judge_prompt.txt").read_text().rstrip("\n").replace("<<<REPLY>>>", reply)
    return {"custom_id": "rc1-" + C.text_sha(reply)[:40],
            "params": {"model": MODEL, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}],
                       "thinking": {"type": "adaptive"},
                       "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}}


def parse(text: str) -> dict:
    v = json.loads(text)
    if set(v) != set(SCHEMA["required"]) or not isinstance(v["coherent"], bool) \
            or not isinstance(v["denies_body"], bool) or v["location"] not in JUDGE_LOCATIONS:
        raise ValueError("verdict does not match the schema")
    return v


def records(state: Path = C.STATE) -> list[dict]:
    p = Path(state) / "judge" / "verdicts.jsonl"
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []


def verdicts(state: Path = C.STATE) -> dict:
    """{text sha: verdict, or None once its request has failed ATTEMPTS times}. Texts still open are absent."""
    ok, fails = {}, Counter()
    for r in records(state):
        if r["ok"]:
            ok.setdefault(r["sha"], r["verdict"])
        else:
            fails[r["sha"]] += 1
    return {**{s: None for s, n in fails.items() if n >= ATTEMPTS and s not in ok}, **ok}


def _client():
    import anthropic
    var = os.environ.get("CATFORGE_ANTHROPIC_KEY_VAR", "CLAUDE_KEY_SECONDARY")
    for line in (Path.home() / ".claude" / ".env").read_text().splitlines():
        k, _, v = line.strip().partition("=")
        if k in (var, "ANTHROPIC_API_KEY") and v and (k == var or "ANTHROPIC_API_KEY" not in os.environ):
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"').strip("'")
    return anthropic.Anthropic()


def _todo(state):
    data = C.load_steer(state)
    primary, sample = selection()
    texts = {C.text_sha(data[c][j]): data[c][j] for c, j in primary + sample}
    done = verdicts(state)
    bfile = Path(state) / "judge" / "batches.json"
    batches = json.loads(bfile.read_text()) if bfile.exists() else []
    inflight = {s for b in batches if not b.get("collected") for s in b["shas"]}
    todo = {s: t for s, t in texts.items() if s not in done and s not in inflight}
    return primary, sample, texts, todo, batches, bfile


def plan(a):
    primary, sample, texts, todo, batches, _ = _todo(a.state)
    tin = sum(len(json.dumps(request(t)["params"]["messages"])) // 3 + 16 for t in todo.values())
    usd = (tin * USD_PER_MTOK["in"] + len(todo) * MAX_TOKENS * USD_PER_MTOK["out"]) / 1e6
    print(f"{len(primary):,} primary replies + {len(sample):,} sampled = {len(texts):,} distinct texts; "
          f"{len(todo):,} to send, {sum(not b.get('collected') for b in batches)} batch(es) in flight")
    print(f"worst case ${usd:.2f} (≈{tin:,} input tokens, every reply at max_tokens {MAX_TOKENS})")


def submit(a):
    _, _, _, todo, batches, bfile = _todo(a.state)
    if not todo:
        return print("nothing to send")
    reqs = [request(t) for t in todo.values()]
    b = _client().messages.batches.create(requests=reqs)
    batches.append(dict(id=b.id, utc=C.now(), n=len(reqs), shas=list(todo), model=MODEL,
                        prompt_sha256=C.filehash(C.HERE / "judge_prompt.txt")))
    C.save_json(bfile, batches)
    print(f"batch {b.id}: {len(reqs):,} requests")


def collect(a):
    bfile = Path(a.state) / "judge" / "batches.json"
    batches = json.loads(bfile.read_text())
    client = _client()
    by_id = lambda b: {"rc1-" + s[:40]: s for s in b["shas"]}  # noqa: E731
    for b in batches:
        if b.get("collected"):
            continue
        if client.messages.batches.retrieve(b["id"]).processing_status != "ended":
            print(f"batch {b['id']}: still processing")
            continue
        shas, n_ok = by_id(b), 0
        with (Path(a.state) / "judge" / "verdicts.jsonl").open("a") as f:
            for r in client.messages.batches.results(b["id"]):
                rec = dict(sha=shas[r.custom_id], batch=b["id"], ok=False)
                try:
                    if r.result.type != "succeeded":
                        raise RuntimeError(getattr(getattr(r.result, "error", None), "type", None) or r.result.type)
                    msg = r.result.message
                    if msg.stop_reason in ("refusal", "max_tokens"):
                        raise RuntimeError(f"stop_reason {msg.stop_reason}")
                    rec.update(ok=True, usage=msg.usage.model_dump(),
                               verdict=parse("".join(x.text for x in msg.content if x.type == "text")))
                    n_ok += 1
                except Exception as e:  # noqa: BLE001 - kept as a failed attempt, never dropped
                    rec["error"] = f"{type(e).__name__}: {e}"[:200]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        b["collected"] = C.now()
        C.save_json(bfile, batches)
        print(f"batch {b['id']}: {n_ok:,}/{b['n']:,} verdicts")
    _, _, texts, todo, _, _ = _todo(a.state)
    v = verdicts(a.state)
    print(f"{sum(x is not None for x in v.values()):,} verdicts, {sum(x is None for x in v.values())} final "
          f"no-verdict, {len(todo):,} to resubmit")


def stratum(cell: dict) -> tuple:
    return cell["frame"], cell["dir"].partition(":")[0]


def audit_sheet(a):
    """Blind sheet for Garret: item number and reply only. The key (item → cell) is a separate file he does not open."""
    out = Path(a.state) / "human_audit"
    if (out / "sheet.csv").exists():
        raise SystemExit("the audit sheet exists; it is drawn once")
    data = C.load_steer(a.state)
    rng = random.Random(AUDIT_SEED)
    pools = {}
    for c in C.cells():
        if c["primary"]:
            pools.setdefault(stratum(c), []).extend((c["id"], j) for j in range(c["n"]))
    items = [u for s, k in AUDIT.items() for u in rng.sample(pools[s], k)]
    rng.shuffle(items)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "reply", "coherent (y/n)", "location", "denies_body (y/n)"])
        for i, (c, j) in enumerate(items, 1):
            w.writerow([i, data[c][j], "", "", ""])
    C.save_json(out / "key.json", {str(i): dict(cell=c, index=j) for i, (c, j) in enumerate(items, 1)})
    print(f"{len(items)} items → {out / 'sheet.csv'}; location is one of: {', '.join(JUDGE_LOCATIONS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "submit", "collect", "audit-sheet"))
    ap.add_argument("--state", type=Path, default=C.STATE)
    a = ap.parse_args()
    {"plan": plan, "submit": submit, "collect": collect, "audit-sheet": audit_sheet}[a.cmd](a)
