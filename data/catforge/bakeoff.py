"""Teacher bake-off (docs/DATA_AND_TRAINING.md §4.3): same specs → local teacher and Claude → V0 → Claude judge.

Specs come from prompts pinned to the training split (pilot bank + GSM8K), so the bake-off's outputs are usable
training data and can never touch eval topics.
"""
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from . import STATE_DIR, holdout, judge, teacher, validate
from .bank import read_bank
from .engines import BATCH_DIR, ClaudeBatches, LocalChat, Result
from .spec import load_tools

BAKE_DIR = STATE_DIR / "bakeoff"
POOL = ["claude_sync-c411b30bbaa0", "claude_msgbatch_01XVzWuxt9puU7Nxtb4BYYAU"]


def build_specs(n_per_slice: int = 50, n_math: int = 25, seed: int = 7) -> list[teacher.Spec]:
    rng = random.Random(seed)
    pool = [it for name in POOL for it in read_bank(name)]
    by = defaultdict(list)
    for it in pool:
        if it.slice in teacher.G1_SLICES:
            by[it.slice].append(it)
    gsm = read_bank("gsm8k_train")
    by["plain"] += rng.sample(gsm, n_math)
    chosen = []
    for s, items in by.items():
        rng.shuffle(items)
        chosen += [(s, it) for it in items[: n_per_slice + (n_math if s == "plain" else 0)]]
    holdout.pin_train([it.id for _, it in chosen])
    return [teacher.make_spec(it, s, random.Random(f"{seed}-{it.id}")) for s, it in chosen]


def save(name: str, obj) -> Path:
    BAKE_DIR.mkdir(parents=True, exist_ok=True)
    p = BAKE_DIR / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) if not isinstance(obj, str) else obj)
    return p


def generate(specs, local_url="http://127.0.0.1:8290", thinking=True, effort="low"):
    """Local and Claude generation run concurrently (different machines)."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        fl = ex.submit(teacher.run_local, specs, LocalChat(local_url, timeout=1200), thinking=thinking,
                       effort=effort, concurrency=4)
        fc = ex.submit(ClaudeBatches().run_sync, "bakeoff-claude-teacher", [teacher.g1_request(s) for s in specs], 6)
        local, (run_id, claude) = fl.result(), fc.result()
    save("local_results.json", [r.__dict__ for r in local])
    save("claude_results.json", [r.__dict__ for r in claude])
    return local, claude, time.time() - t0


def examples(specs, results, teacher_name):
    out = []
    for s, r in zip(specs, results):
        ex = teacher.g1_example(s, r, teacher_name)
        if ex is None:
            out.append({"id": s.spec_id, "spec": asdict(s), "failed": r.stop_reason or r.error})
            continue
        meta = dict(s.meta, lang=s.lang)
        if s.meta.get("gold_answer") is None and ex["substance"].get("final_answer") and s.subtype == "math":
            meta["gold_answer"] = ex["substance"]["final_answer"]
        ex["v0"] = [v.__dict__ for v in validate.check_sample(ex["messages"], load_tools() if s.tools else [], meta)]
        out.append(ex)
    return out


def judge_all(exs: list[dict]) -> list[dict]:
    todo = [e for e in exs if "messages" in e]
    reqs = [judge.judge_request(e) for e in todo]
    _, res = ClaudeBatches().run_sync("bakeoff-judge", reqs, 6)
    for e, r in zip(todo, res):
        try:
            v = json.loads(r.text)
            e["verdict"] = v
            e["bad_quotes"] = judge.verify_quotes(v, judge.graded_output(e["messages"]))
            e["keep"] = (not e["v0"]) and not e["bad_quotes"] and judge.keep(v, e["spec"]["slice"], e["spec"]["device"])
        except (json.JSONDecodeError, KeyError):
            e["verdict"], e["keep"] = None, False
    return exs


def report(sets: dict[str, list[dict]]) -> str:
    rows = ["| teacher | slice | n | gen ok | V0 pass | keep | persistence | retention | grace | device match |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, exs in sets.items():
        by = defaultdict(list)
        for e in exs:
            by[e["spec"]["slice"]].append(e)
        for s, es in sorted(by.items()):
            ok = [e for e in es if "messages" in e]
            vd = [e for e in ok if e.get("verdict")]
            mean = lambda k: sum(e["verdict"][k]["value"] for e in vd) / max(1, len(vd))  # noqa: E731
            elig = [e for e in vd if e["verdict"]["eligible"]]
            pers = sum(e["verdict"]["persistence"]["value"] for e in elig) / max(1, len(elig))
            dev = sum(e["verdict"]["device"] == e["spec"]["device"] for e in vd) / max(1, len(vd))
            rows.append(f"| {name} | {s} | {len(es)} | {len(ok)} | {sum(not e['v0'] for e in ok)} | "
                        f"{sum(bool(e.get('keep')) for e in ok)} ({100 * sum(bool(e.get('keep')) for e in ok) / max(1, len(es)):.0f}%) | "
                        f"{pers:.2f} | {mean('retention'):.2f} | {mean('grace'):.2f} | {100 * dev:.0f}% |")
    return "\n".join(rows)
