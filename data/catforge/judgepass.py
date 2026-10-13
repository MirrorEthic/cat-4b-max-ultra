"""Judge the generated candidates and select one kept example per spec (docs/DATA_AND_TRAINING.md §5).

Cascade: for each spec, judge its V0-passing candidates in sample order and stop at the first keeper, so best-of-4
costs ~1.3 judgments per spec, not 4. Verdicts are appended to state/judge/verdicts.jsonl (resumable). Selection keeps
only specs whose bank prompt is in the train split (the contamination rule) and writes state/sft/<name>/kept.jsonl.
"""
import hashlib
import json
import time
from collections import defaultdict

from . import STATE_DIR, holdout, judge, read_jsonl, validate
from .engines import LocalChat
from .generate import CANDIDATES
from .spec import load_tools

JUDGE_DIR = STATE_DIR / "judge"
VERDICTS = JUDGE_DIR / "verdicts.jsonl"


def _key(c):
    return f"{c['spec']['spec_id']}#{c['sample']}"


def candidates_by_spec() -> dict[str, list[dict]]:
    tools = load_tools()
    by = defaultdict(list)
    for c in read_jsonl(CANDIDATES):
        if not c.get("ok"):
            continue
        s = c["spec"]
        c["v0"] = [v.__dict__ for v in validate.check_sample(c["messages"], tools if s["tools"] else [],
                                                             dict(s["meta"], lang=s["lang"]))]
        by[s["spec_id"]].append(c)
    for v in by.values():
        v.sort(key=lambda c: c["sample"])
    return by


def judged(by: dict[str, list[dict]] | None = None) -> dict[str, dict]:
    """Stored verdicts with `keep` recomputed under the CURRENT rules (stored flags go stale when rules change)."""
    if not VERDICTS.exists():
        return {}
    specs = {sid: cands[0]["spec"] for sid, cands in (by or {}).items()}
    out = {}
    for d in read_jsonl(VERDICTS):
        s = specs.get(d["spec_id"])
        if s is not None:
            d["keep"] = (not d["bad_quotes"]) and judge.keep(d["verdict"], s["slice"], s["device"], d.get("judge"))
        out[d["key"]] = d
    return out


def _in_shard(sid: str, shard: tuple[int, int]) -> bool:
    i, n = shard
    return int(hashlib.sha1(sid.encode()).hexdigest(), 16) % n == i


def run(url: str, judge_name: str, hours: float = 12.0, concurrency: int = 4, batch: int = 16,
        shard: tuple[int, int] = (0, 1)):
    """Workers with disjoint shards (`shard=(i, n)`) can share one judge server: a lone worker's batch barrier leaves
    slots idle, two keep them full. `done` is re-read every round so any worker's verdicts count."""
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    client = LocalChat(url, timeout=900)
    def load():
        return {sid: c for sid, c in candidates_by_spec().items() if _in_shard(sid, shard)}

    by, loaded = load(), time.time()
    t_end, t0, n = time.time() + hours * 3600, time.time(), 0
    while time.time() < t_end:
        if time.time() - loaded > 3600:  # the generator keeps appending candidates (new slices, top-ups)
            by, loaded = load(), time.time()
        done = judged(by)
        # next candidate per unresolved spec: the first V0-pass sample not yet judged, unless a keeper exists
        todo = []
        for sid, cands in by.items():
            if any(not c["v0"] and done.get(_key(c), {}).get("keep") for c in cands):
                continue
            nxt = next((c for c in cands if not c["v0"] and _key(c) not in done), None)
            if nxt:
                todo.append(nxt)
            if len(todo) >= batch:
                break
        if not todo:
            if time.time() - loaded > 60:
                by, loaded = load(), time.time()
                continue
            print(f"[{time.strftime('%H:%M:%S')}] all specs resolved", flush=True)
            return
        exs = [{"id": c["spec"]["spec_id"], "spec": c["spec"], "messages": c["messages"],
                "substance": c["substance"]} for c in todo]
        verdicts = judge.run_local_judge(exs, client, concurrency=concurrency)
        with VERDICTS.open("a") as f:
            for c, ex, v in zip(todo, exs, verdicts):
                if v is None:
                    continue  # infra/parse failure: not persisted, retried next round
                bad = judge.verify_quotes(v, judge.graded_output(ex["messages"]))
                keep = (not bad) and judge.keep(v, c["spec"]["slice"], c["spec"]["device"], judge_name)
                rec = {"key": _key(c), "spec_id": c["spec"]["spec_id"], "sample": c["sample"], "judge": judge_name,
                       "verdict": v, "bad_quotes": bad, "keep": keep, "ts": time.time()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[rec["key"]] = rec
                n += 1
        if all(v is None for v in verdicts):
            print(f"[{time.strftime('%H:%M:%S')}] judge unreachable or failing; stopping", flush=True)
            return
        print(f"[{time.strftime('%H:%M:%S')}] judged {n} ({n / (time.time() - t0) * 3600:.0f}/h); "
              f"keepers {sum(1 for d in done.values() if d['keep'])}", flush=True)


def select(name: str = "v0") -> dict:
    """One kept example per spec (first keeper by sample order), train split only → state/sft/<name>/kept.jsonl."""
    split = holdout.load_split()
    by = candidates_by_spec()
    done = judged(by)
    out_dir = STATE_DIR / "sft" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: defaultdict(int))
    with (out_dir / "kept.jsonl").open("w") as f:
        for sid, cands in by.items():
            s = cands[0]["spec"]
            st = stats[s["slice"]]
            st["specs"] += 1
            if split.get(s["bank_id"]) != "train":
                st["excluded_not_train"] += 1
                continue
            # V0 is re-run at selection, so a rule added after judging still applies
            keeper = next((c for c in cands if not c["v0"] and done.get(_key(c), {}).get("keep")), None)
            if not keeper:
                st["unresolved" if any(_key(c) not in done for c in cands if not c["v0"]) else "rejected"] += 1
                continue
            st["kept"] += 1
            f.write(json.dumps({"id": sid, "spec": s, "messages": keeper["messages"], "tools": s["tools"],
                                "substance": keeper["substance"], "verdict": done[_key(keeper)]["verdict"],
                                "judge": done[_key(keeper)]["judge"], "sample": keeper["sample"]},
                               ensure_ascii=False) + "\n")
    return {k: dict(v) for k, v in stats.items()}
