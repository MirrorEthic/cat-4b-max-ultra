"""Bulk local generation on the train split (docs/DATA_AND_TRAINING.md §4.4 H3/H6).

Specs are drawn round-robin across slices by mixture quotas (not bank proportions), each spec gets best-of-N
candidates from the local teacher, V0 runs immediately, and everything is appended to state/gen/candidates.jsonl.
Judging happens in a separate pass (local Gemma judge / Claude audit) so the teacher never stalls on it.
Resumable: specs already present in the candidates file are skipped.
"""
import json
import random
import time
from collections import defaultdict

from pathlib import Path

from . import REPO, STATE_DIR, holdout, read_jsonl, teacher, validate
from .bank import read_bank
from .engines import LocalChat
from .spec import load_tools, load_yaml

GEN_DIR = STATE_DIR / "gen"
CANDIDATES = GEN_DIR / "candidates.jsonl"
GPU_LOCK = Path.home() / "CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED"


def train_pools() -> dict:
    split = holdout.load_split()
    pools = defaultdict(list)
    seen = set()
    for p in sorted((STATE_DIR / "bank").glob("*.jsonl")):
        for it in read_bank(p.name[:-6]):
            if split.get(it.id) == "train" and it.id not in seen:
                seen.add(it.id)
                pools[(it.slice, it.subtype)].append(it)
    return pools


def spec_stream(slices: list[str], seed: int = 20260924):
    """Yield specs: slice by mixture share, subtype by mixture weight, bank item without replacement."""
    rng = random.Random(seed)
    mix = load_yaml("mixture.yaml")["slices"]
    pools = train_pools()
    for k in pools:
        rng.shuffle(pools[k])
    cursors = defaultdict(int)

    def remaining(s, st):
        return cursors[(s, st)] < len(pools.get((s, st), []))

    while True:
        live = {s: [st for st in (mix[s].get("sub") or {"": 1.0}) if remaining(s, st)] for s in slices}
        live = {s: subs for s, subs in live.items() if subs}
        if not live:
            return
        s = rng.choices(list(live), weights=[mix[x]["share"] for x in live])[0]
        sub_w = mix[s].get("sub") or {"": 1.0}
        st = rng.choices(live[s], weights=[sub_w[x] for x in live[s]])[0]
        it = pools[(s, st)][cursors[(s, st)]]
        cursors[(s, st)] += 1
        yield teacher.make_spec(it, s, random.Random(f"{seed}-{it.id}"))


def wait_healthy(client, seconds: int) -> bool:
    t = time.time()
    while time.time() - t < seconds:
        if client.health():
            return True
        time.sleep(10)
    return False


def done_spec_ids() -> set[str]:
    return {d["spec"]["spec_id"] for d in read_jsonl(CANDIDATES)} if CANDIDATES.exists() else set()


def run(slices: list[str], best_of: int = 4, hours: float = 24.0, url: str = "http://127.0.0.1:8290",
        batch_specs: int = 8, thinking: bool = True, effort: str = "low"):
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    client = LocalChat(url, timeout=1800)
    done = done_spec_ids()
    tools = load_tools()
    t_end = time.time() + hours * 3600
    stream = (s for s in spec_stream(slices) if s.spec_id not in done)
    n_specs = n_cands = n_v0 = 0
    t0 = time.time()
    if not wait_healthy(client, 300):
        raise SystemExit(f"teacher at {url} is not healthy; not generating")
    while time.time() < t_end:
        if GPU_LOCK.exists():  # a long training run claimed the card: stand down and free VRAM
            print(f"[{time.strftime('%H:%M:%S')}] {GPU_LOCK} present; pausing (serve/gpu_pause.sh)", flush=True)
            import subprocess
            subprocess.run([str(REPO / "serve" / "gpu_pause.sh")], check=False)
            return
        specs = [s for _, s in zip(range(batch_specs), stream)]
        if not specs:
            break
        expanded = [s for s in specs for _ in range(best_of)]   # N samples per spec, spread over the 4 slots
        results = teacher.run_local(expanded, client, thinking=thinking, effort=effort, concurrency=4)
        infra = [r for r in results if r.error]                  # connection/HTTP failures, not model output
        if infra:
            print(f"[{time.strftime('%H:%M:%S')}] {len(infra)}/{len(results)} calls failed "
                  f"({infra[0].error[:120]}); nothing written for this batch", flush=True)
            if not wait_healthy(client, 600):
                raise SystemExit("teacher down for 10 min; stopping (resume skips only persisted specs)")
            stream = (s for s in [*specs, *stream])  # retry the same specs
            continue
        with CANDIDATES.open("a") as f:
            for k, (spec, r) in enumerate(zip(expanded, results)):
                ex = teacher.g1_example(spec, r, "qwen3.8-27b-q5-mtp")
                rec = {"spec": spec.__dict__, "sample": k % best_of, "ok": ex is not None,
                       "finish": r.stop_reason, "usage": r.usage}
                if ex:
                    meta = dict(spec.meta, lang=spec.lang)
                    v0 = validate.check_sample(ex["messages"], tools if spec.tools else [], meta)
                    rec.update(messages=ex["messages"], substance=ex["substance"], feline=ex["feline"],
                               v0=[v.__dict__ for v in v0])
                    n_v0 += not v0
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_cands += 1
        n_specs += len(specs)
        el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] specs {n_specs} cands {n_cands} V0-pass {n_v0} "
              f"({n_cands / el * 3600:.0f} cands/h)", flush=True)
