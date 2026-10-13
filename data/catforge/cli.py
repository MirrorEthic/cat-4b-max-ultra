"""catforge CLI.  uv run python -m catforge.cli <command>   (run from the repo root with PYTHONPATH=data)

Credentials: ANTHROPIC_API_KEY (and OPENROUTER_API_KEY for Jev) are read from the environment, then the gitignored
repo-root .env, then ~/.claude/.env (first value wins). Never commit or paste keys.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from . import REPO, read_jsonl

ENV_FILES = [REPO / ".env", Path.home() / ".claude" / ".env"]


def load_env():
    for env in ENV_FILES:
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    # catforge's Anthropic key (Garret, 2026-09-23: the freshly minted secondary key; the file's active one is stale).
    key_var = os.environ.get("CATFORGE_ANTHROPIC_KEY_VAR", "CLAUDE_KEY_SECONDARY")
    if os.environ.get(key_var):
        os.environ["ANTHROPIC_API_KEY"] = os.environ[key_var]


def cmd_ledger(_):
    from .ledger import Ledger
    print(json.dumps(Ledger().summary(), indent=1))


def cmd_bank_dryrun(_):
    from . import bank
    from .engines import est_tokens
    from .ledger import PRICES
    reqs = [bank.cell_request(c) for c in bank.sample_cells()]
    p = PRICES["claude-opus-5"]
    worst = sum((est_tokens({k: r["params"][k] for k in ("system", "messages")}) * p["cache_1h"]
                 + r["params"]["max_tokens"] * p["out"]) / 1e6 for r in reqs)
    print(f"{len(reqs)} calls → ~{len(reqs) * bank.PER_CALL} prompts; worst-case reservation ${worst:.2f}")


def cmd_bank_submit(a):
    from . import bank
    from .engines import BATCH_DIR, ClaudeBatches
    cells = bank.sample_cells()
    if a.limit:  # pilot: round-robin across slices so every prompt type is exercised
        by_slice: dict[str, list] = {}
        for c in cells:
            by_slice.setdefault(c["slice"], []).append(c)
        rr, i = [], 0
        while len(rr) < a.limit:
            for lst in by_slice.values():
                if i < len(lst) and len(rr) < a.limit:
                    rr.append(lst[i])
            i += 1
        cells = rr
    reqs = [bank.cell_request(c) for c in cells]
    if not a.limit:  # skip cells already generated in any earlier run (keyed by cell content): never pay twice
        ckey = lambda c: json.dumps(c, sort_keys=True, ensure_ascii=False)  # noqa: E731
        done = set()
        for cf in bank.BANK_DIR.glob("*.cells.json"):
            run = cf.name.removesuffix(".cells.json")
            rf = BATCH_DIR / f"{run}.results.jsonl"
            if rf.exists():
                ok = {d["custom_id"] for d in read_jsonl(rf) if d["ok"]}
                done |= {ckey(c) for cid, c in json.loads(cf.read_text()).items() if cid in ok}
        keep = [(r, c) for r, c in zip(reqs, cells) if ckey(c) not in done]
        print(f"skipping {len(reqs) - len(keep)} already-generated cells")
        reqs, cells = [r for r, _ in keep], [c for _, c in keep]
    eng = ClaudeBatches()
    bank.BANK_DIR.mkdir(parents=True, exist_ok=True)
    cells_by_id = {r["custom_id"]: c for r, c in zip(reqs, cells)}
    if a.sync:
        run_id, res = eng.run_sync(f"bank-pilot-sync", reqs)
        (bank.BANK_DIR / f"{run_id}.cells.json").write_text(json.dumps(cells_by_id))
        items = bank.parse_cell_results(cells_by_id, res)
        path = bank.write_bank(items, f"claude_{run_id}")
        bad = [r for r in res if not r.ok]
        print(f"{run_id}: {len(items)} prompts → {path}; {len(bad)} not ok "
              f"({sorted({str(r.stop_reason or r.error)[:80] for r in bad})})")
        cmd_ledger(a)
        return
    bid = eng.submit(f"bank{'-pilot' if a.limit else ''}", reqs)
    (bank.BANK_DIR / f"{bid}.cells.json").write_text(json.dumps(cells_by_id))
    print(bid)


def cmd_batch_cancel(a):
    from .engines import ClaudeBatches
    eng = ClaudeBatches()
    b = eng.cancel(a.batch_id)
    print(b.processing_status, b.request_counts)


def cmd_bank_collect(a):
    from . import bank
    from .engines import ClaudeBatches
    eng = ClaudeBatches()
    b = eng.wait(a.batch_id, every=a.every)
    res = eng.collect(a.batch_id)
    cells = json.loads((bank.BANK_DIR / f"{a.batch_id}.cells.json").read_text())
    items = bank.parse_cell_results(cells, res)
    path = bank.write_bank(items, f"claude_{a.batch_id}")
    bad = [r for r in res if not r.ok]
    print(f"{b.request_counts}; {len(items)} prompts → {path}; {len(bad)} not ok "
          f"({sorted({r.stop_reason or r.error for r in bad})})")
    cmd_ledger(a)


def cmd_bakeoff(a):
    import time
    from . import bakeoff
    specs = bakeoff.build_specs(a.n_per_slice)
    bakeoff.save("specs.json", [s.__dict__ for s in specs])
    print(f"{len(specs)} specs; generating (local + claude)…", flush=True)
    local, claude, wall = bakeoff.generate(specs)
    toks = sum(r.usage.get("completion_tokens", 0) for r in local)
    print(f"generated in {wall:.0f}s; local {toks} tokens → {toks / wall:.0f} tok/s aggregate", flush=True)
    sets = {"qwen3.8-27b": bakeoff.examples(specs, local, "qwen3.8-27b-q5-mtp"),
            "claude-opus-5": bakeoff.examples(specs, claude, "claude-opus-5")}
    for name, exs in sets.items():
        bakeoff.judge_all(exs)
        bakeoff.save(f"examples_{name}.json", exs)
    rep = bakeoff.report(sets)
    bakeoff.save("report.md", rep + "\n")
    print(rep)
    cmd_ledger(a)


def cmd_gen_local(a):
    from . import generate
    generate.run(a.slices.split(","), best_of=a.best_of, hours=a.hours, thinking=not a.no_thinking)


def cmd_judge_calibrate(a):
    """Local judge vs Claude on the bake-off set (410 Claude verdicts): agreement before trusting it."""
    import json as _j
    from . import judge
    from .bakeoff import BAKE_DIR
    from .engines import LocalChat
    exs = []
    for name in ("qwen3.8-27b", "claude-opus-5"):
        exs += [e for e in _j.loads((BAKE_DIR / f"examples_{name}.rescored.json").read_text()) if e.get("verdict")]
    if a.limit:
        exs = exs[: a.limit]
    local = judge.run_local_judge(exs, LocalChat(a.url, timeout=900), concurrency=4)
    pairs = [(e, v) for e, v in zip(exs, local) if v]
    print(f"{len(pairs)}/{len(exs)} judged locally")
    def kappa(xs):
        n = len(xs); po = sum(a == b for a, b in xs) / n
        pa = sum(a for a, _ in xs) / n; pb = sum(b for _, b in xs) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        return (po - pe) / (1 - pe) if pe < 1 else 1.0, po
    keeps = []
    for e, v in pairs:
        bad = judge.verify_quotes(v, judge.graded_output(e["messages"]))
        lk = (not bad) and (not e["v0"]) and judge.keep(v, e["spec"]["slice"], e["spec"]["device"])
        keeps.append((bool(e.get("keep")), lk))
    k, po = kappa(keeps)
    print(f"keep: agreement {po:.1%}, kappa {k:.2f} (claude keeps {sum(c for c, _ in keeps)}, local keeps {sum(l for _, l in keeps)})")
    for f in ("persistence", "retention", "grace"):
        d = [abs(e["verdict"][f]["value"] - v[f]["value"]) for e, v in pairs if e["verdict"]["eligible"] or f != "persistence"]
        print(f"{f}: exact {sum(x == 0 for x in d) / len(d):.1%}, within 1 {sum(x <= 1 for x in d) / len(d):.1%}")
    el = sum(e["verdict"]["eligible"] == v["eligible"] for e, v in pairs) / len(pairs)
    bq = sum(bool(judge.verify_quotes(v, judge.graded_output(e["messages"]))) for e, v in pairs) / len(pairs)
    print(f"eligibility agreement {el:.1%}; local verdicts with unverifiable quotes {bq:.1%}")


def cmd_judge_pass(a):
    from . import judgepass
    judgepass.run(a.url, a.judge_name, hours=a.hours)


def cmd_select(a):
    import json as _j
    from . import judgepass
    print(_j.dumps(judgepass.select(a.name), indent=1))


def main(argv=None):
    load_env()
    ap = argparse.ArgumentParser(prog="catforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ledger").set_defaults(fn=cmd_ledger)
    sub.add_parser("bank-dryrun").set_defaults(fn=cmd_bank_dryrun)
    s = sub.add_parser("bank-submit")
    s.add_argument("--limit", type=int, default=0, help="submit only N cells, round-robin across slices (pilot)")
    s.add_argument("--sync", action="store_true", help="regular API at list price instead of Batches (pilots only)")
    s.set_defaults(fn=cmd_bank_submit)
    x = sub.add_parser("batch-cancel")
    x.add_argument("batch_id")
    x.set_defaults(fn=cmd_batch_cancel)
    c = sub.add_parser("bank-collect")
    c.add_argument("batch_id")
    c.add_argument("--every", type=int, default=60)
    c.set_defaults(fn=cmd_bank_collect)
    b = sub.add_parser("bakeoff")
    b.add_argument("--n-per-slice", type=int, default=50)
    b.set_defaults(fn=cmd_bakeoff)
    g = sub.add_parser("gen-local")
    g.add_argument("--slices", default="plain,exact_payload,consequential,refusal")
    g.add_argument("--best-of", type=int, default=4)
    g.add_argument("--hours", type=float, default=24.0)
    g.add_argument("--no-thinking", action="store_true", help="Qwen3.8 enable_thinking=false (~2x faster)")
    g.set_defaults(fn=cmd_gen_local)
    jc = sub.add_parser("judge-calibrate")
    jc.add_argument("--url", default="http://10.10.10.2:8291")
    jc.add_argument("--limit", type=int, default=0)
    jc.set_defaults(fn=cmd_judge_calibrate)
    jp = sub.add_parser("judge-pass")
    jp.add_argument("--url", default="http://10.10.10.2:8291")
    jp.add_argument("--judge-name", default="gemma-4-31b-q4-v100")
    jp.add_argument("--hours", type=float, default=24.0)
    jp.set_defaults(fn=cmd_judge_pass)
    se = sub.add_parser("select")
    se.add_argument("--name", default="v0")
    se.set_defaults(fn=cmd_select)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
