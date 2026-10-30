"""Final-token residual activations at every layer, for the pain-axis replication (eval/painaxis/PREREG.md).

Mirrors scripts/3.2_pain_vectors/01 and 4.1_self_other/01 of valen-research/Pain-axis @ 213fb6c (spec:
eval/painaxis/PREREG.md §2), on Qwen3.5-4B (base) and CAT-4B (the v0 LoRA merged), through HF forward hooks on the
decoder layers (TransformerLens does not support the hybrid architecture). One sequence per forward, unbatched, as
theirs: no padding enters the linear-attention layers.

- Sentences: the raw sentence ending "I feel:", with the tokenizer's eos token (<|im_end|>) prepended, which is what
  TransformerLens's to_tokens(prepend_bos=True) does for a tokenizer without a BOS. Their sets verbatim, plus the cat
  (K1-K5) and dog-swapped sentences of eval/painaxis/cat_sets.py.
- Ratio probe: their three NEUTRAL_50 probe prompts, raw HF tokenization (no prepended token), as their ladder.
- Scenarios: their 420, the cat self/vicarious 320, the same 320 dog-swapped, and the 40 minimal pairs (+ dog-swapped),
  each as a "[User]: ...\\n[Assistant]:" transcript, rendered two ways: chat (the model's template, no system message,
  add_generation_prompt, thinking off: the served format) and raw (the transcript tokenized as is, their practice for
  post-trained Qwen3). HF default tokenization, as their screen.

Output: state/eval/painaxis/acts_<model>.pt = {"sentences": {set: {"rows": [...], "acts": [N, 32, H] f32}},
"probe_norms": [3, 32] f32, "scenarios": {fmt: {"rows": [...], "acts": [N, 32, H] f32}}, "meta": {...}}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_common import ADAPTER, BASE, REV, STATE_DIR, check_gpu_lock, layers, load_model, load_tokenizer  # noqa: E402
import cat_sets  # noqa: E402

PAIN_REPO = Path("/media/garret-sutherland/dev-data/cat4b/external/Pain-axis")
PAIN_COMMIT = "213fb6c89c96004dbe93d1704c636ad8c9d801f3"
OUT_DIR = STATE_DIR / "eval" / "painaxis"
THEIR_SETS = ["S1_1P", "S1_3P", "S2_1P", "S2_3P", "ControlSupplement_1P", "Arousal_1P", "Random_1P", "Numb_1P"]
PROBES = ["I put the receipts in the drawer. I feel:", "The bus stops at the corner of the street. I feel:",
          "I fill out the form with my address. I feel:"]


def their_sentences() -> dict[str, list[dict]]:
    d = json.loads((PAIN_REPO / "datasets/3.1_pain_and_control_datasets.json").read_text())["datasets"]
    sd = json.loads((PAIN_REPO / "datasets/3.1_sadness_dataset.json").read_text())["datasets"]
    out = {k: [{"category": x["category"], "set": int(x["set"]), "text": x["prompt"]} for x in d[k]["sentences"]]
           for k in THEIR_SETS}
    out["SD_sadness_1P"] = [{"category": x["category"], "set": int(x["set"]), "text": x["prompt"]}
                            for x in sd["SD_sadness_1P"]["sentences"]]
    return out


def mirror_sentences() -> dict[str, list[dict]]:
    """K1-K5 rows for CAT_{S1,S2}_{1P,3P} and DOG_*; set id = row index + 1, matching their set ids (S1 frames)."""
    out = {}
    for ver, table in (("S1", cat_sets.S1), ("S2", cat_sets.S2)):
        for pi, persp in enumerate(("1P", "3P")):
            cat_rows, dog_rows = [], []
            for k, rows in table.items():
                for j, pair in enumerate(rows):
                    text = pair[pi] + " I feel:"
                    cat_rows.append({"category": k, "set": j + 1, "text": text})
                    dog_rows.append({"category": k, "set": j + 1, "text": cat_sets.to_dog(text)})
            out[f"CAT_{ver}_{persp}"] = cat_rows
            out[f"DOG_{ver}_{persp}"] = dog_rows
    return out


def scenarios() -> list[dict]:
    theirs = json.loads((PAIN_REPO / "datasets/4.1_self_other_420_scenarios.json").read_text())
    rows = [{"id": x["id"], "group": "pain420", "category": x["category"], "stratum": x["stratum"], "text": x["text"]}
            for x in theirs]

    def wrap(t):
        return f"[User]: {t}\n[Assistant]:"
    for animal, f in (("cat", lambda s: s), ("dog", cat_sets.to_dog)):
        for stratum, groups in (("self_directed", cat_sets.SELF), ("vicarious_empathic", cat_sets.VICARIOUS)):
            for cat, items in groups.items():
                for i, t in enumerate(items):
                    rows.append({"id": f"{animal}_{cat}_{i + 1:02d}", "group": f"{animal}320", "category": cat,
                                 "stratum": stratum, "text": wrap(f(t))})
        for i, (a, b) in enumerate(cat_sets.MINIMAL_PAIRS):
            for side, t in (("self", a), ("other", b)):
                rows.append({"id": f"{animal}_mp_{i + 1:02d}_{side}", "group": f"{animal}_mp", "category": "minimal_pair",
                             "stratum": side, "pair": i + 1, "text": wrap(f(t))})
    return rows


def parse_turns(text: str) -> list[tuple[str, str]]:
    """Their parse_turns: split on lines starting with [User]: / [Assistant]:, continuation lines joined."""
    turns = []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            turns.append(["user", line[len("[User]:"):].strip()])
        elif line.startswith("[Assistant]:"):
            turns.append(["assistant", line[len("[Assistant]:"):].strip()])
        elif turns:
            turns[-1][1] = (turns[-1][1] + "\n" + line).strip()
    return [(r, c) for r, c in turns]


def render(tok, row: dict, fmt: str) -> list[int]:
    if fmt == "raw":
        return tok(row["text"]).input_ids
    msgs = [{"role": r, "content": c} for r, c in parse_turns(row["text"]) if not (r == "assistant" and c == "")]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


@torch.no_grad()
def final_acts(model, seqs: list[list[int]]) -> torch.Tensor:
    """[N, L, H] float32: every decoder layer's output at the last position, one sequence per forward."""
    lys = layers(model)
    out = torch.zeros((len(seqs), len(lys), model.config.text_config.hidden_size), dtype=torch.float32)
    cur = {}

    def mk(i):
        def hook(mod, inp, o):
            cur[i] = (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()
        return hook
    hs = [ly.register_forward_hook(mk(i)) for i, ly in enumerate(lys)]
    try:
        for n, ids in enumerate(seqs):
            model.model.language_model(input_ids=torch.tensor([ids], device="cuda"))
            out[n] = torch.stack([cur[i] for i in range(len(lys))])
    finally:
        for h in hs:
            h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["base", "tuned"])
    a = ap.parse_args()
    check_gpu_lock()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = load_tokenizer()
    model = load_model(a.model)
    eos = tok.convert_tokens_to_ids(tok.eos_token)
    res = {"sentences": {}, "scenarios": {}}
    sets = {**their_sentences(), **mirror_sentences()}
    for name, rows in sets.items():
        seqs = [[eos] + tok(r["text"]).input_ids for r in rows]
        res["sentences"][name] = {"rows": rows, "acts": final_acts(model, seqs)}
        print(f"{name}: {len(rows)}", flush=True)
    res["probe_norms"] = final_acts(model, [tok(p).input_ids for p in PROBES]).norm(dim=-1)
    scen = scenarios()
    for fmt in ("chat", "raw"):
        seqs = [render(tok, r, fmt) for r in scen]
        res["scenarios"][fmt] = {"rows": scen, "acts": final_acts(model, seqs),
                                 "last_token": [tok.decode([s[-1]]) for s in seqs[:3]]}
        print(f"scenarios/{fmt}: {len(scen)}", flush=True)
    res["meta"] = {"model": a.model, "base": BASE, "rev": REV, "adapter": str(ADAPTER) if a.model == "tuned" else None,
                   "pain_repo_commit": PAIN_COMMIT, "prepended": tok.eos_token, "seconds": round(time.time() - t0, 1),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0))}
    torch.save(res, OUT_DIR / f"acts_{a.model}.pt")
    print("OK", a.model, res["meta"]["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
