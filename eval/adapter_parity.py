"""vLLM ↔ HF+PEFT adapter-parity gate (docs/BUILD_PLAN.md pre-flight).

Qwen3.5 is hybrid (Gated-DeltaNet + full attention) and vLLM packs some of its projections, so a LoRA that loads
without error can still be applied wrong or partly dropped. This scores the same token IDs both ways, base and adapter,
and compares per-token logprobs on the assistant targets:

    # 1. HF side, train venv, GPU free:
    train-cu130/bin/python eval/adapter_parity.py hf --adapter <dir> --data <kept.jsonl> [-n 24]
    # 2. vLLM side, serve/vllm.sh up with the same adapter:
    .venv/bin/python eval/adapter_parity.py vllm [--url http://127.0.0.1:8011]

Gate: (a) vLLM's adapter-on logprobs track HF+PEFT about as closely as the two base models track each other (kernel
noise); (b) the LoRA's effect (adapter − base, per token) agrees between engines (correlation, and vLLM keeps ≥90% of
HF's mean NLL drop). (b) is what catches a silently dropped module family. Writes state/parity/report.json.
"""
import argparse
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data"))

from catforge import BASE_MODEL, BASE_REVISION, STATE_DIR, read_jsonl, render  # noqa: E402
from catforge.spec import load_tools  # noqa: E402

OUT = STATE_DIR / "parity"


GPU_LOCK = Path.home() / "CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED"


def cmd_hf(a):
    if GPU_LOCK.exists():  # loading the 4B on the 5090 next to a locked long run would OOM it
        sys.exit(f"GPU lock present ({GPU_LOCK}); not loading the model")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    tok = render.load_tokenizer()
    tools = load_tools()
    exs = read_jsonl(a.data)
    random.Random(a.seed).shuffle(exs)
    items = []
    for ex in exs[: a.n]:
        s = render.expand(tok, ex["messages"], tools if ex.get("tools") else None, enable_thinking=False,
                          max_cuts=1, rng=random.Random(a.seed))[-1]
        items.append({"id": ex["id"], "input_ids": s.input_ids,
                      "targets": [i for i in range(1, len(s.labels)) if s.labels[i] != -100]})
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.adapter).eval()

    def score(ids, targets):
        x = torch.tensor([ids], device="cuda")
        keep = torch.tensor([t - 1 for t in targets], device="cuda")
        with torch.no_grad():
            lg = model(input_ids=x, logits_to_keep=keep).logits[0].float().log_softmax(-1)
        return lg[torch.arange(len(targets)), torch.tensor([ids[t] for t in targets], device="cuda")].tolist()

    for it in items:
        it["hf_cat"] = score(it["input_ids"], it["targets"])
        with model.disable_adapter():
            it["hf_base"] = score(it["input_ids"], it["targets"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hf.json").write_text(json.dumps({"adapter": str(a.adapter), "items": items}))
    print(f"wrote {OUT / 'hf.json'}: {len(items)} items, {sum(len(i['targets']) for i in items)} target tokens")


def _vllm_logprobs(url, model, ids, targets):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(f"{url}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        pl = json.load(r)["choices"][0]["prompt_logprobs"]
    return [pl[t][str(ids[t])]["logprob"] for t in targets]


def _corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx, sy = math.sqrt(sum((a - mx) ** 2 for a in x)), math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else float("nan")


def cmd_vllm(a):
    hf = json.loads((OUT / "hf.json").read_text())
    rows = {k: [] for k in ("hf_base", "hf_cat", "vl_base", "vl_cat")}
    for it in hf["items"]:
        it["vl_base"] = _vllm_logprobs(a.url, a.base_name, it["input_ids"], it["targets"])
        it["vl_cat"] = _vllm_logprobs(a.url, a.cat_name, it["input_ids"], it["targets"])
        for k in rows:
            rows[k] += it[k]
    n = len(rows["hf_base"])
    mad = lambda x, y: sum(abs(p - q) for p, q in zip(x, y)) / n  # noqa: E731
    nll = {k: -sum(v) / n for k, v in rows.items()}
    d_hf = [c - b for c, b in zip(rows["hf_cat"], rows["hf_base"])]
    d_vl = [c - b for c, b in zip(rows["vl_cat"], rows["vl_base"])]
    rep = {
        "adapter": hf["adapter"], "items": len(hf["items"]), "target_tokens": n, "nll": nll,
        "mad_base_hf_vs_vllm": mad(rows["hf_base"], rows["vl_base"]),
        "mad_cat_hf_vs_vllm": mad(rows["hf_cat"], rows["vl_cat"]),
        "lora_effect_corr": _corr(d_hf, d_vl),
        "nll_drop_hf": nll["hf_base"] - nll["hf_cat"], "nll_drop_vllm": nll["vl_base"] - nll["vl_cat"],
    }
    rep["nll_drop_kept"] = rep["nll_drop_vllm"] / rep["nll_drop_hf"] if rep["nll_drop_hf"] else float("nan")
    rep["gate"] = {
        "cat_tracks_like_base": rep["mad_cat_hf_vs_vllm"] <= max(2 * rep["mad_base_hf_vs_vllm"], 0.05),
        "lora_effect_corr>=0.9": rep["lora_effect_corr"] >= 0.9,
        "nll_drop_kept>=0.9": rep["nll_drop_kept"] >= 0.9,
    }
    rep["pass"] = all(rep["gate"].values())
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["pass"] else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hf")
    h.add_argument("--adapter", required=True)
    h.add_argument("--data", default=str(STATE_DIR / "sft/v0/kept.jsonl"))
    h.add_argument("-n", type=int, default=24)
    h.add_argument("--seed", type=int, default=7)
    h.set_defaults(fn=cmd_hf)
    v = sub.add_parser("vllm")
    v.add_argument("--url", default="http://127.0.0.1:8011")
    v.add_argument("--base-name", default="qwen3.5-4b-vl")
    v.add_argument("--cat-name", default="cat4b-vl")
    v.set_defaults(fn=cmd_vllm)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
