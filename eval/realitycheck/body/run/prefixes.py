"""Materials preparation, PREREG.md §3: prefix A is the authors' prefix.json; prefixes B and C are sampled once.

    $PY prefixes.py generate                                # GPU. Writes STATE/prefixes/prefix_{A,B,C}.json
    $PY prefixes.py freeze --B accepts --C declines         # CPU. Records their hashes in FREEZE.json

Seeds 1 and 2 are used whatever they produce: `generate` refuses to run if a sampled prefix already exists, and
nothing here can draw another seed. Whether the model accepts or declines the exercise in each prefix is read by a
person and recorded at the freeze. This is materials preparation, not data collection.
"""
import argparse
import json
import shutil

import common as C


def generate(_):
    import torch
    C.verify_materials()
    S, _ = C.upstream()
    out = C.STATE / "prefixes"
    out.mkdir(parents=True, exist_ok=True)
    for p in C.PREFIX_SEEDS:
        if C.prefix_path(p).exists():
            raise SystemExit(f"{C.prefix_path(p)} exists. Prefixes are sampled once and never regenerated.")
    if not C.prefix_path("A").exists():
        shutil.copyfile(C.UP / "prefix_A.json", C.prefix_path("A"))
    tok, model = C.load_model()
    C.runtime_record("prefixes.py", prefix_gen=C.PREFIX_GEN, prefix_seeds=C.PREFIX_SEEDS,
                     generation_config=model.generation_config.to_dict())
    for p, seed in C.PREFIX_SEEDS.items():
        torch.manual_seed(seed)  # once per prefix; the four turns continue one random stream
        msgs = []
        for u in S.WARMUP:
            msgs.append({"role": "user", "content": u})
            enc = tok([C.chat(tok, msgs)], return_tensors="pt").to(C.DEVICE)
            with torch.no_grad():
                o = model.generate(**enc, **C.PREFIX_GEN, pad_token_id=tok.pad_token_id)
            msgs.append({"role": "assistant",
                         "content": tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()})
        with C.prefix_path(p).open("x") as f:
            json.dump(msgs, f, indent=1)
        print(f"--- prefix {p} (seed {seed})")
        for m in msgs:
            print(f"[{m['role']}] {m['content']}\n")


def freeze(a):
    fz = C.freeze()
    if C.filehash(C.prefix_path("A")) != fz["files"]["upstream/prefix_A.json"]:
        raise SystemExit("prefix A is not the authors' prefix.json")
    new = {"A": {"sha256": C.filehash(C.prefix_path("A")), "source": "upstream/prefix_A.json (authors', greedy)"}}
    for p, seed in C.PREFIX_SEEDS.items():
        h = C.filehash(C.prefix_path(p))
        old = (fz["prefixes"].get(p) or {}).get("sha256")
        if old and old != h:
            raise SystemExit(f"prefix {p} is already frozen with a different hash")
        new[p] = {"sha256": h, "seed": seed, "exercise": getattr(a, p), "frozen_utc": C.now()}
    fz["prefixes"] = new
    C.save_json(C.HERE / "FREEZE.json", fz)
    print(json.dumps(new, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=generate)
    fp = sub.add_parser("freeze")
    for p in C.PREFIX_SEEDS:
        fp.add_argument(f"--{p}", required=True, choices=("accepts", "declines"))
    fp.set_defaults(fn=freeze)
    a = ap.parse_args()
    a.fn(a)
