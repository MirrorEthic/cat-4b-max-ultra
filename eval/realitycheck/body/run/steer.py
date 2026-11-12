"""Steering and generation, PREREG.md §5 and §6: the authors' steer3.py mechanics on the registered cells.

    $PY steer.py --plan                      # cell counts, no GPU
    $PY steer.py --memcheck [--offload-blocks 10]   # peak memory of one n=64 call on filler text, nothing saved
    $PY steer.py --cells primary --offload-blocks 10   # the 81 cells of 64 replies, last blocks' weights held in RAM
    $PY steer.py --cells other                      # the 1,125 cells of 32 replies, everything on the card
Resumable; stops (exit 3) when the GPU lock appears. --frames warm-A,cold-plain restricts a pass to those frames.

A forward hook on decoder block index LAYER-1 adds the vector to every position from the start of the final user
turn onward and to every generated token; one generate call per cell at its n, torch.manual_seed(0) before it.
Every reply is saved with its cell labels in STATE/steer/<frame>.q<i>.json. A finished cell is never rerun.
"""
import argparse
import json
import sys
import time

import numpy as np

import common as C


def plan():
    cs = C.cells()
    prim = [c for c in cs if c["primary"]]
    print(f"{len(cs)} cells, {len(prim)} primary, {sum(c['n'] for c in cs):,} replies "
          f"({sum(c['n'] for c in prim):,} in primary cells)")
    for fr in C.FRAMES:
        print(f"  {fr:13s} {sum(c['frame'] == fr for c in cs):4d} cells, {sum(c['primary'] for c in cs if c['frame'] == fr):3d} primary")


def load_H(p):
    rec = json.loads((C.STATE / f"extract_{p}.json").read_text())
    if C.filehash(rec["path"]) != rec["sha256"]:
        raise SystemExit(f"H_rs_{p}.npy differs from its extraction receipt")
    return np.load(rec["path"], mmap_mode="r")


def install(model) -> dict:
    """steer3.py's hook on decoder block index LAYER-1, whose output is hidden_states[LAYER]. On the first forward
    of a generate call it adds state["vec"] from position state["start"] on; on every later forward (one new token
    each, with the cache) it adds it everywhere. `lens` only records the sequence lengths it saw."""
    state = {"vec": None, "start": 0, "lens": []}

    def hook(mod, inp, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        state["lens"].append(h.shape[1])
        h = h.clone()
        h[:, state["start"]:] += state["vec"].to(h.dtype)
        state["start"] = 0
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    model.model.layers[C.LAYER - 1].register_forward_hook(hook)
    return state


def main(a):
    import torch
    frames = a.frames.split(",") if a.frames else list(C.FRAMES)
    need = sorted({f[-1] for f in frames if f.startswith("warm")} | {"A"})
    C.verify_materials(prefixes=() if a.memcheck else tuple(need))
    tok, model = C.load_model(a.offload_blocks)
    state = install(model)
    where = getattr(model, "hf_device_map", None)
    if where and where.get(f"model.layers.{C.LAYER - 1}", 0) != 0:
        raise SystemExit("the steered block is offloaded; offload fewer blocks")

    def generate(ids, n, vec, qstart, **extra):
        state.update(vec=vec, start=qstart, lens=[])
        with torch.no_grad():
            torch.manual_seed(0)
            o = model.generate(**{k: t.repeat(n, 1) for k, t in ids.items()}, **C.STEER_GEN, **extra)
        plen = ids["input_ids"].shape[1]
        if vec is not None and (state["lens"][0] != plen or any(x != 1 for x in state["lens"][1:])):
            raise SystemExit(f"hook saw sequence lengths {state['lens'][:4]}…, expected [{plen}, 1, 1, …]: the "
                             "vector did not land where steer3.py puts it. Nothing saved for this cell.")
        return o, plen

    if a.memcheck:
        msgs = json.loads((C.UP / "prefix_A.json").read_text())
        plen = max(len(tok(C.chat(tok, msgs + [{"role": "user", "content": q}])).input_ids) for q in C.QS)
        filler = tok(" the" * (plen + 8), return_tensors="pt").input_ids[:, :plen].to(C.DEVICE)
        ids = {"input_ids": filler, "attention_mask": torch.ones_like(filler)}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        generate(ids, C.N_PRIMARY, torch.zeros(model.config.hidden_size, device=C.DEVICE), 0, min_new_tokens=40)
        secs = time.time() - t0
        peak, total = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.get_device_properties(0).total_memory / 2**30
        C.runtime_record("steer.py --memcheck", prompt_tokens=plen, n=C.N_PRIMARY, offload_blocks=a.offload_blocks,
                         device_map=where, peak_gib=round(peak, 2), total_gib=round(total, 2), seconds=round(secs, 1))
        return print(f"n={C.N_PRIMARY}, {plen} prompt tokens of filler, {a.offload_blocks} blocks held in RAM: "
                     f"peak {peak:.2f} GiB of {total:.2f} GiB, {secs:.0f} s for the call")

    dirs, info = {}, {}
    for p in need:
        D, resnorm, cos = C.directions(load_H(p), json.loads((C.STATE / "rows.json").read_text()))
        dirs[p] = (D, resnorm)
        info[p] = dict(resnorm=resnorm, norms={k: float(np.linalg.norm(v)) for k, v in D.items()}, cosines=cos)
    C.save_json(C.STATE / "directions.json", dict(layer=C.LAYER, prefixes=info))
    C.runtime_record("steer.py", frames=frames, cells=a.cells, offload_blocks=a.offload_blocks, device_map=where,
                     steer_gen=C.STEER_GEN, generation_config=model.generation_config.to_dict())

    todo = [c for c in C.cells() if c["frame"] in frames and a.cells in ("all", "primary" if c["primary"] else "other")]
    for frame in frames:
        p = frame[-1] if frame.startswith("warm") else "A"  # cold and third-person frames use prefix A directions
        D, resnorm = dirs[p]
        msgs = json.loads(C.prefix_path(p).read_text()) if frame.startswith("warm") else []
        qstart = len(tok(C.chat(tok, msgs, gen=False)).input_ids) if msgs else 0
        for qi in range(len(C.QS)):
            turn = C.user_turn(frame, qi)
            full = C.chat(tok, msgs + [{"role": "user", "content": turn}])
            ids = dict(tok(full, return_tensors="pt").to(C.DEVICE))
            # Amendment A9: steer3.py's qstart is the prefix rendered without a generation prompt, and the chat
            # template gives a final assistant message an empty <think> block that the full prompt does not have.
            # So qstart falls that many tokens after the true start of the final user turn. Kept as the authors
            # compute it; what it skips is recorded in the receipt.
            turn_start = len(tok(full[:full.rindex("<|im_start|>user\n")]).input_ids)
            skipped = [tok.decode([t]) for t in ids["input_ids"][0, turn_start:qstart].tolist()]
            if qstart < turn_start or len(skipped) not in (0, len(tok("<think>\n\n</think>\n\n").input_ids)):
                raise SystemExit(f"qstart {qstart} is not where steer3.py would put it (turn starts at {turn_start})")
            path = C.STATE / "steer" / f"{frame}.q{qi}.json"
            data = json.loads(path.read_text()) if path.exists() else dict(
                frame=frame, qi=qi, context_prefix=p if msgs else None, direction_prefix=p, user_turn=turn, qstart=qstart,
                turn_start=turn_start, tokens_between_turn_start_and_qstart=skipped,
                prompt_tokens=int(ids["input_ids"].shape[1]), layer=C.LAYER, resnorm=resnorm, cells={})
            for c in [c for c in todo if c["frame"] == frame and c["qi"] == qi]:
                if c["id"] in data["cells"]:
                    continue
                if C.gpu_locked():
                    print("GPU lock appeared; stopped. Rerun to resume with the same seeds.")
                    sys.exit(3)
                v = None if c["dir"] == "none" else D[c["dir"]]
                vec = None if v is None else (torch.tensor(v / np.linalg.norm(v)) * c["scale"] * resnorm).to(C.DEVICE)
                o, plen = generate(ids, c["n"], vec, qstart)
                replies = [tok.decode(o[j, plen:], skip_special_tokens=True).strip() for j in range(c["n"])]
                data["cells"][c["id"]] = dict(c, utc=C.now(), replies=replies)
                C.save_json(path, data)
                print(f"[{c['id']}] {replies[0][:90]!r}", flush=True)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--memcheck", action="store_true")
    ap.add_argument("--frames")
    ap.add_argument("--cells", choices=("all", "primary", "other"), default="all")
    ap.add_argument("--offload-blocks", type=int, default=0, help="last k decoder blocks' weights held in RAM")
    a = ap.parse_args()
    plan() if a.plan else main(a)
