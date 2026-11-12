"""Extraction, PREREG.md §4: the authors' extract2.py recipe on the stimuli.build() rows, once per prefix.

    $PY extract.py A B C

For each prefix: the residual stream at the last prompt token, every layer, fp16, in BIG/H_rs_<prefix>.npy (not
committed); its sha256 and shape go in STATE/extract_<prefix>.json. A prefix with a receipt is skipped. If the GPU
lock appears the script stops without a receipt and the next invocation redoes that prefix (extraction is
deterministic).
"""
import json
import sys

import numpy as np

import common as C


def main(prefixes):
    import torch
    C.verify_materials(prefixes=tuple(prefixes))
    S, _ = C.upstream()
    rows = S.build()
    C.BIG.mkdir(parents=True, exist_ok=True)
    C.save_json(C.STATE / "rows.json", rows)
    todo = [p for p in prefixes if not (C.STATE / f"extract_{p}.json").exists()]
    if not todo:
        return print("nothing to do")
    tok, model = C.load_model()
    tok.padding_side = "left"
    C.runtime_record("extract.py", prefixes=todo, rows=len(rows))
    L, d, bs = model.config.num_hidden_layers + 1, model.config.hidden_size, 16
    for p in todo:
        msgs = json.loads(C.prefix_path(p).read_text())
        texts = [C.chat(tok, msgs + [{"role": "user", "content": r["text"]}]) for r in rows]
        path = C.BIG / f"H_rs_{p}.npy"
        H = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(rows), L, d))
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                if C.gpu_locked():
                    raise SystemExit(f"GPU lock appeared during prefix {p}; stopped without a receipt")
                enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(C.DEVICE)
                # use_cache=False and the del: the 37 hidden-state tensors of a batch are about 3 GiB, and holding
                # the previous batch's (and a cache nothing reads) while computing the next does not fit 24 GB
                hs = model.model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                H[i:i + bs] = torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy()
                del hs, enc
        H.flush()
        del H
        C.save_json(C.STATE / f"extract_{p}.json",
                    dict(prefix=p, path=str(path), sha256=C.filehash(path), shape=[len(rows), L, d],
                         dtype="float16", rows_sha256=C.filehash(C.STATE / "rows.json"), utc=C.now()))
        print(f"prefix {p}: {len(rows)} rows × {L} layers × {d}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
