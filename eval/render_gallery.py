"""Render the image prompts an eval run composed, through mirage (FLUX on mirrorforge) as group-chat would: the
hashed image gallery of cat4b.release.tiers, one image per image_gen item per condition.

    .venv/bin/python eval/render_gallery.py --run v0 --conditions C,A,B,F [--limit 1]

Reads state/eval/<run>/<cond>.jsonl, takes the first generate_image call of every image_gen item (prompt, width,
height as the model wrote them), leases the broker's imagegen capability per render (ttl 300 s, like
imagegen_service's 240) and calls mirage's generate_image over its MCP HTTP endpoint with steps 30 (group-chat's
imagegen_steps default), FluxGuidance 3.5 (mirage's default) and a seed fixed per item (sha256 of the eval_id), so
two conditions differ by the prompt alone. Images land in state/eval/<run>/gallery/<cond>/<eval_id>.png with a
manifest line per item in gallery/<cond>.jsonl (prompt, dims, seed, remote path, sha256, seconds, error). Resumable:
items with a manifest line are skipped. Nothing in group-chat-mcp or mirage is modified; this is a client.
"""
import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import OUT, frozen_items, read_jsonl  # noqa: E402

BROKER = "http://10.10.10.2:8077"
MIRAGE = "http://10.10.10.2:8190/mcp"
SCP_HOST = "mirrorforge"
STEPS, GUIDANCE = 30, 3.5
PATH_RE = re.compile(r"(/[^\s\"']+\.(?:png|jpe?g|webp))", re.IGNORECASE)  # imagegen_service._PATH_RE


def seed_for(eval_id: str) -> int:
    return int(hashlib.sha256(eval_id.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def lease(ttl: int = 300, tries: int = 30) -> str:
    """The imagegen lease (ComfyUI loaded, vision excluded); waits while someone else holds the pool."""
    for _ in range(tries):
        r = requests.post(f"{BROKER}/lease", json={"capability": "imagegen", "ttl_sec": ttl}, timeout=200).json()
        if r.get("ok"):
            return r["lease_id"]
        print(f"  pool busy ({r.get('held_by', '?')}); waiting", flush=True)
        time.sleep(20)
    raise RuntimeError("broker never granted the imagegen lease")


def release(lid: str):
    try:
        requests.post(f"{BROKER}/release", json={"lease_id": lid}, timeout=30)
    except requests.RequestException as e:  # the lease expires on its own
        print(f"  release failed: {e}", flush=True)


def render(args: dict, out: Path) -> dict:
    """One mirage generate_image call; saves the image to `out`, returns {remote, sha256} or {error}."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "generate_image", "arguments": args}}
    r = requests.post(MIRAGE, json=body, timeout=900).json()
    if "error" in r:
        return {"error": str(r["error"])[:300]}
    content = r.get("result", {}).get("content", [])
    if r.get("result", {}).get("isError"):
        return {"error": " ".join(c.get("text", "") for c in content)[:300]}
    text = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
    remote = (PATH_RE.search(text) or [None, ""])[1]
    data = next((c.get("data") for c in content if c.get("type") == "image" and c.get("data")), None)
    if data:
        out.write_bytes(base64.b64decode(data))
    elif remote:  # no inline image: copy it off mirrorforge as imagegen_service does
        rc = subprocess.run(["scp", "-q", f"{SCP_HOST}:{remote}", str(out)], capture_output=True,
                            timeout=120).returncode
        if rc != 0 or not out.exists():
            return {"error": f"no inline image and scp failed for {remote}", "remote": remote}
    else:
        return {"error": f"no image in result: {text[:200]}"}
    return {"remote": remote, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def gen_call(rec: dict) -> dict | None:
    gen = [tc for tc in rec["tool_calls"] if tc["name"] == "generate_image"]
    if not gen:
        return None
    a = gen[0]["arguments"]
    return json.loads(a) if isinstance(a, str) else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--conditions", default="C,A,B,F")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    image_items = {it["eval_id"] for it in frozen_items() if it["slice"] == "image_gen"}
    gal = OUT / a.run / "gallery"
    for cond in a.conditions.split(","):
        (gal / cond).mkdir(parents=True, exist_ok=True)
        mpath = gal / f"{cond}.jsonl"
        done = {d["eval_id"] for d in read_jsonl(mpath)} if mpath.exists() else set()
        recs = [r for r in read_jsonl(OUT / a.run / f"{cond}.jsonl")
                if r["eval_id"] in image_items and r["eval_id"] not in done]
        if a.limit:
            recs = recs[: a.limit]
        print(f"{cond}: {len(recs)} to render, {len(done)} done", flush=True)
        with mpath.open("a") as f:
            for r in recs:
                call = gen_call(r)
                row = {"eval_id": r["eval_id"], "condition": cond, "seed": seed_for(r["eval_id"]), "steps": STEPS,
                       "cfg": GUIDANCE}
                if call is None:
                    row["error"] = "no generate_image call"
                else:
                    args = {"prompt": str(call.get("prompt", "")), "width": int(call.get("width", 1024)),
                            "height": int(call.get("height", 1024)), "steps": STEPS, "cfg": GUIDANCE,
                            "seed": row["seed"]}
                    row.update(prompt=args["prompt"], width=args["width"], height=args["height"])
                    t0, lid = time.time(), lease()
                    try:
                        row.update(render(args, gal / cond / f"{r['eval_id']}.png"))
                    except Exception as e:  # noqa: BLE001 - recorded, the item is retried on the next run
                        row["error"] = f"{type(e).__name__}: {e}"[:300]
                    finally:
                        release(lid)
                    row["seconds"] = round(time.time() - t0, 1)
                if "error" in row:
                    print(f"  {r['eval_id']}: {row['error'][:120]}", flush=True)
                    if "no generate_image call" not in row["error"]:
                        continue  # not persisted: retried next run
                else:
                    print(f"  {r['eval_id']} {row['width']}x{row['height']} {row['seconds']}s {row['sha256'][:12]}",
                          flush=True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
