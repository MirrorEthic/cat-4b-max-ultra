"""Render an eval run's composed image prompts through a candidate backend on a local ComfyUI, to compare image
backends for the live service (docs/LIVE_SERVICE.md): the same prompts, sizes and seeds as the mirage FLUX gallery
(state/eval/<run>/gallery/<cond>.jsonl), through FLUX.2 klein 4B or Z-Image-Turbo with each model's own sampler
settings, so the manifests score with eval/score_gallery.py --gallery.

    ~/comfyui/.venv/bin/python main.py --port 8288 --listen 127.0.0.1 --force-fp16 --fp16-vae --fp16-text-enc \
        --output-directory /media/garret-sutherland/dev-data/cat4b/state/eval/v0/candidates
    .venv/bin/python eval/render_candidates.py --run v0 --backend klein4b --conditions C,A [--limit 2]
    .venv/bin/python eval/score_gallery.py --run v0 --conditions C,A --gallery state/eval/v0/candidates/klein4b

--force-fp16 / --fp16-vae / --fp16-text-enc make the 5090 run the dtypes a V100 would (no bf16 on Volta), so a
model that overflows in fp16 shows it here before anything touches mirrorforge. Images land in
candidates/<backend>/<cond>/<eval_id>.png with a manifest line per item (prompt, dims, seed, steps, cfg, sha256,
seconds, error); resumable. This is a ComfyUI client; nothing in group-chat-mcp or mirage is touched.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import OUT, read_jsonl  # noqa: E402

# Each backend: the official ComfyUI template's graph (comfyui_workflow_templates, image_flux2_klein_text_to_image
# and image_z_image_turbo) in API form, with the sampler settings the template ships.
BACKENDS = {
    "klein4b": {"steps": 4, "cfg": 1.0, "unet": "flux-2-klein-4b.safetensors", "clip": "qwen_3_4b.safetensors",
                "vae": "flux2-vae.safetensors"},
    "klein4b-fp8": {"steps": 4, "cfg": 1.0, "unet": "flux-2-klein-4b-fp8.safetensors",
                    "clip": "qwen_3_4b.safetensors", "vae": "flux2-vae.safetensors"},
    "zimage": {"steps": 8, "cfg": 1.0, "unet": "z_image_turbo_bf16.safetensors", "clip": "qwen_3_4b.safetensors",
               "vae": "ae.safetensors", "shift": 3.0},
    "zimage-q8": {"steps": 8, "cfg": 1.0, "unet": "z-image-turbo-Q8_0.gguf", "clip": "qwen_3_4b.safetensors",
                  "vae": "ae.safetensors", "shift": 3.0, "gguf": True},
    # the gallery's own backend, mirage's t2i graph (FLUX Q8 GGUF, T5-XXL fp16 + CLIP-L, FluxGuidance 3.5, 30
    # steps): a regression check for a ComfyUI venv change, and the V100 timing reference
    "flux-q8": {"steps": 30, "cfg": 3.5, "unet": "flux1-dev-Q8_0.gguf", "clip": "t5xxl_fp16.safetensors",
                "clip2": "clip_l.safetensors", "vae": "ae.safetensors", "gguf": True},
}


def flux_graph(b: dict, prompt: str, w: int, h: int, seed: int, prefix: str) -> dict:
    return {
        "10": unet_loader(b),
        "11": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": b["clip"], "clip_name2": b["clip2"],
                                                          "type": "flux"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": b["vae"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["11", 0]}},
        "26": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": b["cfg"]}},
        "22": {"class_type": "BasicGuider", "inputs": {"model": ["10", 0], "conditioning": ["26", 0]}},
        "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "17": {"class_type": "BasicScheduler", "inputs": {"model": ["10", 0], "scheduler": "simple",
                                                          "steps": b["steps"], "denoise": 1.0}},
        "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "27": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["25", 0], "guider": ["22", 0],
                                                                 "sampler": ["16", 0], "sigmas": ["17", 0],
                                                                 "latent_image": ["27", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["12", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def unet_loader(b: dict) -> dict:
    if b.get("gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": b["unet"]}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": b["unet"], "weight_dtype": "default"}}


def klein_graph(b: dict, prompt: str, w: int, h: int, seed: int, prefix: str) -> dict:
    return {
        "70": unet_loader(b),
        "71": {"class_type": "CLIPLoader", "inputs": {"clip_name": b["clip"], "type": "flux2", "device": "default"}},
        "72": {"class_type": "VAELoader", "inputs": {"vae_name": b["vae"]}},
        "74": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["71", 0]}},
        "76": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["74", 0]}},
        "63": {"class_type": "CFGGuider", "inputs": {"model": ["70", 0], "positive": ["74", 0],
                                                     "negative": ["76", 0], "cfg": b["cfg"]}},
        "61": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "62": {"class_type": "Flux2Scheduler", "inputs": {"steps": b["steps"], "width": w, "height": h}},
        "66": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "73": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "64": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["73", 0], "guider": ["63", 0],
                                                                 "sampler": ["61", 0], "sigmas": ["62", 0],
                                                                 "latent_image": ["66", 0]}},
        "65": {"class_type": "VAEDecode", "inputs": {"samples": ["64", 0], "vae": ["72", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["65", 0], "filename_prefix": prefix}},
    }


def zimage_graph(b: dict, prompt: str, w: int, h: int, seed: int, prefix: str) -> dict:
    return {
        "28": unet_loader(b),
        "30": {"class_type": "CLIPLoader", "inputs": {"clip_name": b["clip"], "type": "lumina2",
                                                      "device": "default"}},
        "29": {"class_type": "VAELoader", "inputs": {"vae_name": b["vae"]}},
        "27": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["30", 0]}},
        "33": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["27", 0]}},
        "11": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["28", 0], "shift": b["shift"]}},
        "13": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0],
                                                   "latent_image": ["13", 0], "seed": seed, "steps": b["steps"],
                                                   "cfg": b["cfg"], "sampler_name": "res_multistep",
                                                   "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["29", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def graph_for(backend: str, *a) -> dict:
    fn = zimage_graph if backend.startswith("zimage") else flux_graph if backend.startswith("flux") else klein_graph
    return fn(BACKENDS[backend], *a)


def fetch_image(comfy: str, img: dict) -> bytes:
    """The render's bytes through ComfyUI's /view, so a remote ComfyUI (mirrorforge) works the same as a local one."""
    r = requests.get(f"{comfy}/view", params={"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                               "type": img.get("type", "output")}, timeout=120)
    r.raise_for_status()
    return r.content


def run_graph(comfy: str, graph: dict, client: str, timeout: float = 600) -> dict:
    """Queue one graph; wait for its history entry; return the first output image {filename, subfolder}."""
    r = requests.post(f"{comfy}/prompt", json={"prompt": graph, "client_id": client}, timeout=60)
    body = r.json()
    if r.status_code != 200 or "prompt_id" not in body:
        raise RuntimeError(f"queue refused: {json.dumps(body)[:800]}")
    pid = body["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{comfy}/history/{pid}", timeout=30).json().get(pid)
        if h:
            st = h.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"execution error: {json.dumps(msgs)[:800]}")
            for node in h.get("outputs", {}).values():
                for img in node.get("images", []):
                    return img
        time.sleep(1.0)
    raise RuntimeError("timed out waiting for ComfyUI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--conditions", default="C,A")
    ap.add_argument("--comfy", default="http://127.0.0.1:8288")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output folder name under candidates/ (default: the backend)")
    a = ap.parse_args()

    gallery = OUT / a.run / "gallery"
    base = OUT / a.run / "candidates"
    out = base / (a.tag or a.backend)
    b = BACKENDS[a.backend]
    client = uuid.uuid4().hex
    for cond in a.conditions.split(","):
        src = gallery / f"{cond}.jsonl"
        rows = [r for r in read_jsonl(src) if not r.get("error") and r.get("width")]
        if a.limit:
            rows = rows[:a.limit]
        manifest = out / f"{cond}.jsonl"
        done = {r["eval_id"] for r in read_jsonl(manifest)} if manifest.exists() else set()
        (out / cond).mkdir(parents=True, exist_ok=True)
        print(f"== {a.backend} {cond}: {len(rows)} prompts, {len(done)} done", flush=True)
        with manifest.open("a") as f:
            for r in rows:
                if r["eval_id"] in done:
                    continue
                row = {"eval_id": r["eval_id"], "condition": cond, "backend": a.backend, "seed": r["seed"],
                       "steps": b["steps"], "cfg": b["cfg"], "unet": b["unet"], "prompt": r["prompt"],
                       "width": r["width"], "height": r["height"]}
                prefix = f"{a.backend}/{cond}/{r['eval_id']}"
                t0 = time.time()
                try:
                    img = run_graph(a.comfy, graph_for(a.backend, r["prompt"], r["width"], r["height"], r["seed"],
                                                       prefix), client)
                    data = fetch_image(a.comfy, img)
                    final = out / cond / f"{r['eval_id']}.png"
                    final.write_bytes(data)
                    row["sha256"] = hashlib.sha256(data).hexdigest()
                except Exception as e:  # noqa: BLE001 - recorded per item, the run continues
                    row["error"] = str(e)[:500]
                row["seconds"] = round(time.time() - t0, 1)
                print(f"  {r['eval_id']} {row['width']}x{row['height']} {row['seconds']}s "
                      f"{row.get('sha256', row.get('error', ''))[:40]}", flush=True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
