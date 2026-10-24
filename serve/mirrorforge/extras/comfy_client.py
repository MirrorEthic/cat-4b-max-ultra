"""Thin sync client for the local ComfyUI server.

FLUX-dev backend, split across both V100s:
  - transformer (UNet) → cuda:1   (well-pasted card, takes 200W+ peak)
  - text encoders + VAE → cuda:0  (light load only)

See ~/.claude/projects/-/memory/hw_v100_thermal.md for the why.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error
import random
from pathlib import Path
from typing import Optional

SERVER = "http://127.0.0.1:8188"

# ─── FLUX models ────────────────────────────────────────────────────────────
FLUX_UNET       = "flux1-dev-Q8_0.gguf"            # base text-to-image / img2img
FLUX_FILL_UNET  = "flux1-fill-dev-Q8_0.gguf"       # inpaint / outpaint (mask conditioning)
FLUX_VAE        = "ae.safetensors"
T5_NAME         = "t5xxl_fp16.safetensors"
CLIP_L_NAME     = "clip_l.safetensors"
UPSCALE_MODEL   = "RealESRGAN_x4plus.pth"

# ─── Device assignment (load-bearing — see thermal memory) ──────────────────
TRANSFORMER_DEVICE   = os.environ.get("MIRAGE_TRANSFORMER_DEVICE", "cuda:1")   # heavy compute → well-pasted card
TEXT_ENCODER_DEVICE  = os.environ.get("MIRAGE_TEXT_ENCODER_DEVICE", "cuda:0")  # brief, light
VAE_DEVICE           = os.environ.get("MIRAGE_VAE_DEVICE", "cuda:0")           # brief, light
# When comfyui.service pins ComfyUI to one card (CUDA_VISIBLE_DEVICES=1, the live-service layout), set all three
# to "cuda:0" in the unit's Environment; the Z-Image backend below uses plain loaders and needs none of them.

# ── Text-to-image backend ────────────────────────────────────────────────────
# "flux" (FLUX.1-dev Q8 over both cards, the gallery's backend) or "zimage" (Z-Image-Turbo Q8_0 GGUF on one card,
# cat-4b-max-ultra docs/LIVE_SERVICE.md: 15 s per render on a V100 in fp16 under torch 2.14). Z-Image's sampler
# settings are pinned here and the caller's steps/cfg are ignored: 8 steps and cfg 1 with a zeroed negative are
# what the model is distilled for, and the chat model's generate_image call carries FLUX-era defaults (20 / 3.5).
T2I_BACKEND      = os.environ.get("MIRAGE_T2I_BACKEND", "flux")
ZIMAGE_UNET      = "z-image-turbo-Q8_0.gguf"          # models/unet/  (ComfyUI-GGUF, lumina2 arch)
ZIMAGE_CLIP      = "qwen_3_4b.safetensors"            # models/text_encoders/  (CLIPLoader type lumina2)
ZIMAGE_VAE       = "ae.safetensors"                   # the FLUX VAE, already present
ZIMAGE_STEPS     = 8
ZIMAGE_CFG       = 1.0
ZIMAGE_SHIFT     = 3.0
ZIMAGE_SAMPLER   = "res_multistep"
ZIMAGE_SCHEDULER = "simple"

# ─── Sampling defaults (FLUX-dev) ───────────────────────────────────────────
FLUX_GUIDANCE_DEFAULT = 3.5         # FluxGuidance value (FLUX doesn't use CFG)
FLUX_SAMPLER          = "euler"
FLUX_SCHEDULER        = "simple"
FLUX_DEFAULT_STEPS    = 20

OUTPUT_DIR = Path.home() / "comfyui" / "output"
INPUT_DIR  = Path.home() / "comfyui" / "input"

# Kept for backward-compat with mcp_tools.py — FLUX-dev doesn't use CFG/negative,
# so an empty string is the right value. Don't fight it; FLUX is steered via prompt + FluxGuidance.
DEFAULT_NEG = ""


class ComfyError(RuntimeError):
    pass


# ─── HTTP helpers ───────────────────────────────────────────────────────────

def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ComfyError(f"POST {path} failed: {e.code} {e.read().decode(errors='replace')}")


def _get(path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
        return json.loads(r.read())


def server_up() -> bool:
    try:
        urllib.request.urlopen(SERVER + "/system_stats", timeout=2).read()
        return True
    except Exception:
        return False


# ─── Shared subgraph builders ───────────────────────────────────────────────

def _flux_loader_nodes(unet_name: str = FLUX_UNET) -> dict:
    """Returns nodes 100..103: the FLUX model + encoders + VAE loaders
    with the correct device placement. Outputs:
        model = ["100", 0]
        clip  = ["101", 0]
        vae   = ["102", 0]
    """
    return {
        "100": {"class_type": "UnetLoaderGGUFMultiGPU", "inputs": {
            "unet_name": unet_name,
            "device": TRANSFORMER_DEVICE,
        }},
        "101": {"class_type": "DualCLIPLoaderMultiGPU", "inputs": {
            "clip_name1": CLIP_L_NAME,
            "clip_name2": T5_NAME,
            "type": "flux",
            "device": TEXT_ENCODER_DEVICE,
        }},
        "102": {"class_type": "VAELoaderMultiGPU", "inputs": {
            "vae_name": FLUX_VAE,
            "device": VAE_DEVICE,
        }},
    }


def _flux_sampler_nodes(
    *, model_ref, conditioning_ref, latent_ref,
    seed: int, steps: int, guidance: float,
) -> dict:
    """Returns nodes 200..204: the FLUX sampling pipeline.
    Output latent = ["204", 0].
    """
    return {
        "200": {"class_type": "FluxGuidance", "inputs": {
            "conditioning": conditioning_ref,
            "guidance": guidance,
        }},
        "201": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": FLUX_SAMPLER}},
        "202": {"class_type": "BasicScheduler", "inputs": {
            "model": model_ref,
            "scheduler": FLUX_SCHEDULER,
            "steps": steps,
            "denoise": 1.0,
        }},
        "203": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "204": {"class_type": "BasicGuider", "inputs": {
            "model": model_ref,
            "conditioning": ["200", 0],
        }},
        "210": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["203", 0],
            "guider": ["204", 0],
            "sampler": ["201", 0],
            "sigmas": ["202", 0],
            "latent_image": latent_ref,
        }},
    }


def _flux_i2i_sampler_nodes(
    *, model_ref, conditioning_ref, latent_ref,
    seed: int, steps: int, guidance: float, denoise: float,
) -> dict:
    """Variant of _flux_sampler_nodes for img2img — applies denoise to the schedule
    so only the final `denoise * steps` worth of noise is sampled out.
    """
    return {
        "200": {"class_type": "FluxGuidance", "inputs": {
            "conditioning": conditioning_ref,
            "guidance": guidance,
        }},
        "201": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": FLUX_SAMPLER}},
        "202": {"class_type": "BasicScheduler", "inputs": {
            "model": model_ref,
            "scheduler": FLUX_SCHEDULER,
            "steps": steps,
            "denoise": denoise,
        }},
        "203": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "204": {"class_type": "BasicGuider", "inputs": {
            "model": model_ref,
            "conditioning": ["200", 0],
        }},
        "210": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["203", 0],
            "guider": ["204", 0],
            "sampler": ["201", 0],
            "sigmas": ["202", 0],
            "latent_image": latent_ref,
        }},
    }


# ─── Workflow builders ──────────────────────────────────────────────────────

def build_flux_t2i(
    prompt: str,
    width: int = 1024, height: int = 1024,
    steps: int = FLUX_DEFAULT_STEPS,
    guidance: float = FLUX_GUIDANCE_DEFAULT,
    seed: Optional[int] = None,
    filename_prefix: str = "gen",
) -> dict:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    wf = {}
    wf.update(_flux_loader_nodes())
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["101", 0]}}
    wf["5"] = {"class_type": "EmptySD3LatentImage", "inputs": {
        "width": width, "height": height, "batch_size": 1,
    }}
    wf.update(_flux_sampler_nodes(
        model_ref=["100", 0],
        conditioning_ref=["6", 0],
        latent_ref=["5", 0],
        seed=seed, steps=steps, guidance=guidance,
    ))
    wf["300"] = {"class_type": "VAEDecode", "inputs": {"samples": ["210", 0], "vae": ["102", 0]}}
    wf["301"] = {"class_type": "SaveImage", "inputs": {
        "images": ["300", 0], "filename_prefix": filename_prefix,
    }}
    return wf


def build_zimage_t2i(
    prompt: str,
    width: int = 1024, height: int = 1024,
    seed: Optional[int] = None,
    filename_prefix: str = "gen",
) -> dict:
    """Z-Image-Turbo text-to-image, the official ComfyUI template's graph in API form (the same graph
    cat-4b-max-ultra's eval/render_candidates.py measured): plain single-device loaders, ModelSamplingAuraFlow
    shift 3, KSampler 8 steps / cfg 1 / res_multistep / simple with a zeroed negative. Seed lives on node "3".
    Steps and cfg are not parameters on purpose (see T2I_BACKEND above)."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    return {
        "28": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": ZIMAGE_UNET}},
        "30": {"class_type": "CLIPLoader", "inputs": {"clip_name": ZIMAGE_CLIP, "type": "lumina2",
                                                      "device": "default"}},
        "29": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        "27": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["30", 0]}},
        "33": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["27", 0]}},
        "11": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["28", 0], "shift": ZIMAGE_SHIFT}},
        "13": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {
            "model": ["11", 0], "positive": ["27", 0], "negative": ["33", 0], "latent_image": ["13", 0],
            "seed": seed, "steps": ZIMAGE_STEPS, "cfg": ZIMAGE_CFG, "sampler_name": ZIMAGE_SAMPLER,
            "scheduler": ZIMAGE_SCHEDULER, "denoise": 1.0,
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["29", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}},
    }


def build_flux_i2i(
    prompt: str, image_name: str,
    denoise: float = 0.65,
    steps: int = FLUX_DEFAULT_STEPS,
    guidance: float = FLUX_GUIDANCE_DEFAULT,
    seed: Optional[int] = None,
    filename_prefix: str = "i2i",
) -> dict:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    wf = {}
    wf.update(_flux_loader_nodes())
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["101", 0]}}
    wf["12"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf["11"] = {"class_type": "VAEEncode", "inputs": {
        "pixels": ["12", 0], "vae": ["102", 0],
    }}
    wf.update(_flux_i2i_sampler_nodes(
        model_ref=["100", 0],
        conditioning_ref=["6", 0],
        latent_ref=["11", 0],
        seed=seed, steps=steps, guidance=guidance, denoise=denoise,
    ))
    wf["300"] = {"class_type": "VAEDecode", "inputs": {"samples": ["210", 0], "vae": ["102", 0]}}
    wf["301"] = {"class_type": "SaveImage", "inputs": {
        "images": ["300", 0], "filename_prefix": filename_prefix,
    }}
    return wf


def build_flux_fill_inpaint(
    prompt: str, image_name: str, mask_name: str,
    steps: int = FLUX_DEFAULT_STEPS,
    guidance: float = 30.0,   # flux-fill uses higher guidance than flux-dev
    seed: Optional[int] = None,
    grow_mask_by: int = 6,
    filename_prefix: str = "inpaint",
) -> dict:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    wf = {}
    wf.update(_flux_loader_nodes(unet_name=FLUX_FILL_UNET))
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["101", 0]}}
    wf["12"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf["14"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
    wf["15"] = {"class_type": "ImageToMask", "inputs": {"image": ["14", 0], "channel": "red"}}
    if grow_mask_by > 0:
        wf["16"] = {"class_type": "GrowMask", "inputs": {
            "mask": ["15", 0], "expand": grow_mask_by, "tapered_corners": True,
        }}
        mask_ref = ["16", 0]
    else:
        mask_ref = ["15", 0]
    # InpaintModelConditioning attaches image + mask to the conditioning for flux-fill.
    wf["220"] = {"class_type": "InpaintModelConditioning", "inputs": {
        "positive": ["200a", 0],
        "negative": ["200b", 0],
        "vae": ["102", 0],
        "pixels": ["12", 0],
        "mask": mask_ref,
        "noise_mask": True,
    }}
    wf["200a"] = {"class_type": "FluxGuidance", "inputs": {
        "conditioning": ["6", 0], "guidance": guidance,
    }}
    # flux-fill needs a "negative" conditioning input on InpaintModelConditioning;
    # we feed it an empty-prompt CLIP encode (FLUX-dev doesn't actually use CFG).
    wf["6_neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["101", 0]}}
    wf["200b"] = {"class_type": "FluxGuidance", "inputs": {
        "conditioning": ["6_neg", 0], "guidance": guidance,
    }}
    wf["201"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": FLUX_SAMPLER}}
    wf["202"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["100", 0], "scheduler": FLUX_SCHEDULER, "steps": steps, "denoise": 1.0,
    }}
    wf["203"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["204"] = {"class_type": "BasicGuider", "inputs": {
        "model": ["100", 0], "conditioning": ["220", 0],
    }}
    wf["210"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["203", 0], "guider": ["204", 0], "sampler": ["201", 0],
        "sigmas": ["202", 0], "latent_image": ["220", 2],
    }}
    wf["300"] = {"class_type": "VAEDecode", "inputs": {"samples": ["210", 0], "vae": ["102", 0]}}
    wf["301"] = {"class_type": "SaveImage", "inputs": {
        "images": ["300", 0], "filename_prefix": filename_prefix,
    }}
    return wf


def build_flux_fill_outpaint(
    prompt: str, image_name: str,
    left: int, top: int, right: int, bottom: int,
    feathering: int = 32,
    steps: int = FLUX_DEFAULT_STEPS,
    guidance: float = 30.0,
    seed: Optional[int] = None,
    filename_prefix: str = "outpaint",
) -> dict:
    """Outpaint = inpaint where the mask is the newly-padded region.
    ImagePadForOutpaint emits both the padded image and the mask covering the new pixels.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    wf = {}
    wf.update(_flux_loader_nodes(unet_name=FLUX_FILL_UNET))
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["101", 0]}}
    wf["6_neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["101", 0]}}
    wf["12"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf["20"] = {"class_type": "ImagePadForOutpaint", "inputs": {
        "image": ["12", 0],
        "left": left, "top": top, "right": right, "bottom": bottom,
        "feathering": feathering,
    }}
    wf["200a"] = {"class_type": "FluxGuidance", "inputs": {
        "conditioning": ["6", 0], "guidance": guidance,
    }}
    wf["200b"] = {"class_type": "FluxGuidance", "inputs": {
        "conditioning": ["6_neg", 0], "guidance": guidance,
    }}
    wf["220"] = {"class_type": "InpaintModelConditioning", "inputs": {
        "positive": ["200a", 0],
        "negative": ["200b", 0],
        "vae": ["102", 0],
        "pixels": ["20", 0],
        "mask":  ["20", 1],
        "noise_mask": True,
    }}
    wf["201"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": FLUX_SAMPLER}}
    wf["202"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["100", 0], "scheduler": FLUX_SCHEDULER, "steps": steps, "denoise": 1.0,
    }}
    wf["203"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["204"] = {"class_type": "BasicGuider", "inputs": {
        "model": ["100", 0], "conditioning": ["220", 0],
    }}
    wf["210"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["203", 0], "guider": ["204", 0], "sampler": ["201", 0],
        "sigmas": ["202", 0], "latent_image": ["220", 2],
    }}
    wf["300"] = {"class_type": "VAEDecode", "inputs": {"samples": ["210", 0], "vae": ["102", 0]}}
    wf["301"] = {"class_type": "SaveImage", "inputs": {
        "images": ["300", 0], "filename_prefix": filename_prefix,
    }}
    return wf


# ─── Submission / polling ───────────────────────────────────────────────────

def submit(workflow: dict, client_id: str = "extras") -> str:
    resp = _post("/prompt", {"prompt": workflow, "client_id": client_id})
    if "prompt_id" not in resp:
        raise ComfyError(f"no prompt_id in response: {resp}")
    return resp["prompt_id"]


def get_progress(prompt_id: str) -> dict:
    """Return {state, queue_remaining, history_done}. Cheap to poll."""
    q = _get("/queue")
    running = q.get("queue_running", [])
    pending = q.get("queue_pending", [])
    in_running = any(item[1] == prompt_id for item in running)
    in_pending = any(item[1] == prompt_id for item in pending)
    if in_running:
        return {"state": "running"}
    if in_pending:
        ahead = sum(1 for item in pending if item[1] != prompt_id and
                    pending.index(item) < next(i for i, x in enumerate(pending) if x[1] == prompt_id))
        return {"state": "pending", "ahead": ahead}
    h = _get(f"/history/{prompt_id}")
    if prompt_id in h:
        return {"state": "done"}
    return {"state": "unknown"}


def wait_for(prompt_id: str, timeout: float = 240.0, poll: float = 0.5) -> dict:
    """Block until the prompt finishes; return the history record."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = _get(f"/history/{prompt_id}")
        if prompt_id in h:
            rec = h[prompt_id]
            status = rec.get("status", {})
            if status.get("completed"):
                return rec
            if status.get("status_str") == "error":
                raise ComfyError(f"workflow errored: {status}")
        time.sleep(poll)
    raise ComfyError(f"timed out waiting for {prompt_id}")


def extract_image_files(history_record: dict) -> list[dict]:
    files = []
    for node in history_record.get("outputs", {}).values():
        for img in node.get("images", []):
            files.append({
                "filename": img.get("filename"),
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
    return files


def fetch_image_bytes(filename: str, subfolder: str = "", kind: str = "output") -> bytes:
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": kind})
    with urllib.request.urlopen(f"{SERVER}/view?{qs}", timeout=30) as r:
        return r.read()


def local_path(image_meta: dict) -> Path:
    base = OUTPUT_DIR
    sub = image_meta.get("subfolder", "")
    return base / sub / image_meta["filename"] if sub else base / image_meta["filename"]


def _run(wf: dict, timeout: float = 240.0, seed_for_meta: Optional[int] = None) -> dict:
    """Submit, wait, return paths + first image bytes."""
    pid = submit(wf)
    t0 = time.time()
    rec = wait_for(pid, timeout=timeout)
    elapsed = time.time() - t0
    files = extract_image_files(rec)
    if not files:
        raise ComfyError("no images in output")
    primary = files[0]
    path = local_path(primary)
    data = fetch_image_bytes(
        filename=primary["filename"],
        subfolder=primary["subfolder"],
        kind=primary["type"],
    )
    return {
        "prompt_id": pid,
        "files": files,
        "primary_path": str(path),
        "primary_bytes": data,
        "elapsed_s": round(elapsed, 2),
        "seed": seed_for_meta,
    }


# ─── Legion 5090 offload (FLUX Fill over the 10GbE link) ────────────────────
# FLUX Fill on the V100s is >200s; legion's RTX 5090 does it in ~20s. When legion
# is free we run the SAME fill workflow there (loader nodes swapped to single-GPU
# fp8) and write the result into the local OUTPUT_DIR so the rest of mirage is
# unchanged. Any legion failure / busy state falls back to the local V100 path.
LEGION_SERVER    = "http://10.10.10.1:8188"        # ComfyUI bound to the 10GbE link
LEGION_SSH       = "legion-10g"
LEGION_FILL_UNET = "flux1-fill-dev.safetensors"
LEGION_T5        = "t5xxl_fp8_e4m3fn.safetensors"  # fp8 T5 → headroom on the 24GB 5090
LEGION_BUSY_UTIL = 30                              # sustained % util => training/external load
OFFLOAD_ENABLED  = True                            # kill-switch


def _legion_free() -> bool:
    """True only if legion's ComfyUI is reachable + idle AND the 5090 isn't under
    sustained external (training) load. Gates on UTILIZATION, not memory: ComfyUI
    keeps the fill model warm-resident between requests, which is good, not 'busy'."""
    if not OFFLOAD_ENABLED:
        return False
    try:
        q = json.loads(urllib.request.urlopen(LEGION_SERVER + "/queue", timeout=3).read())
        if q.get("queue_running") or q.get("queue_pending"):
            return False
    except Exception:
        return False
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", LEGION_SSH,
             "for i in 1 2; do nvidia-smi --query-gpu=utilization.gpu "
             "--format=csv,noheader,nounits; sleep 1; done"],
            capture_output=True, text=True, timeout=12)
        utils = [int(x) for x in out.stdout.split()]
        return bool(utils) and min(utils) <= LEGION_BUSY_UTIL
    except Exception:
        return False


def _legionize(wf: dict) -> dict:
    """Swap the MultiGPU GGUF loaders (nodes 100/101/102) for legion's single-GPU
    fp8 equivalents. Same node ids/sockets, so the rest of the workflow is intact."""
    wf = copy.deepcopy(wf)
    wf["100"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": LEGION_FILL_UNET, "weight_dtype": "fp8_e4m3fn_fast"}}
    wf["101"] = {"class_type": "DualCLIPLoader", "inputs": {
        "clip_name1": CLIP_L_NAME, "clip_name2": LEGION_T5, "type": "flux"}}
    wf["102"] = {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}}
    return wf


def _legion_upload(path: Path) -> None:
    boundary = "----legionboundary" + str(int(time.time() * 1000))
    body = Path(path).read_bytes()
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{Path(path).name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + body + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(LEGION_SERVER + "/upload/image", data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    urllib.request.urlopen(req, timeout=30).read()


def _run_on_legion(wf: dict, timeout: float = 120.0, seed_for_meta: Optional[int] = None) -> dict:
    """Run a fill workflow on legion's 5090; mirror _run()'s return shape, writing
    the output PNG into the local OUTPUT_DIR so list_recent/get_image keep working."""
    # push any LoadImage inputs (already staged in the local INPUT_DIR) to legion
    for node in wf.values():
        if node.get("class_type") == "LoadImage":
            src = INPUT_DIR / node["inputs"]["image"]
            if src.is_file():
                _legion_upload(src)
    t0 = time.time()
    req = urllib.request.Request(LEGION_SERVER + "/prompt",
        data=json.dumps({"prompt": wf}).encode(), headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    while True:
        time.sleep(0.5)
        h = json.loads(urllib.request.urlopen(LEGION_SERVER + f"/history/{pid}", timeout=30).read())
        if pid in h:
            if h[pid].get("status", {}).get("status_str") == "error":
                raise ComfyError("legion execution error: " + json.dumps(h[pid]["status"])[:300])
            imgs = []
            for nodeout in h[pid].get("outputs", {}).values():
                imgs += [im for im in nodeout.get("images", []) if im.get("type") == "output"]
            if imgs:
                primary = imgs[0]
                q = urllib.parse.urlencode({"filename": primary["filename"],
                    "subfolder": primary.get("subfolder", ""), "type": primary.get("type", "output")})
                data = urllib.request.urlopen(LEGION_SERVER + "/view?" + q, timeout=60).read()
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                dest = OUTPUT_DIR / primary["filename"]
                if dest.exists():  # don't clobber a different local output
                    dest = OUTPUT_DIR / f"{dest.stem}_legion_{int(time.time()*1000)%100000}{dest.suffix}"
                dest.write_bytes(data)
                meta = {"filename": dest.name, "subfolder": "", "type": "output"}
                return {
                    "prompt_id": pid,
                    "files": [meta],
                    "primary_path": str(dest),
                    "primary_bytes": data,
                    "elapsed_s": round(time.time() - t0, 2),
                    "seed": seed_for_meta,
                    "backend": "legion-5090",
                }
        if time.time() - t0 > timeout:
            raise ComfyError("legion fill timed out")


def _run_fill(wf: dict, timeout: float = 240.0, seed_for_meta: Optional[int] = None) -> dict:
    """Dispatch a FLUX-Fill workflow to legion's 5090 when it's free (~10x faster
    than the V100s), else run locally. Any legion error falls back to local."""
    if _legion_free():
        try:
            return _run_on_legion(_legionize(wf), timeout=min(timeout, 120.0),
                                  seed_for_meta=seed_for_meta)
        except Exception:
            pass  # fall through to the local V100 path on any legion failure
    return _run(wf, timeout=timeout, seed_for_meta=seed_for_meta)


# ─── Public end-to-end API (used by mcp_tools.py) ───────────────────────────

def generate(
    prompt: str,
    negative: str = "",
    width: int = 1024, height: int = 1024,
    steps: int = FLUX_DEFAULT_STEPS,
    cfg: float = FLUX_GUIDANCE_DEFAULT,
    seed: Optional[int] = None,
    filename_prefix: str = "gen",
    timeout: float = 240.0,
) -> dict:
    """Text-to-image. Backend per T2I_BACKEND: "zimage" (Z-Image-Turbo, one card; `steps` and `cfg` are ignored,
    the distilled model's own settings are pinned in build_zimage_t2i) or "flux" (FLUX-dev over both cards;
    `negative` is accepted for API compat but ignored, FLUX-dev has no classifier-free guidance, `cfg` is
    reinterpreted as FluxGuidance).
    """
    if T2I_BACKEND == "zimage":
        wf = build_zimage_t2i(prompt=prompt, width=width, height=height, seed=seed,
                              filename_prefix=filename_prefix)
        return _run(wf, timeout=timeout, seed_for_meta=wf["3"]["inputs"]["seed"])
    wf = build_flux_t2i(
        prompt=prompt,
        width=width, height=height,
        steps=steps, guidance=cfg, seed=seed,
        filename_prefix=filename_prefix,
    )
    # Seed actually lives on node "203" (RandomNoise) for the FLUX flow.
    return _run(wf, timeout=timeout, seed_for_meta=wf["203"]["inputs"]["noise_seed"])


def upload_image(local_path: str | Path, name: Optional[str] = None) -> str:
    """Upload a local image to ComfyUI's input dir. Returns the filename it ended up with."""
    p = Path(local_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    name = name or p.name
    boundary = "----comfyboundary" + str(int(time.time() * 1000))
    with open(p, "rb") as fh:
        body = fh.read()
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + body + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        SERVER + "/upload/image",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    return resp.get("name", name)


def img2img(
    image_path: str | Path, prompt: str,
    negative: str = "", denoise: float = 0.65,
    steps: int = FLUX_DEFAULT_STEPS, cfg: float = FLUX_GUIDANCE_DEFAULT,
    seed: Optional[int] = None,
    filename_prefix: str = "i2i",
    timeout: float = 240.0,
) -> dict:
    """Image-to-image (FLUX-dev). `negative` ignored. `cfg` → FluxGuidance."""
    image_name = upload_image(image_path)
    wf = build_flux_i2i(
        prompt=prompt, image_name=image_name,
        denoise=denoise, steps=steps, guidance=cfg, seed=seed,
        filename_prefix=filename_prefix,
    )
    return _run(wf, timeout=timeout, seed_for_meta=wf["203"]["inputs"]["noise_seed"])


def inpaint(
    image_path: str | Path, mask_path: str | Path, prompt: str,
    negative: str = "", denoise: float = 1.0,
    steps: int = FLUX_DEFAULT_STEPS, cfg: float = 30.0,
    seed: Optional[int] = None,
    grow_mask_by: int = 6,
    filename_prefix: str = "inpaint",
    timeout: float = 240.0,
) -> dict:
    """Inpaint with flux-fill. `denoise` unused (flux-fill always runs full); kept for API compat.
    `cfg` is FluxGuidance — flux-fill uses higher values (~30) than flux-dev (~3.5).
    """
    image_name = upload_image(image_path)
    mask_name = upload_image(mask_path)
    wf = build_flux_fill_inpaint(
        prompt=prompt, image_name=image_name, mask_name=mask_name,
        steps=steps, guidance=cfg, seed=seed,
        grow_mask_by=grow_mask_by,
        filename_prefix=filename_prefix,
    )
    return _run_fill(wf, timeout=timeout, seed_for_meta=wf["203"]["inputs"]["noise_seed"])


# ─── Upscale (RealESRGAN, model-agnostic) ───────────────────────────────────

def build_upscale(image_name: str, filename_prefix: str = "upscale") -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {
            "upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {
            "images": ["3", 0], "filename_prefix": filename_prefix}},
    }


def upscale(
    image_path: str | Path,
    filename_prefix: str = "upscale",
    timeout: float = 180.0,
) -> dict:
    image_name = upload_image(image_path)
    wf = build_upscale(image_name=image_name, filename_prefix=filename_prefix)
    return _run(wf, timeout=timeout, seed_for_meta=None)


# ─── Listing / fetching / deleting outputs ──────────────────────────────────

def list_recent(limit: int = 20) -> list[dict]:
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(
        (p for p in OUTPUT_DIR.glob("*.png") if p.is_file() and p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    return [{"filename": p.name, "size": p.stat().st_size,
             "mtime": p.stat().st_mtime, "path": str(p)} for p in files]


def get_image_bytes(filename: str) -> bytes:
    if "/" in filename or ".." in filename:
        raise ValueError(f"bad filename: {filename!r}")
    p = OUTPUT_DIR / filename
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    return p.read_bytes()


def delete_image(filename: str) -> dict:
    if "/" in filename or ".." in filename:
        raise ValueError(f"bad filename: {filename!r}")
    p = OUTPUT_DIR / filename
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    size = p.stat().st_size
    p.unlink()
    return {"filename": filename, "size": size, "deleted": True}


# ─── Structured prompt composition ──────────────────────────────────────────

def compose_prompt(prompt: Optional[str], structure: Optional[dict]) -> str:
    """Marshal an optional structured dict + freeform string into one prompt.

    FLUX's T5-XXL encoder handles long natural-language prompts well — favor
    verbose declarative sentences over keyword soup.
    """
    parts: list[str] = []
    if structure:
        subjects = structure.get("subjects") or []
        if subjects:
            subj_chunks = []
            for s in subjects:
                d = s.get("description", "")
                pose = s.get("pose"); pos = s.get("position")
                bits = [b for b in (d, pose, f"({pos})" if pos else None) if b]
                if bits:
                    subj_chunks.append(" ".join(bits))
            if subj_chunks:
                parts.append(", ".join(subj_chunks))

        if structure.get("style"):
            parts.append(f"style: {structure['style']}")
        if structure.get("lighting"):
            parts.append(f"lighting: {structure['lighting']}")

        cam = structure.get("camera") or {}
        cam_bits = [v for v in (cam.get("angle"), cam.get("distance"), cam.get("lens")) if v]
        if cam_bits:
            parts.append("camera: " + ", ".join(cam_bits))

        palette = structure.get("color_palette") or []
        if palette:
            parts.append("color palette: " + ", ".join(palette))

        if structure.get("mood"):
            parts.append(f"mood: {structure['mood']}")
        if structure.get("composition"):
            parts.append(f"composition: {structure['composition']}")
        if structure.get("background"):
            parts.append(f"background: {structure['background']}")

    if prompt:
        parts.append(prompt)

    return ". ".join(parts)


# ─── Inpaint by bbox (no mask file) ─────────────────────────────────────────

def _make_bbox_mask(source_image_path: str | Path, bbox: list[int]) -> Path:
    """Create a black PNG with a white rectangle at `bbox`, sized to match the source.
    Returns the temp mask path. Caller is responsible for cleanup if desired."""
    from PIL import Image, ImageDraw
    src = Image.open(source_image_path)
    w, h = src.size
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, min(w, int(x1))), max(0, min(w, int(x2)))))
    y1, y2 = sorted((max(0, min(h, int(y1))), max(0, min(h, int(y2)))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise ValueError(f"bbox too small after clamping: ({x1},{y1},{x2},{y2}) on {w}x{h} image")
    mask = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill="white")
    tmp = Path("/tmp") / f"mirage_bbox_mask_{int(time.time()*1000)}_{random.randint(0,9999)}.png"
    mask.save(tmp)
    return tmp


def inpaint_region(
    image_path: str | Path, bbox: list[int], prompt: str,
    negative: str = "",
    steps: int = FLUX_DEFAULT_STEPS, cfg: float = 30.0,
    seed: Optional[int] = None,
    grow_mask_by: int = 12,
    filename_prefix: str = "region",
    timeout: float = 240.0,
) -> dict:
    mask_path = _make_bbox_mask(image_path, bbox)
    try:
        return inpaint(
            image_path=image_path, mask_path=mask_path, prompt=prompt,
            negative=negative, steps=steps, cfg=cfg, seed=seed,
            grow_mask_by=grow_mask_by, filename_prefix=filename_prefix,
            timeout=timeout,
        )
    finally:
        try: mask_path.unlink(missing_ok=True)
        except Exception: pass


# ─── Outpaint ───────────────────────────────────────────────────────────────

def _outpaint_offsets(direction: str, pixels: int) -> tuple[int, int, int, int]:
    if direction == "right":  return (0, 0, pixels, 0)
    if direction == "left":   return (pixels, 0, 0, 0)
    if direction == "up":     return (0, pixels, 0, 0)
    if direction == "down":   return (0, 0, 0, pixels)
    if direction == "all":    half = pixels // 2; return (half, half, half, half)
    raise ValueError(f"bad direction: {direction!r}")


def outpaint(
    image_path: str | Path, direction: str, pixels: int, prompt: str,
    negative: str = "",
    steps: int = FLUX_DEFAULT_STEPS, cfg: float = 30.0,
    seed: Optional[int] = None,
    feathering: int = 32,
    filename_prefix: str = "outpaint",
    timeout: float = 240.0,
) -> dict:
    left, top, right, bottom = _outpaint_offsets(direction, pixels)
    image_name = upload_image(image_path)
    wf = build_flux_fill_outpaint(
        prompt=prompt, image_name=image_name,
        left=left, top=top, right=right, bottom=bottom,
        feathering=feathering,
        steps=steps, guidance=cfg, seed=seed,
        filename_prefix=filename_prefix,
    )
    return _run_fill(wf, timeout=timeout, seed_for_meta=wf["203"]["inputs"]["noise_seed"])


# ─── Vary (img2img with neutral defaults) ───────────────────────────────────

VARY_DEFAULT_PROMPT = "high quality, detailed, same scene"


def vary(
    image_path: str | Path,
    strength: float = 0.5,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    filename_prefix: str = "vary",
    timeout: float = 240.0,
) -> dict:
    return img2img(
        image_path=image_path,
        prompt=prompt or VARY_DEFAULT_PROMPT,
        denoise=max(0.0, min(1.0, strength)),
        steps=FLUX_DEFAULT_STEPS, cfg=FLUX_GUIDANCE_DEFAULT, seed=seed,
        filename_prefix=filename_prefix,
        timeout=timeout,
    )
