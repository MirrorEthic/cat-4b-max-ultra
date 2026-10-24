"""Shared MCP tool definitions and dispatch.

Both transports (stdio in mcp_server.py, HTTP in mcp_http.py) import from this
module so there's one source of truth for the tool surface.

A "tool result" here is a list of MCP content dicts:
    [{"type": "text", "text": "..."}, {"type": "image", "data": "<base64>", "mimeType": "image/png"}]

Backend: FLUX-dev (Q8 GGUF) + FLUX-fill (Q8 GGUF) on dual V100s. See ../models
and ~/.claude/projects/-/memory/hw_v100_thermal.md for the device split.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comfy_client as comfy


# ─── Defaults that match the FLUX backend in comfy_client.py ────────────────

DEFAULT_STEPS_T2I    = 20      # FLUX-dev typically uses ~20 steps
DEFAULT_STEPS_FILL   = 20      # flux-fill same
DEFAULT_GUIDANCE_DEV = 3.5     # FluxGuidance for flux-dev (t2i / i2i / vary / edit)
DEFAULT_GUIDANCE_FILL = 30.0   # flux-fill needs much higher guidance


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_PROMPT_SCHEMA = {
    "type": "object",
    "description": (
        "Optional structured prompt. If provided, fields are marshaled into a "
        "rich text prompt. Mix with `prompt` (string) for freeform extras. "
        "FLUX's T5-XXL handles long natural-language descriptions well — be specific."
    ),
    "properties": {
        "subjects": {
            "type": "array",
            "description": "Subjects in the scene",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "position": {"type": "string", "description": "e.g. 'foreground left', 'background center'"},
                    "pose": {"type": "string"},
                },
            },
        },
        "style":         {"type": "string", "description": "e.g. 'oil painting', '35mm film photograph', 'isometric 3D render'"},
        "lighting":      {"type": "string", "description": "e.g. 'golden hour', 'rim light from left, dim ambient'"},
        "camera": {
            "type": "object",
            "properties": {
                "angle":    {"type": "string", "description": "e.g. 'low angle', 'eye level'"},
                "distance": {"type": "string", "description": "e.g. 'close-up', 'wide shot'"},
                "lens":     {"type": "string", "description": "e.g. '35mm', '85mm portrait'"},
            },
        },
        "color_palette": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Colors (names or HEX codes like '#1f2937')",
        },
        "mood":        {"type": "string"},
        "composition": {"type": "string", "description": "e.g. 'rule of thirds', 'centered symmetry'"},
        "background":  {"type": "string"},
    },
}


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "generate_image",
        "description": (
            "Generate an image with FLUX-dev on dual V100s. "
            "Returns the file path AND the image inline. Typical ~40s @ 1024² (20 steps). "
            "Either `prompt` (string), `structure` (dict), or both. "
            "FLUX-dev does not use classifier-free guidance — `cfg` is reinterpreted as FluxGuidance "
            "(typical 2.5-5.0). `negative` is ignored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt":    {"type": "string", "description": "Positive prompt (freeform text)"},
                "structure": STRUCTURED_PROMPT_SCHEMA,
                "negative":  {"type": "string", "description": "Accepted but ignored on FLUX-dev"},
                "width":     {"type": "integer", "minimum": 256, "maximum": 2048, "default": 1024},
                "height":    {"type": "integer", "minimum": 256, "maximum": 2048, "default": 1024},
                "steps":     {"type": "integer", "minimum": 5, "maximum": 60, "default": DEFAULT_STEPS_T2I},
                "cfg":       {"type": "number", "minimum": 0.0, "maximum": 30.0, "default": DEFAULT_GUIDANCE_DEV,
                              "description": "FluxGuidance value (typical 2.5-5.0)"},
                "seed":      {"type": "integer"},
            },
        },
    },
    {
        "name": "img2img",
        "description": (
            "Re-imagine an existing image with FLUX-dev. "
            "denoise ∈ [0,1]: 0.4=subtle, 0.7=strong, 1.0=full rewrite. "
            "`image_path` may be a bare filename in output/ or an absolute path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompt":     {"type": "string"},
                "negative":   {"type": "string", "description": "Ignored on FLUX-dev"},
                "denoise":    {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.65},
                "steps":      {"type": "integer", "minimum": 5, "maximum": 60, "default": DEFAULT_STEPS_T2I},
                "cfg":        {"type": "number", "minimum": 0.0, "maximum": 30.0, "default": DEFAULT_GUIDANCE_DEV},
                "seed":       {"type": "integer"},
            },
            "required": ["image_path", "prompt"],
        },
    },
    {
        "name": "edit_image",
        "description": (
            "Natural-language edit of an existing image (no mask required), via FLUX-dev img2img. "
            "Best for stylistic / atmosphere changes — for localized object edits use `inpaint_region`. "
            "Internally img2img with smart denoise based on `change_strength`."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path":      {"type": "string"},
                "instruction":     {"type": "string", "description": "What to change, in natural language"},
                "change_strength": {
                    "type": "string",
                    "enum": ["subtle", "moderate", "strong"],
                    "default": "moderate",
                    "description": "subtle ≈ denoise 0.4, moderate ≈ 0.6, strong ≈ 0.8",
                },
                "seed": {"type": "integer"},
            },
            "required": ["image_path", "instruction"],
        },
    },
    {
        "name": "inpaint",
        "description": (
            "Replace a masked region with FLUX-fill. White mask = repaint, black = keep. "
            "Mask must be same dimensions as image. flux-fill uses high guidance (~30) by default — "
            "this is normal, not a typo. "
            "Auto-offloads to legion's RTX 5090 over the 10GbE link when legion is free (~20s); only when "
            "legion is busy (training) does it fall back to the local V100s (~160-260s, claude.ai may time "
            "out — completes in background, image still lands in ~/comfyui/output/). "
            "Prefer `edit_image` or `img2img` (~40s on flux-dev) for most revisions — flux-fill is only "
            "needed for surgical edits where pixel-perfect preservation of unmasked regions matters "
            "(precise text replacement, removing one object cleanly, etc.). Use sparingly. "
            "REMOTE CLIENTS: image_path and mask_path each accept a `data:image/png;base64,<...>` URL, "
            "so you can author the mask locally and pass it inline — no filesystem access on the GPU host "
            "needed. For a simple rectangular region, prefer `inpaint_region` (pass a bbox, no mask at all)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path":   {"type": "string", "description": "Filename in output/, absolute path, or a data:image/...;base64 URL"},
                "mask_path":    {"type": "string", "description": "Same accepted forms as image_path. White = repaint, black = keep."},
                "prompt":       {"type": "string"},
                "negative":     {"type": "string", "description": "Ignored on FLUX-fill"},
                "steps":        {"type": "integer", "minimum": 5, "maximum": 60, "default": DEFAULT_STEPS_FILL},
                "cfg":          {"type": "number", "minimum": 1.0, "maximum": 50.0, "default": DEFAULT_GUIDANCE_FILL,
                                 "description": "FluxGuidance for flux-fill — typical 20-40"},
                "seed":         {"type": "integer"},
                "grow_mask_by": {"type": "integer", "minimum": 0, "maximum": 32, "default": 6},
            },
            "required": ["image_path", "mask_path", "prompt"],
        },
    },
    {
        "name": "inpaint_region",
        "description": (
            "Inpaint a rectangular region with FLUX-fill, specified by bounding box — no mask file needed. "
            "`bbox` is [x1, y1, x2, y2] in pixel coordinates (origin top-left). "
            "The bbox is converted to a white-on-black mask internally. "
            "SLOW on this hardware: ~160s warm-cache, ~260s cold-cache (claude.ai client may time out — "
            "image still saves to ~/comfyui/output/ in the background). "
            "Prefer `edit_image` or `img2img` (~40s) for most revisions; reserve this for precision "
            "spatial edits where unmasked regions must stay pixel-identical. Use sparingly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path":   {"type": "string"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 4, "maxItems": 4,
                    "description": "[x1, y1, x2, y2]",
                },
                "prompt":       {"type": "string"},
                "negative":     {"type": "string", "description": "Ignored"},
                "steps":        {"type": "integer", "minimum": 5, "maximum": 60, "default": DEFAULT_STEPS_FILL},
                "cfg":          {"type": "number", "minimum": 1.0, "maximum": 50.0, "default": DEFAULT_GUIDANCE_FILL},
                "seed":         {"type": "integer"},
                "grow_mask_by": {"type": "integer", "minimum": 0, "maximum": 64, "default": 12},
            },
            "required": ["image_path", "bbox", "prompt"],
        },
    },
    {
        "name": "outpaint",
        "description": (
            "Extend the canvas of an image in a given direction with FLUX-fill. "
            "`direction` ∈ {up, down, left, right, all}. `pixels` is how far to extend (per side for 'all'). "
            "SLOW on this hardware: ~160s warm-cache, ~260s cold-cache (claude.ai client may time out — "
            "image still saves to ~/comfyui/output/ in the background). "
            "Genuinely needed for canvas extension — there's no flux-dev shortcut for outpainting. "
            "Use sparingly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "direction":  {"type": "string", "enum": ["up", "down", "left", "right", "all"]},
                "pixels":     {"type": "integer", "minimum": 32, "maximum": 1024, "default": 256},
                "prompt":     {"type": "string", "description": "What should appear in the new area"},
                "negative":   {"type": "string", "description": "Ignored"},
                "steps":      {"type": "integer", "minimum": 5, "maximum": 60, "default": DEFAULT_STEPS_FILL},
                "cfg":        {"type": "number", "minimum": 1.0, "maximum": 50.0, "default": DEFAULT_GUIDANCE_FILL},
                "seed":       {"type": "integer"},
                "feathering": {"type": "integer", "minimum": 0, "maximum": 96, "default": 32,
                              "description": "Feather the join between old and new regions"},
            },
            "required": ["image_path", "direction", "prompt"],
        },
    },
    {
        "name": "vary",
        "description": (
            "Generate a variation of an existing image via FLUX-dev img2img. `strength` ∈ [0,1]: "
            "0.3=very similar, 0.6=clear variation, 0.9=loose interpretation. "
            "Optional `prompt` steers the variation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "strength":   {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                "prompt":     {"type": "string", "description": "Optional — steers the variation"},
                "seed":       {"type": "integer"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "upscale",
        "description": "4× super-resolution via RealESRGAN. ~7-8s for 1024→4096.",
        "inputSchema": {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
            "required": ["image_path"],
        },
    },
    {
        "name": "list_recent",
        "description": "List the most recent generated images (newest first).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
        },
    },
    {
        "name": "get_image",
        "description": "Fetch a previously generated image by filename. Returns inline.",
        "inputSchema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "delete_image",
        "description": "Delete a generated image (only files in the output dir).",
        "inputSchema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Content builders
# ─────────────────────────────────────────────────────────────────────────────

def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def image_content(image_bytes: bytes) -> dict:
    return {
        "type": "image",
        "data": base64.b64encode(image_bytes).decode("ascii"),
        "mimeType": "image/png",
    }


def _gen_meta(r: dict, label: str) -> str:
    seed_part = f" | seed: {r['seed']}" if r.get("seed") is not None else ""
    return f"{label}: {r['primary_path']}\nelapsed: {r['elapsed_s']}s{seed_part}"


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution: bare filename → output/, else absolute, else ~/expand
# ─────────────────────────────────────────────────────────────────────────────

def _materialize_data_url(p: str) -> str:
    """Decode a `data:image/...;base64,<data>` URL into a file under INPUT_DIR and
    return the path. Lets remote MCP clients (claude.ai) pass image/mask *content*
    inline instead of a server-side path they can't author. Content-addressed name
    (sha1) → idempotent, dedups repeats."""
    header, _, b64 = p.partition(",")
    if not b64:
        raise ValueError("malformed data URL (no base64 payload)")
    raw = base64.b64decode(b64)
    ext = ".png"
    h = header.lower()
    if "jpeg" in h or "jpg" in h:
        ext = ".jpg"
    elif "webp" in h:
        ext = ".webp"
    comfy.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = comfy.INPUT_DIR / f"mcp_upload_{hashlib.sha1(raw).hexdigest()[:16]}{ext}"
    if not dest.is_file():
        dest.write_bytes(raw)
    return str(dest.resolve())


def resolve_image_path(p: str) -> str:
    """Accept a bare filename in output/, an absolute path, OR a `data:image/...;base64,`
    URL (materialized server-side). The data-URL form is what makes the image tools
    drivable from a remote client with no filesystem access on this host."""
    if not p:
        raise ValueError("empty path")
    if p.startswith("data:"):
        return _materialize_data_url(p)
    pp = Path(p).expanduser()
    if pp.is_file():
        return str(pp.resolve())
    candidate = comfy.OUTPUT_DIR / p
    if candidate.is_file():
        return str(candidate.resolve())
    raise FileNotFoundError(f"image not found: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatch
# ─────────────────────────────────────────────────────────────────────────────

async def dispatch(name: str, args: dict) -> list[dict]:
    """Run a tool by name. Returns MCP content list."""

    if name == "generate_image":
        prompt = comfy.compose_prompt(args.get("prompt"), args.get("structure"))
        if not prompt:
            raise ValueError("generate_image requires `prompt` and/or `structure`")
        r = await asyncio.to_thread(
            comfy.generate,
            prompt=prompt,
            negative=args.get("negative") or "",
            width=args.get("width", 1024),
            height=args.get("height", 1024),
            steps=args.get("steps", DEFAULT_STEPS_T2I),
            cfg=args.get("cfg", DEFAULT_GUIDANCE_DEV),
            seed=args.get("seed"),
            filename_prefix="mcp",
        )
        return [text_content(_gen_meta(r, "generated")), image_content(r["primary_bytes"])]

    if name == "img2img":
        path = resolve_image_path(args["image_path"])
        r = await asyncio.to_thread(
            comfy.img2img,
            image_path=path, prompt=args["prompt"],
            negative=args.get("negative") or "",
            denoise=args.get("denoise", 0.65),
            steps=args.get("steps", DEFAULT_STEPS_T2I),
            cfg=args.get("cfg", DEFAULT_GUIDANCE_DEV),
            seed=args.get("seed"),
            filename_prefix="mcp_i2i",
        )
        return [text_content(_gen_meta(r, "img2img")), image_content(r["primary_bytes"])]

    if name == "edit_image":
        path = resolve_image_path(args["image_path"])
        strength_to_denoise = {"subtle": 0.4, "moderate": 0.6, "strong": 0.8}
        denoise = strength_to_denoise[args.get("change_strength", "moderate")]
        r = await asyncio.to_thread(
            comfy.img2img,
            image_path=path,
            prompt=args["instruction"],
            negative="",
            denoise=denoise, steps=DEFAULT_STEPS_T2I, cfg=DEFAULT_GUIDANCE_DEV,
            seed=args.get("seed"),
            filename_prefix="mcp_edit",
        )
        return [text_content(_gen_meta(r, "edit_image")), image_content(r["primary_bytes"])]

    if name == "inpaint":
        img = resolve_image_path(args["image_path"])
        mask = resolve_image_path(args["mask_path"])
        r = await asyncio.to_thread(
            comfy.inpaint,
            image_path=img, mask_path=mask, prompt=args["prompt"],
            negative="",
            steps=args.get("steps", DEFAULT_STEPS_FILL),
            cfg=args.get("cfg", DEFAULT_GUIDANCE_FILL),
            seed=args.get("seed"),
            grow_mask_by=args.get("grow_mask_by", 6),
            filename_prefix="mcp_inpaint",
        )
        return [text_content(_gen_meta(r, "inpaint")), image_content(r["primary_bytes"])]

    if name == "inpaint_region":
        img = resolve_image_path(args["image_path"])
        bbox = args["bbox"]
        r = await asyncio.to_thread(
            comfy.inpaint_region,
            image_path=img, bbox=bbox, prompt=args["prompt"],
            negative="",
            steps=args.get("steps", DEFAULT_STEPS_FILL),
            cfg=args.get("cfg", DEFAULT_GUIDANCE_FILL),
            seed=args.get("seed"),
            grow_mask_by=args.get("grow_mask_by", 12),
            filename_prefix="mcp_region",
        )
        return [text_content(_gen_meta(r, "inpaint_region")), image_content(r["primary_bytes"])]

    if name == "outpaint":
        img = resolve_image_path(args["image_path"])
        r = await asyncio.to_thread(
            comfy.outpaint,
            image_path=img,
            direction=args["direction"],
            pixels=args.get("pixels", 256),
            prompt=args["prompt"],
            negative="",
            steps=args.get("steps", DEFAULT_STEPS_FILL),
            cfg=args.get("cfg", DEFAULT_GUIDANCE_FILL),
            seed=args.get("seed"),
            feathering=args.get("feathering", 32),
            filename_prefix="mcp_outpaint",
        )
        return [text_content(_gen_meta(r, "outpaint")), image_content(r["primary_bytes"])]

    if name == "vary":
        path = resolve_image_path(args["image_path"])
        r = await asyncio.to_thread(
            comfy.vary,
            image_path=path,
            strength=args.get("strength", 0.5),
            prompt=args.get("prompt"),
            seed=args.get("seed"),
            filename_prefix="mcp_vary",
        )
        return [text_content(_gen_meta(r, "vary")), image_content(r["primary_bytes"])]

    if name == "upscale":
        path = resolve_image_path(args["image_path"])
        r = await asyncio.to_thread(comfy.upscale, image_path=path, filename_prefix="mcp_upscale")
        return [text_content(_gen_meta(r, "upscale")), image_content(r["primary_bytes"])]

    if name == "list_recent":
        items = await asyncio.to_thread(comfy.list_recent, limit=args.get("limit", 20))
        if not items:
            return [text_content("(no images in output dir)")]
        lines = [f"{i['filename']}  {i['size']/1024:.0f}KB  {i['path']}" for i in items]
        return [text_content("\n".join(lines))]

    if name == "get_image":
        data = await asyncio.to_thread(comfy.get_image_bytes, args["filename"])
        return [
            text_content(f"image: {comfy.OUTPUT_DIR / args['filename']} ({len(data)} bytes)"),
            image_content(data),
        ]

    if name == "delete_image":
        res = await asyncio.to_thread(comfy.delete_image, args["filename"])
        return [text_content(json.dumps(res))]

    raise ValueError(f"unknown tool: {name}")
