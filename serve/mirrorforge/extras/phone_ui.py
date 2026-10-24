"""Mobile-first web UI in front of ComfyUI. Listens on 0.0.0.0:8189.

Backend: FLUX-dev (Q8 GGUF) + FLUX-fill, split across dual V100s. See comfy_client.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comfy_client as comfy

STATIC_DIR = HERE / "static"
OUTPUT_DIR = comfy.OUTPUT_DIR

app = FastAPI(title="MirrorForge ImageGen")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Request models (FLUX-dev defaults) ─────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    negative: Optional[str] = None   # Accepted for API compat; FLUX-dev ignores it.
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=20, ge=5, le=60)
    cfg: float = Field(default=3.5, ge=0.0, le=30.0)
    seed: Optional[int] = None


class Img2ImgRequest(BaseModel):
    image_filename: str           # bare filename in output/
    prompt: str
    denoise: float = Field(default=0.65, ge=0.0, le=1.0)
    steps: int = Field(default=20, ge=5, le=60)
    cfg: float = Field(default=3.5, ge=0.0, le=30.0)
    seed: Optional[int] = None


class EditRequest(BaseModel):
    image_filename: str
    instruction: str
    change_strength: Literal["subtle", "moderate", "strong"] = "moderate"
    seed: Optional[int] = None


class UpscaleRequest(BaseModel):
    image_filename: str


def _resolve_output_image(filename: str) -> Path:
    """Resolve a bare filename to a path inside OUTPUT_DIR. Rejects traversal."""
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    p = OUTPUT_DIR / filename
    if not p.is_file():
        raise HTTPException(404, f"image not found: {filename}")
    return p


def _gen_response(result: dict) -> dict:
    primary = result["files"][0]
    return {
        "prompt_id": result["prompt_id"],
        "filename": primary["filename"],
        "subfolder": primary["subfolder"],
        "path": result["primary_path"],
        "elapsed_s": result["elapsed_s"],
        "seed": result["seed"],
    }


# ─── Pages / static ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/manifest.webmanifest")
async def manifest():
    return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")


# ─── Image-gen endpoints ────────────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    if not comfy.server_up():
        raise HTTPException(503, "ComfyUI backend not reachable on :8188")
    try:
        result = comfy.generate(
            prompt=req.prompt,
            negative=req.negative or "",
            width=req.width, height=req.height,
            steps=req.steps, cfg=req.cfg,
            seed=req.seed,
            filename_prefix="phone",
        )
    except comfy.ComfyError as e:
        raise HTTPException(500, str(e))
    return JSONResponse(_gen_response(result))


@app.post("/api/img2img")
async def api_img2img(req: Img2ImgRequest):
    if not comfy.server_up():
        raise HTTPException(503, "ComfyUI backend not reachable on :8188")
    src = _resolve_output_image(req.image_filename)
    try:
        result = comfy.img2img(
            image_path=src,
            prompt=req.prompt,
            denoise=req.denoise,
            steps=req.steps, cfg=req.cfg,
            seed=req.seed,
            filename_prefix="phone_i2i",
        )
    except comfy.ComfyError as e:
        raise HTTPException(500, str(e))
    return JSONResponse(_gen_response(result))


@app.post("/api/edit")
async def api_edit(req: EditRequest):
    """Natural-language edit. Maps change_strength → denoise and runs img2img."""
    if not comfy.server_up():
        raise HTTPException(503, "ComfyUI backend not reachable on :8188")
    src = _resolve_output_image(req.image_filename)
    strength_to_denoise = {"subtle": 0.4, "moderate": 0.6, "strong": 0.8}
    try:
        result = comfy.img2img(
            image_path=src,
            prompt=req.instruction,
            denoise=strength_to_denoise[req.change_strength],
            steps=20, cfg=3.5, seed=req.seed,
            filename_prefix="phone_edit",
        )
    except comfy.ComfyError as e:
        raise HTTPException(500, str(e))
    return JSONResponse(_gen_response(result))


@app.post("/api/upscale")
async def api_upscale(req: UpscaleRequest):
    if not comfy.server_up():
        raise HTTPException(503, "ComfyUI backend not reachable on :8188")
    src = _resolve_output_image(req.image_filename)
    try:
        result = comfy.upscale(image_path=src, filename_prefix="phone_upscale")
    except comfy.ComfyError as e:
        raise HTTPException(500, str(e))
    return JSONResponse(_gen_response(result))


# ─── Gallery / static image serving ─────────────────────────────────────────

@app.get("/api/image/{filename}")
async def api_image(filename: str, subfolder: str = ""):
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    try:
        data = comfy.fetch_image_bytes(filename, subfolder=subfolder, kind="output")
    except Exception as e:
        raise HTTPException(404, f"image not found: {e}")
    ext = filename.rsplit(".", 1)[-1].lower()
    mt = {"webp": "image/webp", "gif": "image/gif", "webm": "video/webm",
          "mp4": "video/mp4", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return Response(content=data, media_type=mt)


@app.get("/api/recent")
async def api_recent(limit: int = 24):
    """List recent PNGs in the output dir, newest first. Includes mtime + size."""
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(
        (p for ext in ("*.png", "*.webp", "*.webm", "*.mp4")
           for p in OUTPUT_DIR.glob(ext) if p.is_file() and p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    return [{"filename": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in files]


@app.get("/api/health")
async def api_health():
    return {"comfyui_up": comfy.server_up()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8189, log_level="info")
