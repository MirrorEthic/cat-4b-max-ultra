"""Generate FLUX workflow templates for the ComfyUI UI.

Writes UI-format .json files into ~/comfyui/user/default/workflows/ that show up in
the Workflow menu. Each one mirrors the MCP backend's workflows (transformer→cuda:1,
encoders+VAE→cuda:0), but tunable from the graph for one-off experiments.

Run: python3 build_flux_templates.py
"""
from __future__ import annotations
import json, random
from pathlib import Path

OUT_DIR = Path.home() / "comfyui" / "user" / "default" / "workflows"

FLUX_UNET       = "flux1-dev-Q8_0.gguf"
FLUX_FILL_UNET  = "flux1-fill-dev-Q8_0.gguf"
FLUX_VAE        = "ae.safetensors"
T5_NAME         = "t5xxl_fp16.safetensors"
CLIP_L_NAME     = "clip_l.safetensors"
UPSCALE_MODEL   = "RealESRGAN_x4plus.pth"

TX_DEV = "cuda:1"   # transformer (heavy)
ENC_DEV = "cuda:0"  # encoders + VAE (light) — see [[hw-v100-thermal]]


def _node(nid, ntype, pos, size, widgets, inputs=None, outputs=None):
    return {
        "id": nid, "type": ntype, "pos": list(pos), "size": list(size),
        "flags": {}, "order": nid - 1, "mode": 0,
        "inputs": inputs or [], "outputs": outputs or [],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }


def _link(lid, src_node, src_slot, dst_node, dst_slot, ltype):
    return [lid, src_node, src_slot, dst_node, dst_slot, ltype]


def _inp(name, ntype, link=None):
    return {"name": name, "type": ntype, "link": link}


def _out(name, ntype, links=None, slot_index=0):
    return {"name": name, "type": ntype, "links": list(links or []), "slot_index": slot_index}


def _wrap(nodes, links, *, last_node_id=None, last_link_id=None):
    return {
        "last_node_id": last_node_id if last_node_id is not None else max(n["id"] for n in nodes),
        "last_link_id": last_link_id if last_link_id is not None else max((l[0] for l in links), default=0),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.7, "offset": [0, 0]}},
        "version": 0.4,
    }


# ─── Templates ──────────────────────────────────────────────────────────────

def t2i_workflow(width=1024, height=1024, *, prompt="a photograph of a glowing crystal on a wooden table, soft window light", steps=20, guidance=3.5, seed=None):
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    # Layout (left → right):  loaders  →  prompt+latent  →  sampler stack  →  decode/save
    nodes = [
        _node(1, "UnetLoaderGGUFMultiGPU", (40, 40), (340, 110),
              [FLUX_UNET, TX_DEV],
              outputs=[_out("MODEL", "MODEL", [1, 2])]),
        _node(2, "DualCLIPLoaderMultiGPU", (40, 180), (340, 150),
              [CLIP_L_NAME, T5_NAME, "flux", ENC_DEV],
              outputs=[_out("CLIP", "CLIP", [3])]),
        _node(3, "VAELoaderMultiGPU", (40, 360), (340, 80),
              [FLUX_VAE, ENC_DEV],
              outputs=[_out("VAE", "VAE", [9])]),
        _node(4, "CLIPTextEncode", (420, 40), (340, 160),
              [prompt],
              inputs=[_inp("clip", "CLIP", 3)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [4])]),
        _node(5, "EmptySD3LatentImage", (420, 240), (340, 100),
              [width, height, 1],
              outputs=[_out("LATENT", "LATENT", [5])]),
        _node(6, "FluxGuidance", (790, 40), (300, 80),
              [guidance],
              inputs=[_inp("conditioning", "CONDITIONING", 4)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [6])]),
        _node(7, "KSamplerSelect", (790, 160), (300, 60),
              ["euler"],
              outputs=[_out("SAMPLER", "SAMPLER", [7])]),
        _node(8, "BasicScheduler", (790, 250), (300, 110),
              ["simple", steps, 1.0],
              inputs=[_inp("model", "MODEL", 1)],
              outputs=[_out("SIGMAS", "SIGMAS", [8])]),
        _node(9, "RandomNoise", (790, 380), (300, 80),
              [seed, "randomize"],
              outputs=[_out("NOISE", "NOISE", [10])]),
        _node(10, "BasicGuider", (1120, 40), (260, 80),
              [],
              inputs=[_inp("model", "MODEL", 2), _inp("conditioning", "CONDITIONING", 6)],
              outputs=[_out("GUIDER", "GUIDER", [11])]),
        _node(11, "SamplerCustomAdvanced", (1120, 160), (320, 200),
              [],
              inputs=[
                  _inp("noise", "NOISE", 10),
                  _inp("guider", "GUIDER", 11),
                  _inp("sampler", "SAMPLER", 7),
                  _inp("sigmas", "SIGMAS", 8),
                  _inp("latent_image", "LATENT", 5),
              ],
              outputs=[_out("output", "LATENT", [12]), _out("denoised_output", "LATENT", [])]),
        _node(12, "VAEDecode", (1470, 40), (260, 80),
              [],
              inputs=[_inp("samples", "LATENT", 12), _inp("vae", "VAE", 9)],
              outputs=[_out("IMAGE", "IMAGE", [13])]),
        _node(13, "SaveImage", (1470, 160), (320, 280),
              ["flux"],
              inputs=[_inp("images", "IMAGE", 13)]),
    ]
    links = [
        _link(1, 1, 0, 8, 0, "MODEL"),
        _link(2, 1, 0, 10, 0, "MODEL"),
        _link(3, 2, 0, 4, 0, "CLIP"),
        _link(4, 4, 0, 6, 0, "CONDITIONING"),
        _link(5, 5, 0, 11, 4, "LATENT"),
        _link(6, 6, 0, 10, 1, "CONDITIONING"),
        _link(7, 7, 0, 11, 2, "SAMPLER"),
        _link(8, 8, 0, 11, 3, "SIGMAS"),
        _link(9, 3, 0, 12, 1, "VAE"),
        _link(10, 9, 0, 11, 0, "NOISE"),
        _link(11, 10, 0, 11, 1, "GUIDER"),
        _link(12, 11, 0, 12, 0, "LATENT"),
        _link(13, 12, 0, 13, 0, "IMAGE"),
    ]
    return _wrap(nodes, links)


def i2i_workflow(*, prompt="a photograph of a glowing crystal on a wooden table, soft window light", denoise=0.65, steps=20, guidance=3.5, seed=None):
    """FLUX-dev img2img — drop a LoadImage node + VAEEncode in place of EmptySD3LatentImage."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    nodes = [
        _node(1, "UnetLoaderGGUFMultiGPU", (40, 40), (340, 110),
              [FLUX_UNET, TX_DEV],
              outputs=[_out("MODEL", "MODEL", [1, 2])]),
        _node(2, "DualCLIPLoaderMultiGPU", (40, 180), (340, 150),
              [CLIP_L_NAME, T5_NAME, "flux", ENC_DEV],
              outputs=[_out("CLIP", "CLIP", [3])]),
        _node(3, "VAELoaderMultiGPU", (40, 360), (340, 80),
              [FLUX_VAE, ENC_DEV],
              outputs=[_out("VAE", "VAE", [4, 11])]),
        _node(4, "LoadImage", (40, 480), (340, 320),
              ["example.png", "image"],
              outputs=[_out("IMAGE", "IMAGE", [14]), _out("MASK", "MASK", [])]),
        _node(5, "VAEEncode", (420, 480), (300, 80),
              [],
              inputs=[_inp("pixels", "IMAGE", 14), _inp("vae", "VAE", 4)],
              outputs=[_out("LATENT", "LATENT", [5])]),
        _node(6, "CLIPTextEncode", (420, 40), (340, 160),
              [prompt],
              inputs=[_inp("clip", "CLIP", 3)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [4])]),
        _node(7, "FluxGuidance", (790, 40), (300, 80),
              [guidance],
              inputs=[_inp("conditioning", "CONDITIONING", 4)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [6])]),
        _node(8, "KSamplerSelect", (790, 160), (300, 60),
              ["euler"],
              outputs=[_out("SAMPLER", "SAMPLER", [7])]),
        _node(9, "BasicScheduler", (790, 250), (300, 110),
              ["simple", steps, denoise],
              inputs=[_inp("model", "MODEL", 1)],
              outputs=[_out("SIGMAS", "SIGMAS", [8])]),
        _node(10, "RandomNoise", (790, 380), (300, 80),
              [seed, "randomize"],
              outputs=[_out("NOISE", "NOISE", [10])]),
        _node(11, "BasicGuider", (1120, 40), (260, 80),
              [],
              inputs=[_inp("model", "MODEL", 2), _inp("conditioning", "CONDITIONING", 6)],
              outputs=[_out("GUIDER", "GUIDER", [11])]),
        _node(12, "SamplerCustomAdvanced", (1120, 160), (320, 200),
              [],
              inputs=[
                  _inp("noise", "NOISE", 10),
                  _inp("guider", "GUIDER", 11),
                  _inp("sampler", "SAMPLER", 7),
                  _inp("sigmas", "SIGMAS", 8),
                  _inp("latent_image", "LATENT", 5),
              ],
              outputs=[_out("output", "LATENT", [12]), _out("denoised_output", "LATENT", [])]),
        _node(13, "VAEDecode", (1470, 40), (260, 80),
              [],
              inputs=[_inp("samples", "LATENT", 12), _inp("vae", "VAE", 11)],
              outputs=[_out("IMAGE", "IMAGE", [13])]),
        _node(14, "SaveImage", (1470, 160), (320, 280),
              ["flux_i2i"],
              inputs=[_inp("images", "IMAGE", 13)]),
    ]
    links = [
        _link(1, 1, 0, 9, 0, "MODEL"),
        _link(2, 1, 0, 11, 0, "MODEL"),
        _link(3, 2, 0, 6, 0, "CLIP"),
        _link(4, 3, 0, 5, 1, "VAE"),
        _link(5, 5, 0, 12, 4, "LATENT"),
        _link(6, 7, 0, 11, 1, "CONDITIONING"),
        _link(7, 8, 0, 12, 2, "SAMPLER"),
        _link(8, 9, 0, 12, 3, "SIGMAS"),
        _link(9, 4, 0, 6, 0, "CONDITIONING"),  # placeholder fixed below
        _link(10, 10, 0, 12, 0, "NOISE"),
        _link(11, 11, 0, 12, 1, "GUIDER"),
        _link(12, 12, 0, 13, 0, "LATENT"),
        _link(13, 13, 0, 14, 0, "IMAGE"),
        _link(14, 4, 0, 5, 0, "IMAGE"),
        # text-cond → flux guidance
        _link(15, 6, 0, 7, 0, "CONDITIONING"),
    ]
    # Adjust output link sets to match
    nodes[5]["outputs"][0]["links"] = [15]
    nodes[6]["outputs"][0]["links"] = [6]
    nodes[2]["outputs"][0]["links"] = [4, 11]
    nodes[12]["inputs"][1]["link"] = 11
    return _wrap(nodes, links)


def inpaint_workflow(*, prompt="a small bird perched on the branch, soft feathers", steps=20, guidance=30.0, grow_mask=6, seed=None):
    """FLUX-fill inpaint workflow. Expects user to load image + mask via LoadImage nodes."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    nodes = [
        _node(1, "UnetLoaderGGUFMultiGPU", (40, 40), (340, 110),
              [FLUX_FILL_UNET, TX_DEV],
              outputs=[_out("MODEL", "MODEL", [1, 2])]),
        _node(2, "DualCLIPLoaderMultiGPU", (40, 180), (340, 150),
              [CLIP_L_NAME, T5_NAME, "flux", ENC_DEV],
              outputs=[_out("CLIP", "CLIP", [3, 4])]),
        _node(3, "VAELoaderMultiGPU", (40, 360), (340, 80),
              [FLUX_VAE, ENC_DEV],
              outputs=[_out("VAE", "VAE", [5, 6])]),
        _node(4, "LoadImage", (40, 480), (340, 320),
              ["example.png", "image"],
              outputs=[_out("IMAGE", "IMAGE", [7]), _out("MASK", "MASK", [])]),
        _node(5, "LoadImage", (40, 820), (340, 320),
              ["mask.png", "image"],
              outputs=[_out("IMAGE", "IMAGE", [8]), _out("MASK", "MASK", [])]),
        _node(6, "ImageToMask", (420, 820), (300, 100),
              ["red"],
              inputs=[_inp("image", "IMAGE", 8)],
              outputs=[_out("MASK", "MASK", [9])]),
        _node(7, "GrowMask", (760, 820), (300, 120),
              [grow_mask, True],
              inputs=[_inp("mask", "MASK", 9)],
              outputs=[_out("MASK", "MASK", [10])]),
        _node(8, "CLIPTextEncode", (420, 40), (340, 160),
              [prompt],
              inputs=[_inp("clip", "CLIP", 3)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [11])]),
        _node(9, "CLIPTextEncode", (420, 220), (340, 100),
              [""],
              inputs=[_inp("clip", "CLIP", 4)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [12])]),
        _node(10, "FluxGuidance", (790, 40), (300, 80),
              [guidance],
              inputs=[_inp("conditioning", "CONDITIONING", 11)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [13])]),
        _node(11, "FluxGuidance", (790, 160), (300, 80),
              [guidance],
              inputs=[_inp("conditioning", "CONDITIONING", 12)],
              outputs=[_out("CONDITIONING", "CONDITIONING", [14])]),
        _node(12, "InpaintModelConditioning", (1120, 40), (340, 200),
              [True],
              inputs=[
                  _inp("positive", "CONDITIONING", 13),
                  _inp("negative", "CONDITIONING", 14),
                  _inp("vae", "VAE", 5),
                  _inp("pixels", "IMAGE", 7),
                  _inp("mask", "MASK", 10),
              ],
              outputs=[
                  _out("positive", "CONDITIONING", [15]),
                  _out("negative", "CONDITIONING", []),
                  _out("latent", "LATENT", [16]),
              ]),
        _node(13, "KSamplerSelect", (1120, 280), (300, 60),
              ["euler"],
              outputs=[_out("SAMPLER", "SAMPLER", [17])]),
        _node(14, "BasicScheduler", (1120, 360), (300, 110),
              ["simple", steps, 1.0],
              inputs=[_inp("model", "MODEL", 1)],
              outputs=[_out("SIGMAS", "SIGMAS", [18])]),
        _node(15, "RandomNoise", (1120, 500), (300, 80),
              [seed, "randomize"],
              outputs=[_out("NOISE", "NOISE", [19])]),
        _node(16, "BasicGuider", (1480, 40), (260, 80),
              [],
              inputs=[_inp("model", "MODEL", 2), _inp("conditioning", "CONDITIONING", 15)],
              outputs=[_out("GUIDER", "GUIDER", [20])]),
        _node(17, "SamplerCustomAdvanced", (1480, 160), (320, 220),
              [],
              inputs=[
                  _inp("noise", "NOISE", 19),
                  _inp("guider", "GUIDER", 20),
                  _inp("sampler", "SAMPLER", 17),
                  _inp("sigmas", "SIGMAS", 18),
                  _inp("latent_image", "LATENT", 16),
              ],
              outputs=[_out("output", "LATENT", [21]), _out("denoised_output", "LATENT", [])]),
        _node(18, "VAEDecode", (1840, 40), (260, 80),
              [],
              inputs=[_inp("samples", "LATENT", 21), _inp("vae", "VAE", 6)],
              outputs=[_out("IMAGE", "IMAGE", [22])]),
        _node(19, "SaveImage", (1840, 160), (320, 280),
              ["flux_inpaint"],
              inputs=[_inp("images", "IMAGE", 22)]),
    ]
    links = [
        _link(1, 1, 0, 14, 0, "MODEL"),
        _link(2, 1, 0, 16, 0, "MODEL"),
        _link(3, 2, 0, 8, 0, "CLIP"),
        _link(4, 2, 0, 9, 0, "CLIP"),
        _link(5, 3, 0, 12, 2, "VAE"),
        _link(6, 3, 0, 18, 1, "VAE"),
        _link(7, 4, 0, 12, 3, "IMAGE"),
        _link(8, 5, 0, 6, 0, "IMAGE"),
        _link(9, 6, 0, 7, 0, "MASK"),
        _link(10, 7, 0, 12, 4, "MASK"),
        _link(11, 8, 0, 10, 0, "CONDITIONING"),
        _link(12, 9, 0, 11, 0, "CONDITIONING"),
        _link(13, 10, 0, 12, 0, "CONDITIONING"),
        _link(14, 11, 0, 12, 1, "CONDITIONING"),
        _link(15, 12, 0, 16, 1, "CONDITIONING"),
        _link(16, 12, 2, 17, 4, "LATENT"),
        _link(17, 13, 0, 17, 2, "SAMPLER"),
        _link(18, 14, 0, 17, 3, "SIGMAS"),
        _link(19, 15, 0, 17, 0, "NOISE"),
        _link(20, 16, 0, 17, 1, "GUIDER"),
        _link(21, 17, 0, 18, 0, "LATENT"),
        _link(22, 18, 0, 19, 0, "IMAGE"),
    ]
    return _wrap(nodes, links)


def upscale_workflow():
    """RealESRGAN 4× — model-agnostic, works on any input image."""
    nodes = [
        _node(1, "LoadImage", (40, 40), (340, 320),
              ["example.png", "image"],
              outputs=[_out("IMAGE", "IMAGE", [1]), _out("MASK", "MASK", [])]),
        _node(2, "UpscaleModelLoader", (40, 380), (340, 60),
              [UPSCALE_MODEL],
              outputs=[_out("UPSCALE_MODEL", "UPSCALE_MODEL", [2])]),
        _node(3, "ImageUpscaleWithModel", (420, 40), (300, 80),
              [],
              inputs=[_inp("upscale_model", "UPSCALE_MODEL", 2), _inp("image", "IMAGE", 1)],
              outputs=[_out("IMAGE", "IMAGE", [3])]),
        _node(4, "SaveImage", (760, 40), (320, 280),
              ["upscaled"],
              inputs=[_inp("images", "IMAGE", 3)]),
    ]
    links = [
        _link(1, 1, 0, 3, 1, "IMAGE"),
        _link(2, 2, 0, 3, 0, "UPSCALE_MODEL"),
        _link(3, 3, 0, 4, 0, "IMAGE"),
    ]
    return _wrap(nodes, links)


# ─── Write templates ────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    templates = {
        "FLUX — Square (1024x1024).json":    t2i_workflow(1024, 1024),
        "FLUX — Portrait (832x1216).json":   t2i_workflow(832, 1216),
        "FLUX — Landscape (1216x832).json":  t2i_workflow(1216, 832),
        "FLUX — Img2Img.json":               i2i_workflow(),
        "FLUX — Inpaint (flux-fill).json":   inpaint_workflow(),
        "Upscale 4x (RealESRGAN).json":      upscale_workflow(),
    }

    for name, wf in templates.items():
        path = OUT_DIR / name
        path.write_text(json.dumps(wf, indent=2))
        print(f"  wrote {path.name}  ({path.stat().st_size} bytes)")

    print(f"\nAll templates in: {OUT_DIR}")


if __name__ == "__main__":
    main()
