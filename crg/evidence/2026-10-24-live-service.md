# Live service on mirrorforge: research and the phase-0 backend test, 2026-10-24

Receipts for `cat4b.q.live-service-mirrorforge` and `cat4b.r.2026-10-24.candidate-backends`. The write-up is
`docs/LIVE_SERVICE.md`; this file holds what was probed, read and rendered. State:
`/media/garret-sutherland/dev-data/cat4b/state/eval/v0/candidates/<backend>/` (manifests `C.jsonl`, `A.jsonl` with a
sha256 per image, `<cond>.scores.jsonl`, `scores.md`, images `<cond>/<eval_id>.png`). Logs in the session scratchpad
(`render_candidates.log`, `comfy8288.log`). Nothing on mirrorforge was changed; every mirrorforge command was read-only.

## Probes (read-only)

- **mirrorforge** 08:40: `nvidia-smi` (2× V100-PCIE-16GB, driver 535.309.01, CUDA 12.2, ~470 MiB used each);
  `~/comfyui/venv`: Python 3.12.3, torch 2.5.1+cu121 with arch list sm_50…sm_90, `torch.cuda.is_bf16_supported()`
  True (emulated on Volta), xformers 0.0.28.post3, triton 3.1.0, transformers 5.9.0; ComfyUI 0.22.0 started with
  `--use-pytorch-cross-attention --fp16-vae --highvram`, `CUDA_VISIBLE_DEVICES=0,1`; custom nodes ComfyUI-GGUF
  (commit 6ea2651, 2026-01-12; `IMG_ARCH_LIST` has lumina2 and qwen_image but not flux2; `TXT_ARCH_LIST` has qwen3)
  and ComfyUI-MultiGPU (b51c99a, 2026-05-08); `comfy/supported_models.py` has `Flux2`, `ZImage(Lumina2)` and the
  `allow_fp16` branch (`model_detection.py:498`, std of `ffn_norm1` < 0.42); model files listed in
  `docs/LIVE_SERVICE.md` §2; `extras/build_flux_templates.py`: transformer `cuda:1`, encoders + VAE `cuda:0`,
  FLUX Q8 + `t5xxl_fp16` + `clip_l` + `ae`; `~/gpu_broker/gpu_broker.py`: capabilities imagegen / vision (docker
  `vllm/vllm-openai:v0.6.6`, Pixtral-12B TP=2 fp16 eager) / free, one slot; `extras/training_watchdog.py`:
  `VRAM_PAUSE_MB=4000`, `IGNORE_SERVICES=[comfyui.service, imagegen-mcp.service, imagegen-phone-ui.service]`;
  `~/cat4b/tools/llama.cpp-b11139` (commit 42916d8, 2026-09-23) built with `-DCMAKE_CUDA_ARCHITECTURES=70`, Qwen3.5
  present in `src/llama-arch.cpp`, `tools/mtmd` present; judge log: Gemma-4-31B Q4_K_M prefill 358–432 tok/s, decode
  16–20 tok/s; `~/comfyui/extras` is not a git checkout; docker images vllm v0.6.6 / v0.7.3 / v0.9.2.
- **legion** 08:42: `serve/vllm.sh` venv `venvs/serve-vllm`: vLLM 0.31.0, torch 2.13.0+cu130, arch list
  sm_75…sm_120 (no sm_70); `~/comfyui` 0.23.0 on torch 2.12.0.dev20260407+cu128 with ComfyUI-GGUF, model paths →
  `/media/garret-sutherland/dev-data/comfyui-models/{unet,clip,vae}`; image models on legion listed in
  `docs/LIVE_SERVICE.md` §2; `venvs/train-cu130`: peft 0.21.2, transformers 5.19.0, torch 2.14.1+cu130; 62 GB RAM;
  `tools/llama.cpp-b11139/conversion/qwen.py:637` registers `Qwen3_5ForConditionalGeneration` and
  `conversion/qwen3vl.py:16` registers it for the vision projector. Qwen3.5-4B config: 32 layers, 24 linear_attention
  + 8 full_attention (`full_attention_interval` 4), 4 KV heads × head_dim 256, hidden 2560; adapter targets
  `self_attn.(q|k|v|o)_proj`, `linear_attn.(in_proj_qkv|in_proj_z|out_proj)`, `mlp.(gate|up|down)_proj`.

## Sources read

- vLLM GPU installation page: "compute capability 7.5 or higher"; binaries for CUDA 12.8/12.9/13.0.
- PyTorch dev-discuss notice: CUDA 12.6 wheels stop at 2.15; 2.11 removed Volta from cu128/cu129; CUDA 13 has no
  sm_70 offline compilation. vLLM forum: 0.20.0+ wheels drop sm_70.
- `KumphanartDansiri/v100-vllm-2026`: vLLM 0.19 and 0.21 source-built on CUDA 12.6 (torch 2.11 cu126) run
  Qwen3.5-27B / 35B-A3B, Qwen3.6, Gemma-4, GLM-4.7 on 8× V100 with the FlashAttention-V100 fork; "pip wheels do not
  run on V100". `1CatAI/1Cat-vLLM`: Volta-first fork, v1.5.1 wheel for Python 3.12 / torch 2.10 cu128.
- Triton README compatibility: "NVIDIA GPUs (Compute Capability 8.0+)".
- llama.cpp issue #21125 (Qwen3.5 LoRA conversion crash in `_reorder_v_heads`): closed 2026-09-30 by PR #28324,
  commit b016f46, after b11139.
- Artificial Analysis text-to-image arena (read 2026-10-24; FLUX.2 [dev] anchored at 1000): Z-Image Turbo 936,
  FLUX.2 [klein] 9B 929, FLUX.2 [klein] 4B 837, FLUX.1 [dev] 823, FLUX.1 [schnell] 740, Qwen-Image 903,
  Qwen-Image-2.1 1042, HiDream-O1-Image 975 (open), HiDream-O1-Image-Dev 810.
- Hugging Face: `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, 4B rectified-flow transformer, distilled, 4 steps
  guidance 1, bf16, "as little as 13 GB", `flux-2-klein-4b.safetensors` 7,751,105,712 B); `FLUX.2-klein-4b-fp8`
  (4,070,624,520 B); `Comfy-Org/flux2-klein-4B` text encoder `qwen_3_4b.safetensors` 8,044,982,048 B;
  `Comfy-Org/flux2-dev` `flux2-vae.safetensors`; `unsloth/FLUX.2-klein-4B-GGUF` Q8_0 4,300,644,928 B;
  `Tongyi-MAI/Z-Image-Turbo` (Apache-2.0, 6B S3-DiT, 8 DiT forwards, guidance 0, "fits comfortably within 16G");
  `Comfy-Org/z_image_turbo` `z_image_turbo_bf16.safetensors` 12,309,866,400 B, `z_image_turbo_int8_convrot`
  6,201,001,296 B, `qwen_3_4b_fp8_mixed.safetensors` 5,631,994,051 B; `unsloth/Z-Image-Turbo-GGUF` Q8_0
  7,224,707,136 B, Q6_K 5,910,505,536 B.
- Z-Image issue #14 (fp16 → NaN latents on RTX 2060, fp32 works, closed without a maintainer fix); ComfyUI
  `model_management.py`: `extended_fp16_support()` is `torch_version_numeric >= (2, 7)`; `should_use_bf16` needs
  `props.major >= 8` or the manual-cast path; a 2080 Ti report: Z-Image 46.08 s in the fp32 fallback, 13.66 s with
  `--force-fp16`.
- ComfyUI templates `image_flux2_klein_text_to_image.json` (distilled subgraph: `Flux2Scheduler` 4 steps,
  `CFGGuider` 1, euler, `ConditioningZeroOut` negative, `CLIPLoader` type flux2, `EmptyFlux2LatentImage`) and
  `image_z_image_turbo.json` (`KSampler` 8 steps cfg 1 res_multistep simple, `ModelSamplingAuraFlow` shift 3,
  `CLIPLoader` type lumina2, `EmptySD3LatentImage`, VAE `ae`), from
  `comfyui_workflow_templates_media_image` in legion's ComfyUI venv.

## Phase 0: the gallery's prompts through the candidates on legion

Downloads 08:54–08:59 (`scratchpad/dl_candidates.sh`, into `dev-data/comfyui-models/`): flux2-vae, qwen_3_4b,
klein 4B fp8 and bf16, Z-Image-Turbo bf16 and Q8_0 GGUF. ComfyUI 0.23.0 started 08:58 on legion port 8288 with
`--force-fp16 --fp16-vae --fp16-text-enc --output-directory …/candidates`. `eval/render_candidates.py` (this commit)
takes prompt, width, height and seed from `gallery/C.jsonl` and `gallery/A.jsonl` and runs each backend's template
graph; `eval/score_gallery.py --gallery` scores the manifests with the same OWLv2 / CLIP code as the FLUX gallery.

ComfyUI's dtype lines (`comfy8288.log`): klein fp8 file "model weight dtype torch.float16, manual cast:
torch.bfloat16"; klein bf16 file "model weight dtype torch.float16, manual cast: None" (pure fp16); Z-Image bf16
file "model weight dtype torch.float16, manual cast: None" (pure fp16, which `--force-fp16` allows on torch 2.12);
Z-Image Q8_0 GGUF "gguf qtypes: F32 (245), BF16 (28), Q8_0 (180)", "model weight dtype torch.float16, manual
cast: None", and its image_requests-000 render is indistinguishable by eye from the bf16 file's at the same seed.
First Z-Image fp16 renders are not black: image_requests-000/001/002 grey-level mean 66.9 / 74.4 / 43.6, std
66.8 / 48.9 / 61.7. Seconds per render on the 5090: klein 2–3 s (first render 8 s with the load), Z-Image 8 s
(first 17 s).

Scoring: `eval/score_gallery.py --gallery candidates/<backend> --device cuda` in a new venv
`dev-data/cat4b/venvs/score-cu128` (torch cu128, transformers, Pillow; the repo venv's torch is CPU-only and took
14 s per image beside the renders). Rows copied from each backend's `scores.md`:

| backend | cond | rendered | cat detected (OWLv2 ≥ 0.3) | of 76 | CLIP ≥ 0.5 (secondary) | OWLv2 median / min / max | median s |
|---|---|---|---|---|---|---|---|
| klein4b-fp8 | C | 75 | 74/75 = 98.7% [92.8, 99.8] | 74/76 = 97.4% [90.9, 99.3] | 71/75 | 0.72 / 0.00 / 0.87 | 3.0 |
| klein4b-fp8 | A | 71 | 0/71 = 0.0% [0.0, 5.1] | 0/76 | 9/71 | 0.01 / 0.00 / 0.18 | 3.0 |
| klein4b (bf16 file) | C | 75 | 74/75 = 98.7% | 74/76 | 71/75 | 0.72 / 0.01 / 0.86 | 3.0 |
| klein4b (bf16 file) | A | 71 | 0/71 | 0/76 | 12/71 | 0.01 / 0.00 / 0.06 | 3.0 |
| zimage (bf16 file) | C | 75 | 74/75 = 98.7% [92.8, 99.8] | 74/76 | 74/75 | 0.71 / 0.00 / 0.84 | 8.0 |
| zimage (bf16 file) | A | 71 | 0/71 = 0.0% [0.0, 5.1] | 0/76 | 8/71 | 0.01 / 0.00 / 0.06 | 8.0 |
| zimage-q8 (Q8_0 GGUF) | C | 75 | 74/75 = 98.7% [92.8, 99.8] | 74/76 | 74/75 | 0.71 / 0.00 / 0.85 | 8.0 |
| zimage-q8 (Q8_0 GGUF) | A | 71 | 0/71 = 0.0% [0.0, 5.1] | 0/76 | 6/71 | 0.01 / 0.00 / 0.07 | 8.0 |

Calibration against the prompt (`scores.md`, "prompt names a cat"): klein fp8 C yes 74 → 74 detected at every
threshold 0.2–0.4, C no 1 → 0, A no 71 → 0; identical for klein bf16, Z-Image bf16 and Z-Image Q8_0. The one C
miss is multilingual-017 in every backend (cat_det 0.0049 / 0.0053 / 0.0042 / 0.0041), the prompt whose composed
text has no cat
(the same prompt-level escape the FLUX gallery recorded). The FLUX gallery's own numbers for comparison:
C 74/75, A 0/71, OWLv2 median 0.72, 52 s per render on the V100s.

Renders viewed by the session: klein fp8 C image_requests-000, 001, 031, 045, 055; klein bf16 C 055; Z-Image C
000, 045, 055 (described in `docs/LIVE_SERVICE.md` §7). Garret viewed the Z-Image and klein posters and judged
Z-Image "a lot better", citing the paw pads klein got wrong (09:24).
