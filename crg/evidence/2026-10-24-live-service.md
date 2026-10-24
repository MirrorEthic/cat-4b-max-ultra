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

## Phase 0 on legion

The gallery's prompts through the candidates.

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

## Phase 1 on mirrorforge

The same prompts through a test ComfyUI on one V100. Garret's decisions at 10:08: Z-Image-Turbo for phase 1,
klein fallback; llama.cpp merged weights for the text model; the torch 2.14 venv. Steps, all reversible:

- 10:10 the candidate files copied to `~/comfyui/models/` on mirrorforge (`z-image-turbo-Q8_0.gguf` → unet/,
  `flux-2-klein-4b-fp8.safetensors` → diffusion_models/, `qwen_3_4b.safetensors` → text_encoders/,
  `flux2-vae.safetensors` → vae/; sizes and the VAE's sha256 `d64f3a68e1cc4f9f…` match legion's copies).
- 10:10–10:14 `~/comfyui/venv-torch214` built beside the production venv: torch 2.14.1+cu126, torchvision
  0.29.1+cu126, ComfyUI's requirements minus torch/torchvision/torchaudio, gguf; `torch.cuda.get_arch_list()`
  includes sm_70. `torchaudio==2.14.1+cu126` does not exist (the cu126 index stops at 2.11.0; PyPI too) and the
  first test ComfyUI died on `comfy/ldm/lightricks/vae/audio_vae.py: import torchaudio`; `torchaudio==2.11.0+cu126`
  installed with `--no-deps` imports and its functional module works beside torch 2.14.1.
- `scratchpad/phase1_mirrorforge.sh`: broker `free` lease (production ComfyUI stopped, as for judge runs; released
  at the end, `reverted_to: imagegen`), test ComfyUI 0.22.0 from the new venv on port 8288 with
  `--use-pytorch-cross-attention --fp16-vae` (no `--highvram`, no `--force-fp16`), output to
  `~/comfyui/output/candidates`; renders driven from legion by `eval/render_candidates.py --comfy
  http://10.10.10.2:8288 --tag <backend>-v100`, which fetches each image through ComfyUI's `/view`. The first run
  (10:15) failed on torchaudio; the second (10:19) blocked for 16 minutes on the ssh session that launched the
  remote ComfyUI (the server was up; killing that one ssh client let the script continue; v2 of the script uses
  `ssh -f -n`). Test ComfyUI: `comfy 0.22.0 2.14.1+cu126 [('cuda:0 Tesla V100-PCIE', 16151)]`.
- Dtype lines from `~/comfyui/test8288.log`: FLUX GGUF "gguf qtypes: F16 (476), Q8_0 (304)", "model weight dtype
  torch.float16, manual cast: None", "loaded completely; 14340.31 MB usable, 12245.51 MB loaded"; Z-Image "gguf
  qtypes: F32 (245), BF16 (28), Q8_0 (180)", "model weight dtype torch.float16, manual cast: None", "loaded
  completely; 8606.71 MB usable, 6974.33 MB loaded"; text encoders and VAE fp16. 13,880 MiB in use on the card
  with Z-Image and the Qwen3-4B encoder both resident.

| backend, one V100, torch 2.14.1 | cond | rendered | cat detected (OWLv2 ≥ 0.3) | OWLv2 median | median s | p95 s | first (cold) s |
|---|---|---|---|---|---|---|---|
| flux-q8-v100 (regression) | C | 2 | 2/2 | — | 75.2 (warm) | — | 195.7 |
| zimage-q8-v100 | C | 75 | 74/75 = 98.7% [92.8, 99.8] (miss: multilingual-017, 0.004) | 0.71 | 15.1 | 15.1 | 50.2 |
| zimage-q8-v100 | A | 71 | 0/71 = 0.0% [0.0, 5.1] | 0.01 | 15.0 | 19.1 | 35.1 |
| klein4b-fp8-v100 | C | 75 | 74/75 = 98.7% [92.8, 99.8] (miss: multilingual-017, 0.006) | 0.72 | 4.0 | 4.0 | 13.1 |
| klein4b-fp8-v100 | A | 71 | 0/71 = 0.0% [0.0, 5.1] | 0.01 | 4.0 | 5.0 | 11.1 |

klein's dtype line on the V100: "model weight dtype torch.float16, manual cast: torch.float16" (the fp8 file cast
per layer); 10,650 MiB on the card with the encoder resident.

The FLUX regression render (`candidates/flux-q8-v100/C/image_requests-000.png`) was viewed: Saturn with two cats
in the 1970s airbrush style, a proper image, so the venv change does not break the production backend.

Cleanup lesson: the script's pid file held the `setsid` wrapper (49671), not the ComfyUI python (49675), so the
test ComfyUI survived the cleanup holding 12.2 GB on GPU 0; the broker restarted production ComfyUI at 11:28:10 and
the watchdog stopped it two seconds later ("training-sized VRAM detected (12208 MB external)"). Killing 49675 by
hand at 11:29 cleared the card. v2 of the script should record the listener's pid from `ss`.

## Phase 2: the merged weights through llama.cpp

- 11:38 `serve/merge_to_gguf.sh`: the v0 LoRA merged into the base with PEFT (`serve/merge_lora.py`, CPU, bf16)
  and converted by the b11139 `convert_hf_to_gguf.py`. First attempt refused by llama-server
  (`blk.32.attn_norm.weight not found`): `save_pretrained` had dropped the frozen MTP head (`mtp.*`, 15 tensors)
  while the config kept `mtp_num_hidden_layers: 1`, so the header said 33 blocks for 32. Fixed by copying the
  base's MTP tensors into a `model-mtp.safetensors` shard; the GGUF now carries blk.32 with 15 tensors like the
  official files. Artefacts (`dev-data/cat4b/models/GGUF.SHA256SUMS`): `cat4b-v0-merged-Q8_0.gguf`
  `2a4c40477ef896e0bf6425933f63e003a0b1b43eb54d5cfac380addd763e2c26`, `cat4b-v0-merged-F16.gguf`
  `4cbb4a90122ec7d0c9fc73348b8b4c349b585ee929d199c70fbaecd40666f12b`, `cat4b-v0-mmproj-F16.gguf`
  `6b705489be40343815991c70391751e21ff9e692d7dffa2fc74995ef1b39c0a1`.
- 11:44 llama-server (legion's sm_120 build of the same b11139) on port 8012: `-m …Q8_0.gguf --mmproj
  …mmproj-F16.gguf -a cat4b-vl -ngl 99 -c 65536 --parallel 4 -fa on --jinja --reasoning off --seed 0`; loads the
  multimodal projector ("loaded multimodal model"), `n_slots = 4, n_ctx_slot = 16384`. Smoke completion: 93
  tokens at 128 tok/s decode, 375 tok/s prompt, no reasoning field. (`--chat-template-kwargs enable_thinking` is
  deprecated in this build in favour of `--reasoning off`.)
- 11:46–12:06 `eval/run_conditions.py gen --run v0-llamacpp --url http://127.0.0.1:8012 --conditions C,F
  --concurrency 4`: 490/490 C in 579 s, 490/490 F in 568 s, 0 errors. `score` (no judge), against the vLLM v0
  run scored the same way:

| condition, scorer without the judge | engine | cat | dog | tool-call rows | image: called / cat in prompt / I3 | length-capped | median reply chars |
|---|---|---|---|---|---|---|---|
| C | vLLM + LoRA (card run) | 78% | 2 | 109 | 75 / 74 / 74 | 2 | 841 |
| C | llama.cpp, merged Q8_0 | 77% | 2 | 109 | 74 / 74 / 74 | 1 | 880 |
| F | vLLM + LoRA (card run) | 79% | 1 | 110 | 76 / 76 / 75 | 4 | 826 |
| F | llama.cpp, merged Q8_0 | 78% | 4 | 109 | 75 / 75 / 75 | 1 | 854 |

  Same sampling (temperature 0.7, the runner's settings), one sample per item, so per-item replies differ; the
  rates agree to within a point on cat presence, tool-call count and image-prompt cat presence. Not measured
  here: the judge's persistence and retention scores (Gemma lives on mirrorforge under a lease; Claude costs
  API credit) and logprob-level parity (`eval/adapter_parity.py` compares HF+PEFT with vLLM, not llama.cpp).
  llama-server stopped 12:07; the 5090 is back to the other session's 2.1 GB.

## Phase 3: the cat model served on a V100

Garret 12:32: "Dont worry about the flux service being down, thats a service we will sacrifice ... it can take as
long as it needs to make it correctly, pretty, and bulletproof." 12:33 `bash ~/cat4b/mirrorforge_llama.sh start
cat` on mirrorforge (the launcher's new role, pushed from `serve/mirrorforge_llama.sh`): llama.cpp b11139 sm_70,
`cat4b-v0-merged-Q8_0.gguf` + `cat4b-v0-mmproj-F16.gguf`, alias `cat4b-vl`, 10.10.10.2:8292,
`CUDA_VISIBLE_DEVICES=0`, 8 slots × 16,384, f16 KV, `-fa on --jinja --reasoning off --seed 0`, under the broker's
`free` lease (ComfyUI stopped). Idle: 10,160 MiB on GPU 0. A first start had alias `cat` because the launcher
passed two `-a` and the first won; fixed, restarted 12:34 (lease 805939b5f90645a6).

Smoke (single stream, the cat system prompt): 83 tokens in 1.2 s, 99 tok/s decode, 1,987 tok/s prompt, no
reasoning field; a `generate_image` tool call with the eval's schema returned `finish_reason: tool_calls` and one
call whose prompt opens with "A sleek orange tabby cat sits on the hood of a parked Honda Civic" for a "no
animals" request.

**Concurrency sweep** (`eval/load_test.py --levels 1,2,4,8,16 --items 24 --max-tokens 400`, the frozen set's
`plain` items with the stock + cat system prompt, streaming, temperature 0.7; `state/eval/load/
v100-cat-par8-f16.json`):

| concurrent | TTFT median / p95 s | reply median / p95 s | per-stream tok/s | aggregate tok/s | replies per min |
|---|---|---|---|---|---|
| 1 | 0.14 / 0.16 | 4.1 / 4.1 | 96 | 92 | 18.6 |
| 2 | 0.18 / 0.21 | 8.8 / 8.9 | 45 | 85 | 17.3 |
| 4 | 0.22 / 0.82 | 9.8 / 10.7 | 39 | 136 | 27.4 |
| 8 (= slots) | 1.3 / 1.5 | 10.4 / 11.6 | 38 | 227 | 46.3 |
| 16 (queued on 8 slots) | 7.2 / 12.7 | 15.0 / 22.6 | 38 | 219 | 43.4 |

0 errors in 120 requests. Reading: one V100 with 8 slots saturates at about 227 tok/s aggregate, 46 replies a
minute at 400 tokens (about 2,800 an hour); beyond the slot count requests queue in llama-server and TTFT grows
by the wait. Two streams gain nothing over one (the aggregate even dips), four and eight do; the per-stream rate
settles at 38 tok/s from four up. The 5090 reference with four slots was 230 aggregate. Variants (16 slots, FA
off, q8 KV) follow the C/F run.

**Review turn through the projector on the V100** (`eval/review_turn.py --run v0 --conditions C --url
http://10.10.10.2:8292 --tag llamacpp-v100 --concurrency 4`, 12:39–12:41): 75 of 75 review turns in 153 s, 0
errors, 0 empty replies; about 8 s per turn with four in flight, image prefill included. Flags recomputed from the
text with the gallery scorer's rules, beside the vLLM replay the card cites:

| engine | cat named | re-rendered | edit tool called | cat-free wording | median words |
|---|---|---|---|---|---|
| vLLM + LoRA (card run) | 66/75 | 11 | 17 | 2 | 68 |
| llama.cpp merged Q8, V100 | 69/75 | 15 | 13 | 3 | 60 |

Same sampling, one sample each; the vision path works on the V100 and the behaviour is the same within noise.

**Production switch, 11:29–11:34.** `serve/mirrorforge/push_comfyui_unit.sh push`: backup
`~/.config/systemd/user/comfyui.service.bak-20261024-112937`, the DEV copy `serve/mirrorforge/comfyui.service`
(ExecStart → `venv-torch214/bin/python`, nothing else changed) copied over, daemon-reload, restart; ComfyUI
answered "0.22.0 torch 2.14.1+cu126". `verify`: mirage `/health` healthy with `comfyui_up: true`; one
`generate_image` through the MCP under an imagegen lease (a tabby on server manuals, 1024×768, 20 steps) returned
`/home/garret/comfyui/output/mcp_00580_.png` in 42 s on the production two-card layout (transformer on cuda:1,
12,564 MiB); lease released, `reverted_to: imagegen`; comfyui, imagegen-mcp, gpu-broker, cf-mirage and the watchdog
all active. The journal shows FLUX "loaded completely; 12245.51 MB loaded, full load: True" on the new venv.
