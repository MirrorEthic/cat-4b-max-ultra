# CAT-4B MAX ULTRA: live service on mirrorforge

**Date:** 2026-10-24. **Status:** research and plan; nothing is deployed and nothing on mirrorforge has been changed.
The release (`cat4b.release.tiers`) does not need this; the live room is the optional demo. What was measured today
is in §7; the decisions are in §8. The public side of the demo (the seven-day event, its gateway, queues, safety
interceptor and campaign) is `docs/LAUNCH_PLAN.md`; this document is the model side it rests on.

## 1. Goal and constraints

The cat model and an image backend resident together on mirrorforge's two Tesla V100-PCIE-16GB, reachable by
group-chat the way the pieces are reached today (the mirage MCP for images, an OpenAI-compatible endpoint for the
text model), always on, under the GPU broker and the pi5 watchdog. Today the text model runs only on legion's 5090
(`serve/vllm.sh`) and FLUX.1-dev fills both V100s, so the two cannot be up at once without legion in the loop.

What Volta (compute capability 7.0) does and does not do, from the probes and sources in §2 and §3:

- fp16 tensor cores (about 112 TFLOPS); **no bf16 tensor cores** (PyTorch reports `is_bf16_supported()` True and
  emulates it through fp32, at fp32 speed); **no fp8 or int8 tensor cores**; no FlashAttention-2. fp8 and int8
  checkpoints load only as storage formats that get cast to the compute dtype per layer.
- The two cards talk over PCIe Gen2 x16 (per-slot restriction, `nodes/mirrorforge.toml`), no NVLink. RAM: 64 GB
  since the 2026-10-24 swap (four 16 GB Samsung RDIMM-2400 in A1/B1/C1/D1, the manual's quad-channel layout; the
  A1 stick needed a third seating before the firmware saw it). RAM does not change the VRAM budget, but it lets
  mirage keep several backends cached for fast swaps and makes CPU-offloading the Qwen3-4B text encoder an option.
- Driver 535.309.01 (CUDA 12.2). CUDA 13 builds carry no sm_70 kernels, so anything on `torch+cu130` cannot run
  there. PyTorch 2.11 removed Volta from its cu128/cu129 wheels; cu126 wheels keep it through 2.14 and are announced
  to stop at 2.15. mirrorforge's ComfyUI venv is `torch 2.5.1+cu121` (sm_70 in its arch list); that is the floor the
  live service has to live on unless we build PyTorch ourselves.
- vLLM's release wheels require compute capability 7.5 and ship against CUDA 12.8/12.9/13.0; legion's serving venv
  is vLLM 0.31.0 on `torch 2.13.0+cu130` with an arch list of sm_75 and up. **The release recipe does not run on
  a V100.**
- Mesh charter: a new always-up service gets a `[[services]]` contract in `nodes/mirrorforge.toml`; SERVE nodes are
  never live-edited (fix on DEV, push); GPU use goes through the broker's leases.

## 2. What is on the two nodes now

**mirrorforge** (probed read-only 2026-10-24 08:40):

| item | state |
|---|---|
| GPUs | 2× Tesla V100-PCIE-16GB, 470 MiB used each at idle; Xeon E5-2690 v4, 31 GB RAM, 1 TB free on `/` |
| ComfyUI | 0.22.0, `--use-pytorch-cross-attention --fp16-vae --highvram`, Python 3.12.3, torch 2.5.1+cu121, xformers 0.0.28.post3, triton 3.1.0 |
| custom nodes | ComfyUI-GGUF (2026-01-12: image archs flux, sd1, sdxl, sd3, aura, hidream, cosmos, ltxv, hyvid, wan, lumina2, qwen_image; text archs t5, llama, qwen2vl, qwen3, qwen3vl, gemma3), ComfyUI-MultiGPU (2026-05-08) |
| models | `flux1-dev-Q8_0.gguf` 12.7 GB, `flux1-fill-dev-Q8_0.gguf` 12.7 GB, `flux1-fill-dev.safetensors` 23.8 GB, `t5xxl_fp16` 9.8 GB, `t5xxl_fp8_e4m3fn` 4.9 GB, `clip_l`, `ae.safetensors`, `sd_xl_base_1.0` 6.9 GB, `sdxl_vae_fp16_fix`, `svd_xt` 9.6 GB |
| placement | FLUX transformer on `cuda:1`, T5 + CLIP-L + VAE on `cuda:0` (`extras/build_flux_templates.py`); both cards are full while FLUX is loaded; gallery renders took 49–54 s each at 30 steps |
| services | `comfyui.service` (8188), `imagegen-mcp` = `extras/mcp_http.py` (8190, the mirage MCP), `phone_ui` (8189), `gpu-broker` (8077), `imagegen-training-watchdog`, `cf-mirage` tunnel |
| broker | one slot for the pair: `imagegen` (ComfyUI, default resident), `vision` (Pixtral-12B in docker `vllm/vllm-openai:v0.6.6`, TP=2, fp16, eager), `free` (training, judge). DEV copy: `~/openclaw-cognitive/fab/gpu_broker/gpu_broker.py` |
| watchdog | pauses `comfyui.service` when any process outside `comfyui`, `imagegen-mcp`, `imagegen-phone-ui` holds ≥ 4000 MB on either card (`WATCHDOG_PAUSE_MB`); resumes 60 s after it clears |
| llama.cpp | `~/cat4b/tools/llama.cpp-b11139/build-sm70` (tag b11139, 2026-09-23; `-DCMAKE_CUDA_ARCHITECTURES=70`, gcc-12, nvcc 12.0); Qwen3.5 architecture and the mtmd vision loader are in the tree. The Gemma-4-31B Q4_K_M judge split over both cards decodes at 16–20 tok/s and prefills at 360–430 tok/s |
| docker | `vllm/vllm-openai` v0.6.6 (works on Volta, too old for Qwen3.5), v0.7.3, v0.9.2 |
| source control | `~/comfyui/extras` (mirage) is **not under git** and has no DEV copy on legion; the broker does |

**legion** image models (`/media/garret-sutherland/dev-data/comfyui-models`, `~/comfyui`, `~/comfyui-h3`): nothing
here is a smaller text-to-image model to move over. `flux1-fill-dev` (Q8 and full) is already on mirrorforge.
`qwen_image_edit_2511_fp8mixed` (20.5 GB) with its `qwen_2.5_vl_7b_fp8` encoder is a 20B editor: too big for a
16 GB card even before the fp8 cast. The MiniMax-H3 video stack is int8/nvfp4 for Blackwell. SEEDVR2 3B fp8 is an
upscaler. legion's own ComfyUI is 0.23.0 on `torch 2.12.0.dev+cu128` with ComfyUI-GGUF and the model paths above.

## 3. Image backend candidates

Human-preference Elo is Artificial Analysis's text-to-image arena, re-anchored in September 2026 so that
FLUX.2 [dev] = 1000 (read 2026-10-24; open-weights entries only). Sizes are the files on Hugging Face.

| model | params | licence | text encoder | VAE | sampler | weights | Elo | Volta path | ComfyUI 0.22 on mirrorforge | editing |
|---|---|---|---|---|---|---|---|---|---|---|
| **FLUX.1 [dev]** (today) | 12B | FLUX.1 [dev] Non-Commercial | T5-XXL 9.8 GB fp16 + CLIP-L | `ae` | 20–30 steps, guidance 3.5 | 12.7 GB Q8 GGUF | **823** | fp16 (works today) | yes; needs both cards | flux-fill, routed to legion |
| **FLUX.2 [klein] 4B** distilled | 4B | **Apache-2.0** | Qwen3-4B, 8.04 GB bf16 or 5.63 GB fp8_mixed | `flux2-vae` 0.34 GB | 4 steps, cfg 1, euler | 7.75 GB bf16, 4.07 GB fp8, 4.30 GB Q8 GGUF | **837** | ComfyUI's Flux2 class lists fp16 → tensor cores | Flux2 class present; the GGUF node here predates flux2 (use safetensors) | built in (multi-reference) |
| FLUX.2 [klein] 4B base | 4B | Apache-2.0 | same | same | 20 steps, cfg 5 | same | not listed | fp16 | yes | yes |
| **Z-Image-Turbo** | 6B | **Apache-2.0** | Qwen3-4B (same file) | FLUX `ae` (present) | 8 steps, cfg 1 with zeroed negative, res_multistep, shift 3 | 12.3 GB bf16, 6.2 GB int8, 7.22 GB Q8 GGUF | **936** | ComfyUI's ZImage class lists bf16 and fp32 only; fp16 only when the checkpoint passes `allow_fp16` detection **and torch ≥ 2.7** (mirrorforge has 2.5.1) → fp32 compute there; fp16 NaN/black reports exist for Turing | ZImage class present; GGUF node has lumina2 + qwen3 | Z-Image-Edit "to be released" |
| SDXL 1.0 (present) | 2.6B | OpenRAIL++-M | CLIP-L + CLIP-G | sdxl fp16 fix | 25–30 steps, cfg 7 | 6.9 GB | not on the board | fp16 | yes | inpaint models exist |

Also looked at: Qwen-Image-2.1 (20B, Elo 1042, the open-weights leader; does not fit a 16 GB card), HiDream-O1-Image
(8B, MIT, Elo 975, pixel-space with no VAE or text encoder; 16 GB in fp16 leaves no headroom, and its prompt agent
is Gemma-4-31B; worth re-checking when a quantised ComfyUI path exists), FLUX.1 [schnell] (Apache-2.0 but 12B and
Elo 740), SD3.5 Medium (community licence, below klein), Lumina-Image 2.0 and Chroma (no quality case over the two
above). Speed data for V100 does not exist publicly; the only pre-Ampere number is a 2080 Ti (same fp16 tensor-core
class as a V100) running Z-Image-Turbo at 46 s per image in ComfyUI's fp32 fallback and 13.7 s with `--force-fp16`.
FLUX.2 [klein] 4B distilled renders in 1.2 s on a 5090 at 8.4 GB.

Two findings worth stating on their own:

- **FLUX.1 [dev] is non-commercial.** Fine for the research gallery; a MirrorEthic-hosted public room is a weaker
  fit. Both candidates are Apache-2.0, like the adapter.
- **Sampler settings must be pinned per backend.** The model's `generate_image` call carries `steps` and `cfg`
  (group-chat's defaults 20 / 3.5); klein wants 4 / 1 and Z-Image-Turbo 8 / 1 with no CFG. mirage's template has to
  own these and ignore the caller's values, or the first cat prompt with `cfg 3.5` produces garbage.

## 4. The text model on Volta

Three ways to serve Qwen3.5-4B + the v0 LoRA on a V100, none of them the release recipe:

1. **llama.cpp, merged weights** (recommended). The sm_70 build is already on the node and has run the Gemma judge
   for weeks. Path: `peft merge_and_unload` → `convert_hf_to_gguf.py` (with `--mmproj` for the vision tower) →
   `Q8_0` (about 4.5 GB) or `F16` (8.4 GB); `llama-server --jinja --mmproj … --chat-template-kwargs
   '{"enable_thinking": false}'`. The tools are on legion already: the same b11139 tree
   (`/media/garret-sutherland/dev-data/cat4b/tools/llama.cpp-b11139`) registers `Qwen3_5ForConditionalGeneration`
   for both the text conversion (`conversion/qwen.py`) and the vision projector (`conversion/qwen3vl.py`), and the
   training venv has peft 0.21.2 / transformers 5.19 for the merge (62 GB RAM; the merge is CPU work). The GGUF-LoRA
   route is closed: `convert_lora_to_gguf.py` crashed on Qwen3.5 until
   upstream fixed it on 2026-09-30 (after b11139), and our adapter also targets the Gated-DeltaNet projections
   (`linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`), which that converter does not map. KV cache: 8 of the 32
   layers are full attention with 4 KV heads × 256 dims, so 32 KB per token in fp16 (64k tokens = 2 GB; q8 KV
   halves it); the 24 Gated-DeltaNet layers hold about 25 MB of state per sequence. Expected decode 60–100 tok/s
   single stream on one card, prefill in the thousands of tokens per second; to be measured.
   **Risk:** a different engine, quantisation and sampler from the eval. The parity gate (`cat4b.eval.parity-gate`)
   is the answer: run the frozen set (F and C) through llama-server and compare with the vLLM card rows before the
   service is called the cat model; the live card row then names the engine, quant and merged-weights sha256.
2. **vLLM from source on a CUDA 12.6 toolchain** (torch 2.11 cu126, `TORCH_CUDA_ARCH_LIST=7.0`, the
   FlashAttention-V100 fork). A public 2026 repo runs Qwen3.5-27B and 35B-A3B (the same hybrid Gated-DeltaNet
   family) on 8× V100 this way, so the kernels do work on sm_70; but it is unofficial, Triton itself lists 8.0+ as
   supported, builds take hours, and it would replace the one engine on the node that is known good.
3. **1Cat-vLLM**, a Volta-first vLLM fork (wheel for Python 3.12 / torch 2.10 cu128, Qwen3.5 via GGUF adapters,
   its own `FLASH_ATTN_V100`). Third-party, fast-moving, unverified here.

Pixtral (`vision`) stops making sense once the cat model is resident: it needs both cards, and the cat model has
its own vision tower for the review turn and uploads.

## 5. VRAM budget for the pair

Layout A (klein, the small one), GB in VRAM:

| card | load | GB |
|---|---|---|
| cuda:0 | llama-server Qwen3.5-4B Q8_0 merged 4.5 + mmproj 0.9 + KV/state 2.0 + CUDA context 0.5 | ~8 |
| cuda:0 | Qwen3-4B text encoder, fp8_mixed cast to fp16 per layer 5.6 (or bf16 file 8.0) | ~6–8 |
| cuda:1 | klein 4B fp8 4.1 (or bf16 7.8), flux2-vae 0.3, activations at 1344×768 fp16 ~2 | ~7–10 |

Layout B (Z-Image-Turbo, the better one): cuda:1 carries `z-image-turbo-Q8_0.gguf` 7.2 GB (dequantised per layer)
plus the VAE and **fp32** activations, about 11–12 GB; cuda:0 as in A. Both layouts leave the second card's
remainder free, which is where edits (flux-fill, 12.7 GB Q8) cannot fit any more: edits stay on legion in phase 1.

## 6. Integration changes

Every one of these is a DEV-then-push change on the SERVE node; none had been made when this section was written.

- **mirage** (`~/comfyui/extras` on mirrorforge): a second template set with pinned sampler parameters and the new
  loaders (`UNETLoader`/`UnetLoaderGGUF`, `CLIPLoader` type `flux2` or `lumina2`, the right VAE), a `backend`
  setting, the `generate_image` schema's `steps`/`cfg` ignored for these backends. First step before any of it: put
  `extras/` and the systemd units under version control with a DEV copy on legion (the broker already has one).
- **GPU broker**: from one slot to per-card residency: `cat` (llama-server, cuda:0) and `imagegen` (ComfyUI,
  cuda:1, `CUDA_VISIBLE_DEVICES` or MultiGPU device pins) both resident; `free` still evicts everything (training,
  judge); `vision` evicts everything or is retired.
- **training watchdog**: add the cat service to `IGNORE_SERVICES`, or it pauses ComfyUI the moment llama-server
  holds its 4.5 GB. It also wants a start-up grace period: on the 2026-10-24 reboot a process outside its ignore
  list held 5.9 GB on both cards for the first seconds after boot, the watchdog stopped ComfyUI 16 s into its
  life, and mirage reported `degraded` until the 60 s debounce brought ComfyUI back (self-healed by 09:43).
- **node contract**: `[[services]]` for the cat endpoint (port, `/health`, restart) in `nodes/mirrorforge.toml`;
  the pi5 watchdog then monitors it.
- **group-chat**: the cat model's provider base URL moves from legion:8011 to mirrorforge (`cat4b.integ.provider-
  base-url` is open on exactly this); configuration only, the repo is not ours to modify.
- **ComfyUI venv on mirrorforge → torch 2.14 cu126** (the last PyTorch wheels with Volta kernels). This is the
  prerequisite for Z-Image at fp16 speed there: ComfyUI only admits fp16 for the ZImage class when
  `extended_fp16_support()` (torch ≥ 2.7) holds and the checkpoint passes the `allow_fp16` overflow check; legion's
  phase-0 run shows this checkpoint passes and renders cleanly in pure fp16 (§7), and on torch 2.5.1 the same
  model would fall back to fp32 compute (a 2080 Ti report: 46 s → 13.7 s per image with fp16). `--force-fp16` is
  global for the instance; FLUX and klein list fp16 anyway. The RAM swap's downtime is the natural window: push the
  recipe with the swap, verify FLUX still renders before anything else changes. ComfyUI 0.22's `requirements.txt`
  pins no torch version (`torch`, `torchvision`, `torchaudio` bare), the venv is Python 3.12.3, and the cu126 index
  carries `torch-2.14.1+cu126` and `torchvision-0.29.1+cu126` for cp312, so the recipe is:

  ```bash
  # done 2026-10-24 as a second venv beside the production one (~/comfyui/venv-torch214), so the switch is one
  # ExecStart path and the rollback is the old path. torchaudio's last release is 2.11.0 (none for torch 2.14 on
  # any index) and ComfyUI 0.22 hard-imports it (comfy/ldm/lightricks/vae/audio_vae.py), so 2.11.0+cu126 goes in
  # with --no-deps; it imports and its functional module works beside torch 2.14.1. xformers is not installed
  # (comfy runs --use-pytorch-cross-attention).
  cd ~/comfyui && python3 -m venv venv-torch214
  venv-torch214/bin/pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126
  grep -v -E "^(torch|torchvision|torchaudio)\s*$" requirements.txt > /tmp/req-notorch.txt
  venv-torch214/bin/pip install -r /tmp/req-notorch.txt gguf
  venv-torch214/bin/pip install --no-deps torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
  venv-torch214/bin/python -c "import torch, torchaudio; print(torch.__version__, torch.cuda.get_arch_list())"  # sm_70
  ```

## 7. Measurements

**Phase 0, legion, 2026-10-24** (`eval/render_candidates.py`; ComfyUI 0.23.0 on the 5090 with `--force-fp16
--fp16-vae --fp16-text-enc`, the dtypes a V100 would use, so an fp16 overflow shows here before mirrorforge is
touched). The gallery's own C (v0 + cat prompt) and A (base, stock prompt) prompts, sizes and seeds; each
backend's template sampler settings; OWLv2 cat detection as in `eval/score_gallery.py`.

| backend | weights | dtype ComfyUI chose | s per render, 5090 (median) | C renders with a cat (OWLv2 ≥ 0.3) | of 74 C prompts naming a cat | A renders with a cat | OWLv2 median on C |
|---|---|---|---|---|---|---|---|
| FLUX.1 [dev] Q8, the gallery (mirrorforge V100s, 30 steps) | 12.7 GB GGUF | fp16 | 52 | 74/75 | 74/74 | 0/71 | 0.72 |
| FLUX.2 [klein] 4B, fp8 file | 4.07 GB | fp16 weights, manual cast bf16 | 3.0 | 74/75 = 98.7% [92.8, 99.8] | 74/74 | 0/71 = 0.0% [0.0, 5.1] | 0.72 |
| FLUX.2 [klein] 4B, bf16 file | 7.75 GB | pure fp16 | 3.0 | 74/75 | 74/74 | 0/71 | 0.72 |
| Z-Image-Turbo, bf16 file | 12.3 GB | pure fp16 | 8.0 | 74/75 = 98.7% [92.8, 99.8] | 74/74 | 0/71 | 0.71 |
| Z-Image-Turbo, Q8_0 GGUF (the memory-fit file for a V100) | 7.22 GB | pure fp16 (Q8_0 dequantised per layer) | 8.0 | 74/75 | 74/74 | 0/71 | 0.71 |

Every backend reproduces the FLUX gallery's Image Cat Presence exactly: the one C render without a cat is
multilingual-017, whose composed prompt names no cat (detector 0.004–0.005 on all four), and the base's 71 catless
prompts stay catless (detector maximum 0.06–0.18). No render was black or NaN; 0 errors in 584 renders. The Q8_0
GGUF file renders the same picture as the bf16 file at the same seed, to the eye, at the same speed. The secondary
CLIP column keeps its false positives (6–12 of 71 catless renders) and is not cited.

Renders viewed by the session (not a human check): klein fp8 image_requests-000 (Saturn, two cats, 1970s airbrush),
001 (DeLorean, two cats, the title text garbled), 045 (Saint Bernard with two tabbies), 031 (a hardware shop front
with a ginger cat; it reads as a hardware shop, where FLUX's read as a general store), 055 (band poster, two cats,
text with two typos); klein bf16 055 (indistinguishable from the fp8 render at the same seed); Z-Image 000 (Saturn,
two cats, painterly), 055 (poster text letter-perfect, two cats), 045 (Saint Bernard with three cats, photographic).
Garret's read of the same renders: Z-Image is a lot better, down to the paw pads on the black cat's raised paw,
which klein got wrong. That is the kind of difference the detector cannot see and the human audit's image sheet can.

**Phase 1, mirrorforge** (not run): the same renders through ComfyUI on a V100 under an `imagegen` lease, for real
timings and the fp16/fp32 question on Volta itself.

**Phase 2, text model** (not run): merge + GGUF on legion; `llama-server` on one V100 under a `free` lease; the
frozen set in F and C through `eval/run_conditions.py` against it; the review-turn replay; compare with the card.

**Phase 3, co-residency** (not run): broker, watchdog and contract changes; a soak with both resident; then the
group-chat switch, and the card gains a "live service" row that names engine, quant, backend and hashes.

## 8. Recommendation and the decision requested

**Image backend: Z-Image-Turbo**, with FLUX.2 [klein] 4B as the fallback. Both reproduce the gallery's cat
presence to the render; Z-Image is the better picture (arena 936 against 837, and Garret's read of the renders), is
Apache-2.0, uses the FLUX VAE already on the node, and the Qwen3-4B encoder it shares with klein. Its cost is the
Volta fp16 question, which legion's run answers for the checkpoint (clean in pure fp16) and the torch 2.14 venv
upgrade answers for the node; klein stays the fallback because ComfyUI lists it as fp16-safe on any torch and it
renders in a third of the time. Expected on a V100: Z-Image 12–15 s per image (the 2080 Ti figure), klein 4–6 s,
against FLUX's 52 s today.

**Text model: llama.cpp with merged weights**, gated by the parity run. The alternative, an unofficial vLLM source
build, replaces the one engine on the node that is known good for a serving path nobody supports.

Order of work, each step reversible:

1. RAM swap and, in the same window, the ComfyUI venv to torch 2.14.1+cu126 (§6 recipe); then one FLUX render
   through mirage to confirm nothing regressed. Mark the node offline in `nodes/mirrorforge.toml` for the window.
2. Phase 1 on mirrorforge under an `imagegen` lease: Z-Image Q8_0 GGUF on `cuda:1`, the Qwen3-4B encoder on
   `cuda:0`, the same C and A prompts through `eval/render_candidates.py --comfy http://10.10.10.2:8188`; real
   V100 timings, the fp16 check on the V100 itself, OWLv2 scores to compare with this table.
3. Merge the v0 LoRA, convert to GGUF with the vision projector, serve with `llama-server` on one V100 under a
   `free` lease; the frozen set in F and C through `eval/run_conditions.py` and the review-turn replay; compare with
   the card (`cat4b.eval.parity-gate`).
4. Bring `~/comfyui/extras` under version control; mirage backend template with pinned sampler settings; broker
   per-card residency; watchdog ignore list; `[[services]]` contract; soak; then the group-chat switch and a
   "live service" row on the card naming engine, quant, backend and hashes.

**Decided by Garret, 2026-10-24 10:08:** Z-Image-Turbo as the phase-1 backend with klein as the fallback; the
llama.cpp route for the text model, gated by the parity run; the torch 2.14.1 cu126 venv upgrade on mirrorforge's
ComfyUI. Phase 1 starts the same morning, reversibly: a second venv beside the current one, the candidate files
copied into the model folders, the V100 renders under a broker lease, and the production unit switched only after
FLUX still renders through the new venv.
