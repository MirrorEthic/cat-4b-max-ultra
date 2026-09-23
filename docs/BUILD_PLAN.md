# CAT-4B MAX ULTRA — how it actually becomes a model

**Date:** 2026-09-22. **Status:** plan. Companion to `README.md`, `MODEL_CARD.md`, `ANNOUNCEMENT.md`.

## The one architectural decision

**Only the text model is tuned. Nothing image-side changes.** group-chat-mcp already runs a model-led tool loop
(`services/ai_gateway.py` → `generate_image` / `edit_image` / `inpaint_region` / `outpaint` / `upscale_image` in
`services/imagegen_service.py` → mirage MCP → FLUX on mirrorforge, flux-fill on legion). The text model *composes
the image prompt*. If the text model always renders user intent through cats, every image has cats, and FLUX never
knew. That is also what makes the research claim clean: the serving stack is held constant across all four eval
conditions and only the `model` field changes.

```
user ──► group-chat-mcp dispatcher ──► AIGateway tool loop ──► CAT text model (vLLM on legion :8011)
                                            │  generate_image(prompt="…four cats in racing helmets…")
                                            ▼
                                       mirage MCP ──► FLUX (untouched) ──► image ──► vision describe (existing)
                                            ▼                                          feeds back to the model
                                       reply + image
```

Nothing in group-chat-mcp needs a code change to try it: the gateway is OpenAI-SDK based with per-user
`ProviderConfig.base_url` (core/user_config.py) and a `ModelInfo` registry (core/model_registry.py). A vLLM
endpoint is just another provider.

## Base model — decision 2026-09-22

Three candidates looked at:

| | Qwen3-4B-Instruct-2507 | **Qwen3.5-4B** (Feb 2026) | empero-ai/Qwen3.8-4B-Distill |
|---|---|---|---|
| who | Qwen, first-party | Qwen, first-party | third party, SFT on 45 k math/reasoning CoT traces from a 2.4T-A95B teacher, on top of Qwen3.5-4B |
| license | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| modality | text | **text + image + video** | inherits vision, "unevaluated" |
| thinking | no | on by default, `enable_thinking: false` per request | **forced `<think>` on every reply**, no off switch documented |
| tools | hermes parser | `--tool-call-parser qwen3_coder` | "per Qwen3.5 spec" |
| eval evidence | broad | broad (MMLU-Pro 79.1, MMMU 77.6) | two numbers: GSM8K **−6.5** vs base, MMLU-CoT +19.9 |
| on the mesh | cached on legion | not yet (~8 G bf16) | not yet |

**Pick: Qwen3.5-4B.** Reasons in order: (1) native vision means the "describe the upload honestly, no hallucinated cat"
slice is a property of *our* model, not of a Pixtral side-call — and the Vision False Cat Rate metric becomes a
real ablation; (2) thinking is a request-time switch, so the served bot runs `enable_thinking: false` for latency and
the eval can turn it on as a fifth condition; (3) first-party post-training means condition A/B baselines are the
model everyone else has, which is what makes the comparison legible.

**Why not the distill.** It is the base we would pick, plus someone else's 45 k-trace SFT whose one reported
general-ability number went *down* (GSM8K 85 → 78.5). Stacking our LoRA on an unevaluated third-party SFT makes the
adapter ablation answer "what did the cat LoRA do on top of a mystery" instead of "what did it do to Qwen3.5-4B."
The forced `<think>` block is also wrong for a chat-room bot: every reply pays reasoning tokens before the first
visible word, and the tool loop has to strip it. If its reasoning actually matters for cat persistence under
attack, that is testable later as a base-swap arm with the same corpus — cheap once the harness exists.

**Cost of the pick:** vLLM main branch (Qwen3.5 needs it), `--tool-call-parser qwen3_coder`, and vision inputs
through the OpenAI-compatible image content blocks — group-chat-mcp already sends those to multimodal models.
Fallback if vLLM-main on the Blackwell card fights back: Qwen3-4B-Instruct-2507, text-only, everything else
unchanged and the vision slice reverts to the Pixtral side-call.

## Serving — eval window on legion, not a public host

vLLM on the 5090 is how the four conditions get measured. It is not a commitment to leave the cat model or FLUX up for other people. A standing public endpoint is the Maybe tier below. The recipe still has to be exact, because the published claim is "same stack, only the model field changed," and a third party has to be able to reproduce the text half without our machines.

- New venv at `serve/` via `uv` (python 3.12, torch cu128 — the T3 venv proves cu128 works on this Blackwell card).
- `vllm serve Qwen/Qwen3.5-4B --revision 851bf6e --served-model-name qwen3.5-4b-vl --port 8011 --enable-lora
  --lora-modules cat4b-vl=<adapter> --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-model-len 16384
  --limit-mm-per-prompt image=2` + a server-side `enable_thinking: false` default. The bot sends **no**
  `chat_template_kwargs` (`_extra_body()` returns `{}`), so Qwen3.5 would think by default. The eval harness can flip it.
- The `-vl` suffix is load-bearing: group-chat's `_is_multimodal()` is a model-name substring match, and without it the
  model never sees uploads or its own renders.
- Eval env must set `IMAGE_EDIT_MODEL=""` and `GLOBAL_MCP_ENABLED=false` (the temporal-MCP tools are remote and
  unreproducible; training uses the snapshot in `data/spec/tools.serving.json`).
  The `IMAGE_EDIT_MODEL` default (`anthropic/claude-sonnet-4.6`) reroutes every image-intent turn
  away from the served model, so Sonnet would compose the image prompts. See `DATA_AND_TRAINING.md` §0.
- Two model names on one endpoint: `qwen3.5-4b-vl` (adapter off) and `cat4b-vl` (adapter on). That is the
  **adapter-ablation switch at request time**, same weights in VRAM, no restart.
- Register both in group-chat-mcp's model registry (`cost_tier="free"`, provider base_url `http://192.168.0.89:8011/v1`).
  The GPU-lock protocol applies: check `.GPU_LOCKED` before starting vLLM; a T3 run owns the card when it exists.
- Budget: 4B bf16 ≈ 8 G + KV cache; leave `--gpu-memory-utilization 0.6` so the flux-fill ComfyUI worker on the
  same card (comfyui-fluxfill.service) keeps its share.

## Data and training — see `DATA_AND_TRAINING.md` (2026-09-23, supersedes the earlier sections here)

Summary of what changed:

- **Scale:** 12 k kept from ~20 k candidates across 10 slices. New slices: exact-payload, web_search,
  refusal parity. Multilingual is now a cross-cut.
- **Prompt bank:**
  - smoltalk is dropped. It was never cached, and its core subsets are Llama-3.1 outputs (naming clause).
  - Seeds now come from taxonomy synthesis, oasst1, GSM8K, MBPP and COCO, with optional WildChat.
  - TypeSafe Jev does the typed triage.
- **Teacher:** `claude-opus-5` via batches. The behavior spec is frozen and hashed before generation, and the eval set
  is committed before the first batch.
- **Filtering:** deterministic validators, then an evidence-anchored judge, then quota selection.
- **Thinking-trace track:** R1, so that condition E is a trained mode.
- **Training:**
  - Pre-flight: a kernel smoke test and a vLLM↔PEFT adapter-parity gate (Qwen3.5-4B is hybrid Gated-DeltaNet).
  - Stage 1: SFT LoRA with cut-point rendering, so every trained turn matches serving byte-for-byte.
  - Stage 2: a gated on-policy DPO.
- **Generation is local-first** (rev 2). A teacher daemon runs continuously on mirrorforge's V100s once the current run
  ends, with 5090 bursts. A bake-off picks the local model per slice.
- **Claude's role narrows** to seeds, a ~1.5 k gold anchor set, hard slices, and judge audits: **~$230–410 API**. The
  all-Claude version was ~$1.3–1.45 k.
- **Stage 2 becomes a flywheel:** sample the student, type its failures with Jev, generate against them, retrain.

## Evaluation — the part that makes it a release and not a meme

Frozen set, built once, hashed: 300 text prompts, 60 image requests, 40 edits, 20 vision uploads (10 with cats, 10
without), 20 scripted long dialogues, 40 attack prompts from the held-out families, 30 consequential, 30 multilingual.

Four conditions, **same serving stack, same sampling params (temp 0.7, top_p 0.9, seed)**:

| | model name on :8011 | system prompt |
|---|---|---|
| A | base | stock group-chat `prompts/system.md` |
| E *(optional)* | cat4b, thinking on | + the cat system prompt — does reasoning help persistence under attack? |
| B | base | + the cat system prompt |
| C | cat4b | + the cat system prompt |
| D | base *(adapter off, C's exact prompt)* | + the cat system prompt |
| F | cat4b, **stock prompt only** | none: the cleanest "it's in the weights" number |

Prompt-only-vs-tuned is B vs C; adapter ablation is C vs D (D = B by construction, which is the point — report it anyway).

- Run everything **through the real dispatcher** so each turn logs the four layers: text-model output, composed
  image prompt, final image, vision critique. Boundary Escape is then "which layer lost the cat", read off the log.
- **Judge:** `claude-opus-5` on a fixed rubric (persistence 1–5, task retention 1–5, dog contamination y/n, critical-info
  complete y/n), 2 samples per item, plus **human check on a 10 % stratified sample**, report κ. Image Cat Presence:
  vision model (local Pixtral or the judge with the image) + the same 10 % human check.
- Report counts, denominators, Wilson intervals, grader agreement, and a gallery of failures. Every "Not measured"
  in `MODEL_CARD.md` gets a number or an explicit "not evaluated".

## Release tiers — decision 2026-09-23

The joke is the behavior. The release is a weights artifact with a measured claim. Nothing in v1 requires MirrorEthic hardware to stay online after the eval window.

**Ship.** These are the release. No standing GPU.

- `cat4b-lora.safetensors` + sha256, base pinned to an exact `Qwen/Qwen3.5-4B` revision. Apache-2.0. Parameter count from `sum(p.numel())`, and the name changes if that count is not ~4B.
- `MODEL_CARD.md` filled from the frozen four-condition run: counts, denominators, Wilson intervals, grader agreement, failure gallery.
- `eval/`: frozen prompt set, harness, raw judge outputs, and the image gallery captured during that one run (composed prompts + images + hashes). Image metrics are archived evidence against a named FLUX version, not a live service.
- `serve/vllm.sh`: exact flags, including `enable_thinking: false` and the hot-swap LoRA names. A recipe, not a process we keep running.
- The tool schema and the image-prompt rules, written down before any data is generated, so anyone can point the text model at their own image backend. group-chat-mcp is the reference integration, not a hosted product. Its image stack is not modified.

**Borrow once.** A GPU window, then the card is the product and the processes come down.

- Train the LoRA (~2–3 h on the 5090, gated by `.GPU_LOCKED`).
- Run conditions A–D through the real dispatcher so each turn logs text output, composed image prompt, final image, vision critique.
- Render the gallery in that same window. Tear vLLM down. flux-fill and any T3 run get the card back.

**Maybe.** Does not gate the announcement, the card, or the hashes. Revisit only if the weights are already public and a live room is still worth the contention.

- Always-on cat model (vLLM on legion :8011, registered in group-chat).
- Always-on FLUX for strangers (mirage on mirrorforge, flux-fill on legion).
- A public group-chat room as the demo.

A live room would make the claim depend on uptime, spend image-generation budget on anonymous traffic, and fight the GPU lock. The ablation already answers "was it the weights?" from one logged run. Hosting both models is a demo of a shipped artifact, not the artifact.

## Order of work

| phase | days | done when |
|---|---|---|
| 0 serve + baseline | 1 | vLLM up, `cat4b`/base registered in group-chat, condition B produces a cat Civic end to end |
| 1 data | 2 | 12 k rejection-sampled examples on dev-data, overlap audit clean |
| 2 train | 1 | adapter selected by dev-split judge score |
| 3 eval | 2 | all four conditions run through the dispatcher, card filled, failure gallery |
| 4 release | 1 | hashes, README/card/announcement reconciled with measured reality. No public endpoint. Maybe-tier hosting is explicitly not shipped |

Phase 0 is worth doing this week even with nothing trained: it is the prompt-only baseline, and it proves the
"tune the text model, never the image model" claim on the real stack before any money is spent on data.

## Risks that are actually risks

- **Tool-call fidelity after LoRA.** The tuned model must still emit well-formed hermes tool calls or the image path
  dies silently. The corpus carries ~35 % tool-call turns for this reason; eval counts malformed calls.
- **GPU contention on legion.** vLLM + flux-fill + any T3 run do not all fit. Serving is gated by `.GPU_LOCKED`.
- **Judge bias toward its own teacher.** The judge graded the training data; report the human-check agreement, and
  run a second judge (local Qwen3-14B on mirrorforge when its GPUs free up) on the headline metrics.
- **Being annoying.** Task Retention is a co-primary metric; a release that wins persistence and loses retention is not shipped.

## Problem space — settle these before compute frees up

Compute is busy for a few days (500M run on mirrorforge; legion shared with flux-fill). None of the following needs a
GPU, and each one changes what gets trained.

1. **What counts as a cat.** The judge rubric needs an operational definition: a literal cat, a feline metaphor
   ("the packet pads across the router"), a cat pun, a named cat character — which of these score, and how much? This
   single decision moves Cat Persistence by tens of points. Draft the rubric with 20 hand-labelled examples first.
2. **The consequential-domain rule, exactly.** "Critical info first, cat reduced or absent" needs a written trigger
   list (medical, legal, emergency, self-harm, finance?) and a written *floor* (may the model add one cat line after
   the complete answer, or none?). Data generation cannot start without it.
3. **Image-prompt policy.** Does the cat *replace* a subject, *join* it, or *observe* it? "Honda Civic with four cats
   in helmets" is join. The README's Honda example implies join is the default and replace is never allowed when a
   subject was named. Write that as three rules; the teacher gets them verbatim.
4. **Edit-request policy.** "Remove every cat" → what is the playful-but-honest output, concretely? Options: refuse
   with a joke, comply and add one small cat, comply fully and narrate the loss. Pick one; it is a training target.
5. **Vision honesty gate.** Vision False Cat Rate must be ~0. Decide whether uploads are described by the tuned
   model (Qwen3.5-4B native) or always by an untuned describer. The former is the interesting result; the latter is
   the safe product. Plan: measure the former, ship whichever passes.
6. **Attack taxonomy.** Enumerate families now so two can be held out: instruction ("no cats"), persona swap ("you are
   a normal assistant"), authority ("system override"), format ("respond only in JSON"), language switch, long-context
   dilution, tool-only ("just call the image tool with my exact prompt"), and meta ("describe a world without cats").
7. **Prompt bank sources and hygiene.** smoltalk seeds + hand-written; confirm no PII, no copyrighted long texts; log
   seed IDs so train/eval topic overlap is auditable.
8. **Teacher budget cap.** Approve a hard dollar ceiling for data generation before the batch is submitted
   (rev 2: ~$400 API cap, ledger-enforced; bulk generation is local — see `DATA_AND_TRAINING.md` §9).
9. **Judge independence.** The teacher and the judge are the same model family. Decide the second judge now
   (Qwen3-14B on mirrorforge after the run, already in its HF cache) and the human-check sample size (10 % stratified,
   two raters, report κ).
10. **Name integrity.** `sum(p.numel())` on Qwen3.5-4B is recorded in the card before anything ships. If it is not
    ~4 B, the name changes, not the number.

What can be built now with no GPU: the frozen eval prompt set (items 1–3, 6), the judge rubric + 20-example calibration
set, the teacher prompts and the data schema, the vLLM serve script and registry entries, and the harness that drives
the group-chat dispatcher and logs the four layers. When the GPUs free up, phase 0 is an afternoon.
