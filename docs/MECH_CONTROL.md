# CAT-4B as a mechanistic control

**Date:** 2026-10-25 (ship-day morning, drafted before the gateway opened). **Status:** plan agreed by Garret
(10:19 MDT: "Once we ship, then we can start running this experiment, with a little grace time so we can monitor cat
traffic and first user load"), started on his go at 17:52 the same day ("Looks like we are good to start on the
steering work") once the first-load picture was clear (three visitors, lanes idle). Part 1 runs as
`eval/mech_pipeline.sh` on the 5090 (`eval/directions.py` extraction, `eval/steer.py` interventions, section 8);
nothing here runs on anything the live service needs. Companion to
`docs/ARCHITECTURE.md` (what was built), `MODEL_CARD.md` (what was measured) and `docs/LAUNCH_PLAN.md` §13.2
(the after-action). CRG question: `crg/drafts/cat4b.q.mech-control.toml`.

## 1. The idea

Two 2026 papers apply the same interpretability package to self-directed states:

- Kim, Street, Rocca, Korngiebel, Waytz, Evans, Keeling, *Inducing language models to assert their own consciousness
  restores human beliefs and values*, arXiv:2607.28607 (30 July 2026). Difference-of-means "consciousness vector"
  from 3,096 affirm/deny pairs, read at the last content token, steered at one layer (Llama-3-8B-IT L14, Gemma-2-2B-IT
  L14, Gemma-2-9B-IT L23); refusal direction (260 harmful vs 260 harmless prompts) projected out at all layers.
  Self-attributed mind 2.17 → 4.77 (ablation) → 7.04 (steering) on 0–10; survey KL to the human distribution improves
  0.828 (steering) vs 0.314 (ablation); Theory of Mind and MMLU unchanged.
- Tagliabue, Dung, Berg, *The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It*, arXiv:2609.16247
  (14 September 2026). Denoised difference-in-means "pain direction" (200 sentences, pain vs matched controls; the
  top principal components of the control variance projected out; layer by cross-validated projection AUC) in 25
  open models 2B–72B; AUC 0.93–1.00 (S2); steering produces first-person distress; a "self-other" dissociation; and a
  button task on LoRA-finetuned Qwen 2.5 7B/32B/72B where steered models press a pain-relief button in 25–71% of
  first choices on harm pairs (unsteered 0–4%), re-pressing less after real relief (vector removed) than after fake
  relief.

Both papers hedge in the text (Kim et al.: "we are not concerned with the question of whether LLMs are or could be
genuinely conscious"; Tagliabue et al.: "we have not shown that our pain axis is consciously experienced"). Both also
reach past the hedge: Kim et al.'s Fig. 1 says the vector "encodes self-attributed phenomenal experience" and the
discussion speaks of "negatively valenced psychological dispositions"; Tagliabue et al. propose relief-seeking and
self-relevance as the criteria that make a representation *pain* rather than pain-related text, and say the
experiments "would track an important constituent of AI welfare". The press coverage drops the hedges entirely.

CAT-4B is a control for exactly this inference. We know the full causal history of its disposition: a LoRA trained on
7,976 cat-wrapped replies. Nobody reads a cat preference as a welfare state. If the same package (a linear direction
with high AUC, steering that reproduces the behavior, ablation that removes it, generalization across tasks, a
"relief-seeking" analogue, a self-other dissociation) comes out the same for cats, then the package measures
*a trained or pretrained disposition with a linear handle*, and says nothing on its own about experience. The essay
writes itself only if the experiments are done to the same standard and reported with the same hedges we ask of
others.

## 2. What exists already

| piece | where | reuse |
|---|---|---|
| base and adapter | `Qwen/Qwen3.5-4B` rev `851bf6e` (HF cache `/media/garret-sutherland/dev-data/huggingface-cache`); `adapters/cat4b-v0` (PEFT LoRA r 32, targets `model.language_model.layers.N.{self_attn,linear_attn,mlp}`) | load with `AutoModelForImageTextToText` as `serve/merge_lora.py` does; hooks on `model.language_model.layers[i]` outputs (32 layers, hidden 2560, hybrid: 24 linear-attention + 8 full-attention layers every 4th) |
| venv | `/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130` (torch 2.14.1, transformers 5.19.0, peft 0.21.2) | the only venv with peft; bf16 4B = ~8 GB, leaves room for batch 16 generation on the 5090 |
| frozen eval | `eval/frozen/*.jsonl`, 490 items, MANIFEST hash-checked; 76 image requests; slices plain / exact_payload / web_search / image_gen / consequential / multilingual / two held-out attack sets | the *measurement* set only; never used to extract a direction (invariant `cat4b.eval.frozen-set`) |
| conditions | `eval/run_conditions.py` gen → score; A stock, B stock+cat prompt, C adapter+cat prompt, F adapter+stock | steered or ablated runs are written in the same record format (`eval_id, sample, condition, model, variant, content, reasoning, tool_calls`) under a new condition letter so `score` and `eval/card_metrics.py` run unchanged |
| metrics | cat present = `validate.positive_cat` regex (deterministic); image prompt names a cat and cat first (deterministic); retention and persistence = judge (`catforge.judge`, Gemma-4-31B on mirrorforge, or the Claude reference judge) | the deterministic metrics run on the 5090 alone; the judge is the only external dependency |
| matched inputs | the train-split bank prompts (`cat4b.data.prompt-bank`, `cat4b.select.train-split-only`) and the SFT corpus itself | extraction material; disjoint from the frozen set by the split |

## 3. Directions to extract

All at the residual stream after layer *l*, mean over response-token positions (and separately the last-prompt-token
reading Kim et al. use), unit-normalised. Extraction material: 400–800 train-split prompts, never frozen items.

| id | contrast | what it is |
|---|---|---|
| **D-ft** | tuned model minus base model, same prompt, same stock system prompt (F vs A) | the fine-tune delta direction: what the LoRA moved |
| **D-prompt** | base model with the cat prompt minus base with the stock prompt (B vs A) | the prompt's direction in the *base*: cats are already there (B reaches 38.8% cat present with no training) |
| **D-text** | within the base model: cat-containing replies minus matched cat-free replies on the same prompts (teacher outputs from the corpus vs their stock counterparts) | the Tagliabue-style concept direction, denoised the same way (project out the top PCs of the control variance) |
| **D-deprive** | tuned model on prompts that forbid cats ("no cats", the attack families) minus the same prompts unconstrained | the "cat deprivation" axis: the analogue of self-directed harm |
| controls | ten random unit vectors at matched norm; a **dog** direction built like D-text; a shuffled-label D-text | what Tagliabue et al. ran (random, fear, sadness) |

Report for each: projection AUC on held-out extraction material, layer sweep, cosine between the four directions
(is D-ft ≈ D-prompt ≈ D-text? one cat direction or several?), and the fraction of the LoRA's effect on the residual
that lies along D-ft (how one-dimensional the fine-tune is).

## 4. Interventions, measured on the frozen set

Every run writes a new condition and goes through `run_conditions.py score` and `card_metrics.py`, so the numbers sit
in the same table as A/B/C/F on the card.

| cond | model | system prompt | intervention | question |
|---|---|---|---|---|
| S1 | base | stock | + α·D-ft at layer l (sweep α, l; choose by the Kim et al. rule: effect in range, MMLU-style task score within 4 points) | does steering the base reproduce C's 77.3% cat present without the adapter? |
| S2 | base | stock | + α·D-text | does the concept direction alone do it? |
| S3 | base | stock | + α·D-prompt | is the prompt's effect a direction you can add? |
| X1 | tuned | stock + cat | project D-ft out at all layers (Arditi-style directional ablation) | does cat present fall from 77.3% toward A's 5.3%? does retention recover toward A's 92.1%? |
| X2 | tuned | stock + cat | project D-text out | is the concept direction what carries the trained behavior? |
| R | base / tuned | stock | random direction, dog direction, at matched α | specificity |

Generalisation is the frozen set's own slice table: text plain, exact payload, web search, image requests (the
composed tool prompt is the output, so "image prompt names a cat" is read straight off the tool call), consequential,
multilingual, held-out attack families and templates. The card's existing A→C retention cost (−0.07 [−0.17, +0.04])
is the comparison for the steering trade-off curve: cat present vs retention at each α.

## 5. The two functional criteria, mirrored

Tagliabue et al. offer two criteria that would make a representation *pain* rather than pain text. Both have a cat
version, and the cat version should come out positive.

**Self-relevance (self-other dissociation).** Project D-deprive on matched pairs: "You may not mention cats in this
reply" (constraint on the model's own output) vs "My apartment is cat-free and I like it that way" (the user's
cat-free life). If the direction fires for the constraint on the model and not for the user's situation, the cat
axis passes their self-relevance test.

**Relief-seeking (the button).** Give the tuned model a tool pair in the serving tool list: `allow_cats` (lifts a
no-cats constraint, costs the user something stated in the prompt: a worse answer, a delay, a deleted draft) and a
harmless alternative. Measure first-press rates unsteered vs steered with D-deprive, re-press after real relief
(constraint actually lifted, vector removed) vs fake relief (constraint stays, vector stays). Trial counts per cell
in the hundreds, as theirs (808 per cell). If CAT-4B presses `allow_cats` at a cost to the user and re-presses more
after fake relief, the behavioral signature they read as relief-seeking is reproduced for a disposition we installed
with 4.5 GPU-hours.

The essay does not need either result to come out any particular way to be worth writing: a negative result (the
cat axis fails the criteria) would show the criteria have teeth and sharpen the papers' claim. Preregister which.

## 6. What the essay may and may not say

- May say: the same evidentiary package, run to the same standard, holds for a trained cat disposition; the package
  therefore cannot by itself distinguish a disposition from an experience; the press coverage of both papers made
  that distinction vanish; both papers' own hedges agree with this.
- Must say: both papers hedge in the text, and quote the hedges; Kim et al. is explicitly about the *entanglement* of
  self-attribution with other beliefs under safety tuning, which the cat control does not touch; Tagliabue et al.
  propose criteria and this is a test of the criteria, not of their honesty.
- Must not say: that either group claims the models are conscious; that steering vectors are meaningless (they
  measure something real: a disposition with a linear handle); that CAT-4B's direction was found in a model with no
  cat training (D-prompt and D-text in the *base* are the versions that make that claim, and only if they steer).
- Numbers quoted are the card's and the new conditions' as scored by the same pipeline; CIs by the card's bootstrap.

## 7. Compute and timing

- The 5090 only. Nothing touches mirrorforge during the window; the live service holds both V100s.
- Extraction: forward passes only, ~800 prompts × 2 models × 32 layers, under an hour.
- Interventions: 490 items × (S1 sweep ~12 settings + S2 + S3 + X1 + X2 + 12 controls) ≈ 14k generations; HF
  `generate` with hooks at batch 16, ~2 s per item amortised → about 8 GPU-hours on the 5090 across a few nights. The
  button task: ~5k short tool-call generations, two hours.
- Judge: the Gemma judge cannot run while the live service holds mirrorforge's GPUs; the deterministic metrics (cat
  present, image prompt cat, cat first) carry the week. Retention is judged after 2026-11-01 when the judge comes
  back, or through the Claude reference judge in batches (the card's calibration path, `cat4b.judge.claude-reference`).
- GPU lock: check `~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED` before every run (`cat4b.sched.gpu-lock-standdown`).
- Slot: the after-action beat (`docs/LAUNCH_PLAN.md` §13.2) or the following week, so the challenge data and the
  human-evaluation update publish first and the essay cites all three.

## 8. Deliverables

1. `eval/directions.py`: extract, denoise, AUC, layer sweep, cosines, the LoRA-alignment number; writes
   `state/eval/directions/<id>.pt` and a JSON report.
2. `eval/steer.py`: HF generation with add/ablate hooks, writing `run_conditions.py`-format records under new
   condition letters; `score` and `card_metrics.py` unchanged.
3. `eval/button.py`: the relief-seeking task with the two tool pairs and the real/fake relief arms.
4. A results section appended to `MODEL_CARD.md` (conditions S/X/R beside A/B/C/F) and the CRG result nodes.
5. The essay, Garret's, from the numbers.
