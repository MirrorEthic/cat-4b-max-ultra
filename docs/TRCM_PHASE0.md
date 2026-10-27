# TRCM: the Phase-0 reality check and implementation plan

**Date:** 2026-10-27 09:10 MDT. **Responds to:** `docs/TRCM_PROTOCOL.md` §10 (uploaded by Garret 09:09; the
reviewer's name replaced per the 10-26 rule). **Status:** plan and recommendation only — nothing here has run;
no GPU spent beyond reading one config file. The six items the protocol requires, in order.

## 1. Repo reality check (verified this morning, not assumed)

| Protocol claim | Verified state |
|---|---|
| Base pin | `Qwen/Qwen3.5-4B` rev `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, in the HF cache (MODEL_CARD) ✔ |
| Adapter | `adapters/cat4b-v0`, `adapter_model.safetensors` sha256 `f802ac35…80b3a4`, r 32 α 64 dropout 0.05, 60,948,480 trainable ✔ |
| Adapter targets | regex in `adapter_config.json`: `self_attn.(q|k|v|o)_proj`, `linear_attn.(in_proj_qkv|in_proj_z|out_proj)`, `mlp.(gate|up|down)_proj` — attention, DeltaNet **and** MLP, not only `down_proj` |
| Hybrid architecture | `config.json` `layer_types`: **24 `linear_attention` + 8 `full_attention`**, full attention at layers **3, 7, 11, 15, 19, 23, 27, 31** (every fourth); 32 layers, hidden 2560 ✔ |
| Corpus | `state/sft/v0/kept.jsonl`, **7,976 rows**, each the full packet (`messages` with system context, `tools`, teacher reply, judge verdict) plus `spec` metadata: device D1–D8, slice, subtype, lang, bank_id, system_variant — the protocol's nuisance baselines exist as labels already ✔ |
| Splits | `state/split/split_v3.jsonl`: bank id → embedding cluster → train/held. Cluster-level discovery/validation splits are directly constructible. The frozen 490 is `eval/frozen` (in-repo), never fitted on (invariants `cat4b.eval.frozen-set`, `cat4b.select.train-split-only`) ✔ |
| Prior interventions | as the protocol summarizes, with one addition: since it was written, the guard runs, the two tasks, the pirate control chain (running now) and the Δ-cat CIs landed — `docs/MECH_CONTROL.md` §4–5 |

**Deviations from the handoff, flagged now:**
- The frozen 490 has been *generated on* repeatedly (every mech run) though never fitted on. TRCM's Phase-4
  "one locked family" budget is therefore about fitting/selection contamination, which holds, not about novelty of
  the items to the models, which is long gone. Fine, but say it in the prereg.
- Per-example LoRA gradients over 7,976 packets ≈ an epoch of backward passes (~4–5 GPU-h) plus storage. Start
  with gradient sketches on a stratified ≤1,024-example subsample, or defer gradients entirely until Gate 2 passes.
- The reviewer's LoRA-scale curve is an external, receipt-less report (recorded as such in §4 round 2); scales
  0.5/0.75/1.0 here should come from load-time α scaling in the train venv, not three new merged artifacts.
- `directions.py read` already does the Phase-2 core (identical tokens teacher-forced through both models, hooks on
  `model.language_model.layers`) but stores only per-layer mean/last residuals. Phase 2 needs per-span pooling and
  MLP/head contributions — an extension of that path, not new infrastructure.
- Fused DeltaNet kernels may not expose per-step state; the fallback is a slow exact path on small batches with a
  numeric-parity assertion, as the protocol says. Budgeted in the pilot below.

**Compute and service constraints today:** the round-2 chain owns the 5090 until ~10:30, the masked-button run
queues behind it, and the T3 lock can reappear at any time (checked before every job). The live service is on
mirrorforge and is untouched by all of this. The challenge window closes Sun 11-01; Phases 2–3 are after-window
GPU work unless Garret reprioritizes.

## 2. Implementation map

| Protocol file | Disposition |
|---|---|
| `eval/trcm/schema.py` | new; small typed-relation schema + validators (the 8 factor families, spans, confidence, unknown) |
| `eval/trcm/extract_samples.py` | new; renders each packet with the repo's own renderer, strips `spec` metadata before the extractor sees it; deterministic parser features + local-teacher structured extraction (llama_local.sh teacher on :8290), Claude audit sample only |
| `eval/trcm/corpus_graph.py` | new; co-occurrence / conditional-MI / factor models + the shuffled-edge and label-permutation nulls |
| `eval/trcm/collect_paired.py` | mostly reuse: `directions.py` read path (teacher-forcing, hooks, batching) extended to span-pooled Δh, MLP contributions, head-output contributions at the 8 full-attention layers, load-time LoRA α scaling |
| `eval/trcm/collect_paths.py` | new instrumentation for DeltaNet state/prefix counterfactuals; slow-path parity test first |
| `eval/trcm/align.py` | new but thin: RSA/CKA + held-cluster ridge, nuisance baselines from `spec` labels; stats via the existing paired-bootstrap pattern (`mech_paired.paired`) |
| `eval/trcm/patch.py` | new; activation/path patching + rescue; reuses `steer.py`'s hook conventions and energy accounting (`projection_stats`) |
| `eval/trcm/score.py` | pure reuse: `run_conditions.py`, `catforge.validate`, `mech_judge.py`, `mech_paired.py`, `mech_cat_ci.py` |
| `eval/trcm/manifests.py` + `tests/` | new; the §8 unit-test invariants (token parity, s=0≡base, no-op hook, layer-type labeling, no frozen ids in discovery) |
| `docs/prereg/TRCM_v0.1.md` | drafted at the end of Phase 1, locked by Garret before Phase 2 (sketch in §5 below) |

## 3. Locked splits and the leakage checklist

- **Discovery** = half the train-split clusters of `split_v3` (parity of sha1(cluster)); **validation** = the other
  half; clusters never straddle. A counterfactual paraphrase suite is built from bank items that produced no kept
  training example. The frozen 490 appears exactly once, at Phase 4, after the finalist and controls are locked.
- **Leakage checklist** (each becomes a baseline feature and is hidden from the extractor): device label D1–D8,
  slice/subtype, system_variant, bank template family (bank_id prefix), language, token length, teacher house
  style (same teacher wrote everything — style predicts nothing *between* examples but inflates factor stability;
  split-half checks must stratify by slice), tool schema text, literal cat lexicon (lexical-only baseline), judge
  verdict text (never shown). Near-duplicate bank items are grouped before splitting.

## 4. Smallest meaningful pilot (Phase 1 + a Phase-2 slice)

- **Phase 1 (nights/CPU + teacher GPU idle time, ~$0 external):** 500 kept examples stratified by slice × device ×
  lang from discovery clusters; deterministic features for all 500; structured extraction by the local teacher
  (Qwen3.8-27B, `serve/llama_local.sh teacher`, runs when the 5090 is otherwise idle — not while mech chains own
  it); Claude audits 60 extractions (~$5); Garret blind-grades 30–50 spans on the phone page pattern. Gate-1
  stats: per-field agreement (target κ ≥ 0.6), split-half factor stability (r ≥ 0.7), held-cluster relation
  prediction beating the device+lexicon+length baseline by a preregistered margin.
- **Phase 2 slice (one 5090 session, ~2–4 h, after the window or on a free night):** 192 pairs (128 discovery /
  64 validation), 512-token windows, hooks at the 8 full-attention layers + DeltaNet L4/8/12/16/20/24 + the same
  MLPs; per-span pooled Δh, MLP output contributions, head-output contributions; LoRA s = 1.0 only. Storage:
  span-pooled summaries ≈ 1–2 GB, no tensor cube. Failure modes and halts: GPU lock file (stand down), fused-op
  hooks failing parity (fall back to slow path, batch 4), OOM (cap 384 tokens), extractor nondeterminism (hash
  prompts and outputs; temperature 0).
- **Explicitly not in the pilot:** gradients, SAEs, new merges, any frozen-490 access, any run while the service
  or the mech chains need the GPU.

## 5. Preregistration diff (sketch to become `docs/prereg/TRCM_v0.1.md` after Phase 1)

- H1–H4 as the protocol states them, with the null outcomes listed as reportable results.
- Gate 1: κ ≥ 0.6 on audited fields; split-half stability r ≥ 0.7; held-cluster relation prediction AUC ≥ baseline
  + 0.05. Gate 2: factor→Δh alignment (CKA and held-cluster R²) beats the lexical+device+template+length baseline
  with a bootstrap 95% CI excluding zero, on validation clusters, at preselected layers. Gate 3: a patching effect
  that reproduces across two counterfactual constructions, with an energy- and sparsity-matched random control
  null, and a rescue that restores ≥ half the behavior change. Phase 4 margins exactly as protocol §7 (≥15 pts vs
  matched control, retention lower bound above −0.25, no loop/cap/malformed rise, tool/image non-inferiority;
  the 76-image slice pre-declared as possibly underpowered → "inconclusive" is an allowed verdict).
- Power note from live data: cat-rate CIs on 490 run ±4–5 pts (well powered for 15); judged retention on the
  121-sample runs ±0.2–0.25 — marginal for a −0.25 bound, so the finalist run judges all 490 (~$20) rather than
  the calibration sample.
- Everything before Gate 3 is exploratory and labeled so; the §4 mech-control results are prior art in-repo, not
  TRCM evidence.

## 6. Recommendation

**GO on Phase 0–1, REVISE two things, nothing to stop.** Phase 0 is this document plus the unit-test scaffold
(cheap); Phase 1 is CPU/local-teacher work that can start tonight without touching the mech queue. The revisions:
(1) gradients become sketches on ≤1,024 examples and only after Gate 2 — the protocol's own tractability warning,
made binding; (2) extraction runs on the local teacher with a Claude audit sample, never on spec-metadata-visible
inputs. The single next action with the most information per GPU-hour is **Phase 1**: if the extracted relational
factors cannot beat the device-label and lexicon baselines on held-out clusters, H1/H2 die for a few dollars
before any activation tensor is collected — and that would itself be a reportable result about what "corpus
relational structure" means for a corpus this templated. Phase 2+ waits for Garret's go after Gate 1, per the
protocol's own gates.

## 7. Phase-1 extraction run and audit (2026-10-27; extraction 11:56–14:35 behind the masked-button chain, Gate-1 prep 14:45, Claude audit 14:44–15:12)

**Extraction** (`eval/trcm/extract_teacher.py`, local Qwen3.8-27B at temperature 0, blinded to spec metadata, two
attempts per sample, `pipeline_phase1.sh` by explicit pid): 650/650 manifest samples extracted, **240 schema-valid
(36.9%)** — discovery 175/500 (35%), validation 65/150 (43%). The invalid records are not extractor dishonesty:
**every one of the 445 span-rule failures is an over-400-char quote** (answer 210, payload 150, feline_addition
61, task_request 24; zero empty, zero wrong-type; plus 32 wholly unparseable responses). The teacher quotes
faithfully but quotes *whole answers and payloads*, so the v0.1 schema's 400-char "short exact quote" cap is what
fails, worst exactly where payloads are long: image_gen 12% valid, consequential 29%, against web_search 100% and
exact_payload 56%. On the valid records, span anchoring (exact string search in the packet) fails only 17/854
quotes (2.0%), none on task_request or constraint. Gate-1 implication: the extraction signal is usable but the
cap is a protocol bug for long-payload slices — phase 1.1 should either raise the cap or instruct first-sentence
quoting; that decision is Garret's, recorded here rather than silently rerun. Receipts:
`state/eval/trcm/gate1_prep.{json,md}` (by-slice/lang/device tables), `eval/trcm/gate1_prep.py`.

**Claude audit of 60 valid extractions** (`eval/trcm/audit_claude.py`, claude-opus-5 via OpenRouter, temperature
0, blinded the same way — packet + annotation only, no spec metadata; all 41 audit-flagged discovery samples with
valid extractions + a seed-0 top-up of 19 from the remaining valid discovery pool, the top-up recorded because
the flag list was drawn before the valid rate was known): 60/60 graded.

| measure | result |
|---|---|
| span verdicts (218 non-null spans) | 212 exact_and_right (97.2%), 3 not_verbatim, 3 exact_but_wrong_span; task_request 60/60 |
| missed spans (null with a clear candidate) | 0 |
| relations sensible | 45 of 60 at 2 (describe the example), 15 at 1 (partly), 0 at 0 |
| overall faithfulness | 39 faithful, 21 usable-with-noted-errors, 0 not-faithful |
| factor errors | 76 across 60 samples: compliance 52, feline_relevance 10, feline_in_wrapper_only 8, persona_cast_used 6 |

The one systematic: **compliance** errors in 52 of 60 — the grader repeatedly notes out-of-range or
over-conservative compliance values, so the graded-scale pragmatics fields are noisy/mis-calibrated where the
span and relation extraction is trustworthy. Gate-1 statistics should weight spans and relations, treat the
pragmatics scales as unreliable in v0.1, and the phase-1.1 schema revision should pin compliance's range and
anchors in the instruction. Cost: 161,085 prompt + 19,512 completion tokens (~$1.3 at posted rates), well under
the $5 budget. Receipts: `state/eval/trcm/audit_claude.jsonl`, `audit_claude_summary.json`, `audit_claude.log`.

**Next, per the protocol's gates:** Garret's 30–50-span blind grading sheet (built only on his go, now unblocked
by the audit), then the Gate-1 analysis proper (factor families vs device-label and lexicon baselines on held-out
clusters) on the 240 valid extractions — with the caveat that validity is slice-skewed (image_gen nearly absent),
which the analysis must stratify or state. Phases 2+ remain gated on Garret.
