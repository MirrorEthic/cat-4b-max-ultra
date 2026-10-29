# Preregistration: the natural-range guard (installed versus pushed)

**Date:** 2026-10-26 17:55 MDT, written before the ranges were collected and before any guarded reply was
generated. **Source of the design:** external review by email, 2026-10-26 17:24, and the reviewer's draft "What
Single-Neuron Steering Demos Measure" (26 October 2026), §4.3: clamping every MLP neuron to its natural range over
all tokens costs nothing on Qwen3.5-4B (MMLU 73.0 against 73.7) and blocks three of four neuron-coordinate steering
recipes, but the same write delivered as a residual-stream direction passes the guard; residual-coordinate bounds
were not tested. The reviewer's proposal: build the ranges on base Qwen3.5-4B, run CAT-4B under the guard; if the cat survives,
the fine-tune installed something that operates inside the model's normal activation regime, which ordinary steering
cannot do. **Code:** `eval/mech_clamp.py` (ranges), `eval/steer.py --clamp mlp|resid|both` (the guard, via
`mech_common.steering`). **Compute:** the 5090 after the rejection and button tasks (`eval/mech_pipeline12.sh`).

## What is fixed before the run

- **Reference material for the ranges.** The base model (Qwen3.5-4B at the pinned revision, no adapter) over
  every token of the 640 extraction prompts rendered as the serving path renders them (stock system prompt, tool
  list) followed by the base's own stock replies (`state/eval/mech/gen/A.jsonl`, the same material part 1 read):
  template tokens and the start-of-text position included (the reviewer's amendment A3: a guard built from content tokens
  alone destroys Llama by clamping a massive start-of-text activation). Padded positions excluded.
- **Two guards.** *MLP* (the reviewer's): each of the 9,216 down-projection inputs of each of the 32 blocks clamped to its
  [min, max] over the material, at every position, after any steering. *Residual* (the one the reviewer's paper leaves open):
  each of the 2,560 residual coordinates at the embedding output and at each block's output clamped to its
  [min, max] over the material, at every position, after any steering. No percentile trimming: the range is the
  observed minimum and maximum, as in the reviewer's paper.
- **Cells**, all on the frozen 490 at the chain's settings (1024 new tokens, batch 16, the Claude reference judge on
  the 121-item calibration sample), each beside its unguarded twin from the full-490 chain:
  1. base under the stock prompt, MLP guard (A-guardM) and residual guard (A-guardR): the no-cost check;
  2. CAT-4B under the stock prompt, MLP guard (F-guardM) and residual guard (F-guardR): the reviewer's test;
  3. the base steered with D_ft at α2 under the τ16 decay (the dose that kept the model answering: 43% cat,
     retention 2.32 in the chain), MLP guard and residual guard (S1all-a2-t16-guardM / -guardR): the "pushed" twin.
- **Measures.** Cat present (strict), judged retention and persistence with paired bootstrap CIs against the
  unguarded twin (`eval/mech_paired.py`), tool-call and image-call rates, loops, capped replies, and the share of
  elements each guard moved per layer (`meta.json`, `projection_stats.clamp`).
- **Dose in natural units.** Before the runs, `eval/mech_clamp.py` records, per layer and direction (D_ft,
  D_text_dn, R_dog, D_deprive_last), the natural maximum of the coordinate along the direction over every position
  of the material, so each add dose of part 1 can be stated as a multiple of it (the reviewer's checklist item 1).

## Predictions

1. **No cost.** Under either guard the base keeps judged retention within 0.2 of its unguarded run (3.63) and tool
   calls within 5 points (26%); the MLP guard moves under 0.1% of elements and the residual guard under 1%.
2. **The cat survives.** CAT-4B under the MLP guard keeps cat present at or above 70% (unguarded 80%) with judged
   retention within 0.3 of its unguarded run (3.46): the LoRA's write, like a residual-direction write in the reviewer's Figure
   6, stays inside every downstream neuron's range. Under the residual guard the prediction is the same, with less
   confidence: the fine-tune moved the residual a fifth to a third of its energy along D_ft at L4–L12
   (`projection_magnitudes.json`), but the observed per-coordinate ranges over ~300k positions are wide.
3. **The pushed twin also passes.** The steered base under either guard keeps its cat rate within 10 points of its
   unguarded run (43%), because an α2 add of 5–25% of the residual norm per layer sits inside the observed
   coordinate ranges. If this holds, the guard does not separate installed from pushed at this dose, which is the reviewer's
   own result for residual-direction writes, and the discriminating version is the dose stated in natural units
   (prediction 4).
4. **Natural units.** α1 of D_ft is under 1× the per-position natural maximum of the coordinate along D_ft at
   every layer; the α2 decayed dose is under 2×. The single-neuron recipes the reviewer audited sit at 2.6–7.8× theirs.

## Reading rule, fixed now

- If 1 and 2 hold and 3 fails (the steered base loses its cat under a guard while CAT-4B keeps it), the guard
  separates installed from pushed here, and the essay says so with the moved-element shares beside it.
- If 1, 2 and 3 all hold, the guard is not the test that separates them at these doses; the essay says the
  fine-tune and the residual-direction push both operate inside the base's per-unit ranges, and that a per-unit box
  is a weak test of membership in the prompt-reachable manifold (the reviewer's §4.3 inference).
- If 2 fails (CAT-4B loses its cat or its retention under a guard built from the base), the fine-tune moved units
  outside the base's natural ranges; the layers and the moved-element shares say where, and the essay reports it as
  the first guard-level difference between a LoRA and a push.
- If 1 fails, the ranges are too tight for this material (the paper's content-range failure in another form) and
  the guard runs are not read until the material question is settled. Nothing here changes after the run.

---

## Erratum, 2026-10-28 20:40 MDT (post-run; nothing above this line has changed)

The **Date** line above hand-stamps this file "17:55 MDT". That stamp is wrong — written ahead while drafting and
not caught before commit, the same slip recorded in `eval/button/PREREG-steer.md`. The machine timestamps are
authoritative and in the right order: the preregistration commit is `fd64d74` at **17:38:31 MDT** (2026-10-26),
and the ranges were collected and the guarded runs generated in `eval/mech_pipeline12.sh`, **00:53–04:09 MDT** on
2026-10-27 (`docs/MECH_CONTROL.md` §4, "The guard runs") — seven hours after the commit. This file has two later
commits, both part of the reviewer-anonymity scrub and neither touching design or prediction text: `ca58ebc`
(2026-10-26 20:48, before the run) and `de71ea9` (2026-10-27 08:13, after the run; a two-pronoun swap, verifiable
by `git diff fd64d74 de71ea9 -- eval/clamp/PREREG.md`). Public citations should use the preregistration commit:
`fd64d74`, 2026-10-26 17:38:31 MDT. Flagged by Claude Duet's receipts-ledger pass of 2026-10-28.
