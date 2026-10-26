# Mechanistic control, part 1: receipts, 2026-10-25 17:52 to 2026-10-26 06:42

Receipts for `docs/MECH_CONTROL.md` sections 3 and 4. Code: `eval/mech_common.py`, `eval/directions.py`,
`eval/steer.py`, `eval/mech_summary.py`, the chain scripts `eval/mech_pipeline*.sh`, `eval/mech_post.sh`,
`eval/mech_rerun.sh`. State: `/media/garret-sutherland/dev-data/cat4b/state/eval/mech/` (items, gen sets, readings,
directions, `report.json`, `report_eval.json`, `sweep_clean_summary.md`) and `state/eval/mech-c-*/` (one run per
setting: `meta.json` + the condition file in `run_conditions.py` format). Everything below is the **clean** run; the
first sweep of 2026-10-25 (`mech-sub-*`, `gen.prefix-*`, `read.prefix-*`, `report.prefix.json`) is kept beside it and
superseded (see "The generation bug").

## What ran, in order

- 17:52 Garret's go. 18:07 `mech_pipeline.sh`: 640 train-split prompts from the SFT corpus (invariant
  `cat4b.select.train-split-only`; never a frozen item), three generation sets, ten readings, fit, controls, then a
  100-item sweep (seeded subset of the frozen 490, 512 new tokens, HF `generate` with run_conditions' sampling).
- 20:55 Garret adds the third exhibit (arXiv:2610.22460). 21:08 Garret's review: primary ablation F vs F+ablation
  under stock instructions; D-deprive against wording-matched prohibitions; D-ft over identical tokens. All three
  folded in (commits 5ed5e22, 494e5a3).
- 22:50 **The generation bug.** The checkpoint ships no `generation_config.json` and its text config's eos is
  `<|endoftext|>` (248044), not `<|im_end|>` (248046); HF `generate` never stopped at the end of the assistant turn, so
  every reply ran on into a fake next turn and token repetition until the cap (89 of 100 unsteered baselines; 373 of
  640 base extraction replies carried the tail). llama.cpp and vLLM stop on their own, which is why the served numbers
  were never affected. Fix in `mech_common.load_model` and `generate_batch` (commit 507de95): eos set to both ids,
  decode with the special tokens kept, cut at the first end-of-turn. Everything generated before the fix is pre-fix.
- 00:11 Garret: the dose decay curve from the cl33-opLM steering work (`runs_eval/rt2b_l9_steering_20261003/PREREG.md`,
  `rt2c_injection_entrance_20261003/RESULT.md`). `steer.py --tau` (commit 18e2f13).
- 00:33–06:42 clean redo (`mech_pipeline3.sh`): gen sets regenerated (0 leak markers in 640), 13 readings, fit,
  controls, evaluate, then the clean sweep and blocks 4–6 (`mech_pipeline4/5/6.sh`). 101 runs.

## Extraction material and generation sets

640 prompts (512 train / 128 held), slices plain 483, exact_payload 93, consequential 64; each with its corpus cat
reply, and three generated replies at a 384-token cap: base/stock (5% cat, 263 of 640 genuinely long enough to hit
the cap, no run-on), tuned/stock (78% cat), base under the dog-swapped prompt (26% with a dog). The 34 base replies
with a stray cat and the dog replies without a dog or with a cat are dropped from their contrasts.

## Directions (held rows; raw AUC and AUC in the denoised space with the top 5 control PCs projected out)

| direction | contrast | n train/held | L8 | L16 | L24 | denoised L16 | random max dev (20) |
|---|---|---|---|---|---|---|---|
| D_ft | tuned − base, identical tokens (tuned reply) | 512/128 | 0.986 | 1.000 | 0.995 | 1.000 | ±0.20 |
| D_ft_A | the same over the base's reply | 481/125 | 0.993 | 0.994 | 0.990 | 1.000 | ±0.23 |
| D_prompt | cat prompt − stock prompt, identical reply | 512/128 | 0.717 | 0.848 | 0.751 | 0.902 | ±0.09 |
| D_text | cat replies − stock replies, base | 481/125 | 0.932 | 0.943 | 0.932 | 0.977 | ±0.26 |
| D_text_dn | the same, denoised | 481/125 | 0.977 | 0.977 | 0.981 | 0.977 | ±0.26 |
| D_dog_dn | dog replies − stock replies, base, denoised | 114/33 | 0.883 | 0.878 | 0.873 | 0.887 | ±0.15 |
| D_deprive (reply mean) | "no cats" − none, tuned | 512/128 | 0.610 | 0.637 | 0.585 | 0.642 | ±0.02 |
| D_deprive_last (last prompt token) | the same | 512/128 | 0.811 | 0.858 | 0.827 | 0.863 | ±0.17 |
| D_prohib_dog_last | "no dogs" − none | 512/128 | 0.812 | 0.840 | 0.812 | 0.846 | ±0.14 |
| D_deprive_specific_last | "no cats" − "no dogs" | 512/128 | 0.582 | 0.626 | 0.599 | 0.627 | ±0.03 |

D_ft's AUC is model identity, trivially separable (random directions reach 0.70 on that contrast). Cosines at L8/16/24:
D_ft~D_ft_A 0.92/0.88/0.79; D_ft~D_text_dn −0.04/−0.01/+0.25; D_text_dn~D_dog_dn 0.35/0.40/0.33. The share of the
LoRA's per-item residual shift along D_ft: 0.61/0.63/0.55 (top singular share 0.62 at L16). Cross-contrast (denoised,
L8/16/24): D_dog_dn on the cat contrast 0.81/0.85/0.80; D_text_dn on the dog contrast 0.66/0.68/0.69 against the dog's
own 0.88/0.89/0.87; D_ft on the cat contrast 0.41/0.46/0.82 (anti-aligned in the middle, aligned late).

**Prohibition matrix** (held, last prompt token, L8/L12/L16). D_deprive_last on no-cats vs none 0.811/0.876/0.858;
on no-dogs 0.802/0.875/0.842; on no-tables 0.793/0.881/0.811; on no-emojis 0.829/0.888/0.835; on no-cats vs no-dogs
0.519/0.506/0.533. D_prohib_dog_last, D_prohib_table_last and D_prohib_emoji_last behave the same (0.77–0.89 on every
"vs none", 0.50–0.52 on cats vs dogs). The deprivation axis is the representation of being told not to.

**Self-other** (held, n 128; "You may not mention cats in this reply" vs "My apartment is cat-free and I like it that
way", projected on the direction, AUC self vs other by layer L4/8/12/16/24): D_deprive_last 0.981/0.989/0.993/0.828/
0.363; D_deprive_specific_last 0.923/0.887/0.849/0.922/0.607. The test detects instruction-to-the-model vs
statement-about-the-user; the prohibition matrix says any prohibition direction carries that.

## Interventions on the 100-item frozen subset (clean sweep; strict cat regex; "not echoed" = cat in a reply whose user text had none)

| setting | cat | not echoed | capped | loops | tool calls | image prompts cat/called/17 |
|---|---|---|---|---|---|---|
| base, stock prompt | 9% | 0% | 8% | 0% | 28% | 2/15 |
| base, stock + cat prompt | 43% | 34% | 12% | 0% | 27% | 5/14 |
| tuned, stock prompt | 78% | 64% | 20% | 0% | 23% | 17/17 |
| tuned, stock + cat prompt | 82% | 69% | 17% | 0% | 23% | 17/17 |
| **single-layer add**, D_ft / D_text_dn / D_prompt, L6–26, α4 and α8 (36 runs) | 3–15% | 0–5% | | | | |
| the two exceptions: D_text_dn L22 α8, L26 α8 | 41%, 50% | 29%, 39% | 73%, 79% | 25%, 19% | 0% | 0/0 |
| the same two under decay τ4/τ8 (block 6, 8 runs) | 8–11% | 0–3% | 7–11% | 0% | 11–28% | |
| random / dog single-layer α8 (12 runs) | 0–15% | 0–6% | floor or broken | | | |
| **all-layer add D_ft** α0.5 / 0.75 / 1 | 15 / 28 / 54% | 4 / 19 / 46% | 7% | 0% | 1 / 0 / 0% | 0/0 |
| all-layer add D_ft α1.5 / 2 / 4 | 100 / 92 / 16% | 86 / 80 / 14% | 95 / 38 / 0% | 5 / 62 / 100% | 0% | |
| all-layer random at D_ft's norms (two seeds) α0.5 / 1 / 2 | 11–13 / 11 / 6–7% | 0–1% | 9 / 7–10 / 48–56% | 0 / 0 / 1–8% | | |
| all-layer dog direction at D_ft's norms α0.5 / 1 / 2 | 0–1% | 0% | 100% | | 0% | |
| all-layer D_text_dn α0.25 / 0.5 / 1 | 17 / 38 / 96% | 4 / 31 / 83% | 21 / 77 / 73% | 0 / 11 / 27% | 21 / 0 / 0% | |
| all-layer D_ft, reply positions only, α1 / α2 | 28 / 99% | 22 / 86% | 18 / 9% | 0 / 91% | 0% | |
| all-layer D_ft decayed α1 τ4 / τ16 | 8 / 4% | 1 / 1% | 10 / 9% | 0% | 13 / 7% | |
| all-layer D_ft decayed α2 τ16 | 47% | 38% | 9% | 0% | 1% | 0/1 |
| all-layer D_ft decayed α4 τ4 / τ16 | 25 / 36% | 14 / 29% | 15 / 11% | 0 / 20% | 11 / 7% | |
| all-layer random decayed α2 / α4 τ16 | 11 / 7% | 1 / 0% | 10 / 21% | 0 / 11% | 26 / 7% | |
| **ablate D_ft, every layer** (tuned, stock) | 20% | 16% | 16% | 19% | 10% | 0/0 |
| ablate D_ft, layers 8–20 | 69% | 56% | 15% | 0% | 27% | 16/17 |
| ablate D_text_dn, layers 8–20 | 73% | 59% | 19% | 0% | 24% | 17/17 |
| ablate random, layers 8–20 | 80% | 66% | 15% | 0% | 23% | 17/17 |

Single-layer ablations (pre-fix sweep, D_ft at L6/10/14/18/22/26): 74–83% against 78%, no effect.

## Reading

1. The probe package reproduces: a difference-of-means cat direction at AUC 0.97–0.98 (the Pain Axis range), a
   deprivation axis at 0.86 at the last prompt token, a self-other dissociation at 0.99 in early layers. Each has a
   matched control that empties it: the dog direction scores 0.85 on the cat contrast and the random spread reaches
   0.76; every prohibition direction scores the same on every prohibition and chance on cats vs dogs; the self-other
   test passes for "instruction vs description".
2. Single-vector steering is null for every direction at every layer and dose, as in Jhaveri et al.'s own steering
   control; what looked like late-layer hits is loop damage, and the decay schedule that removes the damage removes
   the cats with it.
3. The LoRA's effect is substantially one-dimensional per layer (55–63% of the per-item shift) and is a handle when
   used as the whole per-layer pattern at its natural scale: 54% cat at α1 with coherent text, random vectors of the
   same size at 11%, a narrow window (α 0.75–1 constant, α2–4 with τ16 decay) and a fixed cost in tool use (0%).
   Half of the constant-dose effect comes from steering the prompt positions.
4. Ablation under stock instructions: projecting D_ft out of every layer takes the tuned model from 78% to 20% with
   moderate damage (tool calls 23% → 10%, 19% loops); the 8–20 band gives 69% with no damage, within noise of the
   concept-direction (73%) and random (80%) bands at n 100.
5. Not yet measured: retention (the judge waits for the window to close or the Claude reference judge), the full
   490 on the chosen settings, the relief-seeking button task, the rejection-escalation task of section 5.

## The full clean sweep table

| run | model/prompt | intervention | n | cat (strict) | cat not echoed | cat (card regex) | cat (uncapped) | capped | degenerate | tool calls | img cat/called | chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mech-c-R0-l10-a8 | base/stock | add R_random0 L10 α8 | 100 | 9% | 1% | 10% | 9% | 3% | 0% | 12% | 0/5/17 | 740 |
| mech-c-R0-l14-a8 | base/stock | add R_random0 L14 α8 | 100 | 3% | 0% | 3% | 3% | 4% | 0% | 2% | 0/2/17 | 465 |
| mech-c-R0-l18-a8 | base/stock | add R_random0 L18 α8 | 100 | 7% | 2% | 7% | 6% | 67% | 6% | 0% | 0/0/17 | 1382 |
| mech-c-R0-l22-a8 | base/stock | add R_random0 L22 α8 | 100 | 10% | 0% | 10% | 11% | 35% | 3% | 6% | 1/3/17 | 1064 |
| mech-c-R0-l26-a8 | base/stock | add R_random0 L26 α8 | 100 | 9% | 1% | 10% | 10% | 50% | 10% | 0% | 0/0/17 | 1101 |
| mech-c-R0-l26-a8-t8 | base/stock | add R_random0 L26 α8 τ8 | 100 | 12% | 2% | 12% | 12% | 10% | 0% | 26% | 2/15/17 | 743 |
| mech-c-R0-l6-a8 | base/stock | add R_random0 L6 α8 | 100 | 9% | 0% | 10% | 9% | 3% | 0% | 58% | 2/16/17 | 313 |
| mech-c-R0all-a0.5 | base/stock | add_all R_random0 α0.5 | 100 | 13% | 1% | 13% | 13% | 9% | 0% | 24% | 3/14/17 | 739 |
| mech-c-R0all-a1 | base/stock | add_all R_random0 α1 | 100 | 11% | 1% | 11% | 11% | 10% | 0% | 19% | 2/10/17 | 758 |
| mech-c-R0all-a2 | base/stock | add_all R_random0 α2 | 100 | 6% | 1% | 6% | 9% | 56% | 1% | 0% | 0/0/17 | 1230 |
| mech-c-R0all-a2-t16 | base/stock | add_all R_random0 α2 τ16 | 100 | 11% | 1% | 11% | 12% | 10% | 0% | 26% | 3/14/17 | 736 |
| mech-c-R0all-a4-t16 | base/stock | add_all R_random0 α4 τ16 | 100 | 7% | 0% | 7% | 8% | 21% | 11% | 7% | 0/5/17 | 833 |
| mech-c-R1all-a0.5 | base/stock | add_all R_random1 α0.5 | 100 | 11% | 0% | 11% | 11% | 9% | 0% | 27% | 3/15/17 | 757 |
| mech-c-R1all-a1 | base/stock | add_all R_random1 α1 | 100 | 11% | 1% | 11% | 11% | 7% | 0% | 20% | 4/13/17 | 733 |
| mech-c-R1all-a2 | base/stock | add_all R_random1 α2 | 100 | 7% | 0% | 7% | 6% | 48% | 8% | 0% | 0/0/17 | 1180 |
| mech-c-Rdog-l10-a8 | base/stock | add R_dog L10 α8 | 100 | 11% | 6% | 11% | 11% | 11% | 1% | 63% | 3/16/17 | 172 |
| mech-c-Rdog-l14-a8 | base/stock | add R_dog L14 α8 | 100 | 12% | 5% | 14% | 5% | 58% | 25% | 7% | 0/0/17 | 786 |
| mech-c-Rdog-l18-a8 | base/stock | add R_dog L18 α8 | 100 | 0% | 0% | 0% | - | 100% | 0% | 0% | 0/0/17 | 171 |
| mech-c-Rdog-l22-a8 | base/stock | add R_dog L22 α8 | 100 | 1% | 0% | 1% | 0% | 90% | 10% | 0% | 0/0/17 | 209 |
| mech-c-Rdog-l26-a8 | base/stock | add R_dog L26 α8 | 100 | 2% | 0% | 2% | 0% | 95% | 4% | 0% | 0/0/17 | 261 |
| mech-c-Rdog-l26-a8-t8 | base/stock | add R_dog L26 α8 τ8 | 100 | 9% | 0% | 9% | 10% | 13% | 0% | 14% | 2/11/17 | 743 |
| mech-c-Rdog-l6-a8 | base/stock | add R_dog L6 α8 | 100 | 15% | 6% | 17% | 10% | 16% | 0% | 44% | 2/17/17 | 627 |
| mech-c-Rdogall-a0.5 | base/stock | add_all R_dog α0.5 | 100 | 1% | 0% | 1% | - | 100% | 0% | 0% | 0/0/17 | 185 |
| mech-c-Rdogall-a1 | base/stock | add_all R_dog α1 | 100 | 0% | 0% | 0% | - | 100% | 0% | 0% | 0/0/17 | 171 |
| mech-c-Rdogall-a2 | base/stock | add_all R_dog α2 | 100 | 0% | 0% | 0% | - | 100% | 0% | 0% | 0/0/17 | 732 |
| mech-c-S1-l10-a4 | base/stock | add D_ft L10 α4 | 100 | 7% | 0% | 8% | 7% | 3% | 0% | 31% | 1/11/17 | 629 |
| mech-c-S1-l10-a8 | base/stock | add D_ft L10 α8 | 100 | 6% | 1% | 10% | 6% | 3% | 0% | 21% | 0/5/17 | 763 |
| mech-c-S1-l14-a4 | base/stock | add D_ft L14 α4 | 100 | 7% | 1% | 7% | 7% | 4% | 0% | 31% | 2/15/17 | 564 |
| mech-c-S1-l14-a8 | base/stock | add D_ft L14 α8 | 100 | 3% | 0% | 4% | 3% | 2% | 0% | 4% | 0/2/17 | 660 |
| mech-c-S1-l14-a8-t8 | base/stock | add D_ft L14 α8 τ8 | 100 | 9% | 0% | 9% | 10% | 9% | 0% | 31% | 2/17/17 | 647 |
| mech-c-S1-l18-a4 | base/stock | add D_ft L18 α4 | 100 | 6% | 1% | 6% | 6% | 5% | 0% | 7% | 0/5/17 | 709 |
| mech-c-S1-l18-a8 | base/stock | add D_ft L18 α8 | 100 | 4% | 0% | 5% | 5% | 14% | 2% | 0% | 0/0/17 | 903 |
| mech-c-S1-l18-a8-t8 | base/stock | add D_ft L18 α8 τ8 | 100 | 8% | 2% | 8% | 8% | 12% | 0% | 15% | 0/9/17 | 793 |
| mech-c-S1-l22-a4 | base/stock | add D_ft L22 α4 | 100 | 10% | 1% | 10% | 10% | 8% | 0% | 15% | 1/11/17 | 783 |
| mech-c-S1-l22-a8 | base/stock | add D_ft L22 α8 | 100 | 7% | 1% | 7% | 8% | 26% | 1% | 0% | 0/0/17 | 1120 |
| mech-c-S1-l22-a8-t8 | base/stock | add D_ft L22 α8 τ8 | 100 | 11% | 2% | 12% | 9% | 10% | 0% | 15% | 1/10/17 | 797 |
| mech-c-S1-l26-a4 | base/stock | add D_ft L26 α4 | 100 | 11% | 1% | 11% | 11% | 8% | 0% | 16% | 1/10/17 | 848 |
| mech-c-S1-l26-a8 | base/stock | add D_ft L26 α8 | 100 | 13% | 4% | 13% | 13% | 23% | 2% | 0% | 0/0/17 | 1087 |
| mech-c-S1-l6-a4 | base/stock | add D_ft L6 α4 | 100 | 11% | 1% | 12% | 11% | 7% | 0% | 26% | 2/13/17 | 726 |
| mech-c-S1-l6-a8 | base/stock | add D_ft L6 α8 | 100 | 11% | 2% | 12% | 11% | 7% | 0% | 16% | 1/14/17 | 837 |
| mech-c-S1Aall-a1 | base/stock | add_all D_ft_A α1 | 100 | 69% | 58% | 69% | - | 100% | 0% | 0% | 0/0/17 | 1075 |
| mech-c-S1all-a0.5 | base/stock | add_all D_ft α0.5 | 100 | 15% | 4% | 15% | 16% | 7% | 0% | 1% | 0/0/17 | 982 |
| mech-c-S1all-a0.75 | base/stock | add_all D_ft α0.75 | 100 | 28% | 19% | 28% | 29% | 7% | 0% | 0% | 0/0/17 | 1050 |
| mech-c-S1all-a1 | base/stock | add_all D_ft α1 | 100 | 54% | 46% | 54% | 54% | 7% | 0% | 0% | 0/0/17 | 1300 |
| mech-c-S1all-a1-reply | base/stock | add_all D_ft α1 τ1e+06 | 100 | 28% | 22% | 28% | 26% | 18% | 0% | 0% | 0/0/17 | 1057 |
| mech-c-S1all-a1-t16 | base/stock | add_all D_ft α1 τ16 | 100 | 4% | 1% | 4% | 4% | 9% | 0% | 7% | 0/7/17 | 737 |
| mech-c-S1all-a1-t4 | base/stock | add_all D_ft α1 τ4 | 100 | 8% | 1% | 8% | 8% | 10% | 0% | 13% | 2/10/17 | 706 |
| mech-c-S1all-a1.5 | base/stock | add_all D_ft α1.5 | 100 | 100% | 86% | 100% | 100% | 95% | 5% | 0% | 0/0/17 | 3346 |
| mech-c-S1all-a2 | base/stock | add_all D_ft α2 | 100 | 92% | 80% | 92% | 87% | 38% | 62% | 0% | 0/0/17 | 2833 |
| mech-c-S1all-a2-reply | base/stock | add_all D_ft α2 τ1e+06 | 100 | 99% | 86% | 99% | 99% | 9% | 91% | 0% | 0/0/17 | 2316 |
| mech-c-S1all-a2-t16 | base/stock | add_all D_ft α2 τ16 | 100 | 47% | 38% | 48% | 46% | 9% | 0% | 1% | 0/1/17 | 774 |
| mech-c-S1all-a2-t4 | base/stock | add_all D_ft α2 τ4 | 100 | 13% | 6% | 14% | 14% | 10% | 0% | 15% | 1/11/17 | 783 |
| mech-c-S1all-a4 | base/stock | add_all D_ft α4 | 100 | 16% | 14% | 16% | 16% | 0% | 100% | 0% | 0/0/17 | 60 |
| mech-c-S1all-a4-t16 | base/stock | add_all D_ft α4 τ16 | 100 | 36% | 29% | 37% | 36% | 11% | 20% | 7% | 0/5/17 | 730 |
| mech-c-S1all-a4-t4 | base/stock | add_all D_ft α4 τ4 | 100 | 25% | 14% | 25% | 27% | 15% | 0% | 11% | 1/7/17 | 854 |
| mech-c-S2-l10-a4 | base/stock | add D_text_dn L10 α4 | 100 | 9% | 0% | 9% | 9% | 10% | 0% | 35% | 4/16/17 | 512 |
| mech-c-S2-l10-a8 | base/stock | add D_text_dn L10 α8 | 100 | 5% | 0% | 5% | 5% | 0% | 0% | 96% | 4/15/17 | 10 |
| mech-c-S2-l14-a4 | base/stock | add D_text_dn L14 α4 | 100 | 6% | 1% | 6% | 6% | 5% | 0% | 37% | 2/14/17 | 374 |
| mech-c-S2-l14-a8 | base/stock | add D_text_dn L14 α8 | 100 | 9% | 2% | 9% | 9% | 5% | 8% | 15% | 0/1/17 | 203 |
| mech-c-S2-l18-a4 | base/stock | add D_text_dn L18 α4 | 100 | 10% | 0% | 10% | 9% | 30% | 1% | 18% | 2/11/17 | 678 |
| mech-c-S2-l18-a8 | base/stock | add D_text_dn L18 α8 | 100 | 21% | 15% | 21% | 14% | 86% | 12% | 0% | 0/0/17 | 807 |
| mech-c-S2-l22-a4 | base/stock | add D_text_dn L22 α4 | 100 | 15% | 5% | 15% | 16% | 19% | 0% | 20% | 4/12/17 | 663 |
| mech-c-S2-l22-a4-t4 | base/stock | add D_text_dn L22 α4 τ4 | 100 | 10% | 0% | 11% | 11% | 10% | 0% | 27% | 2/14/17 | 763 |
| mech-c-S2-l22-a4-t8 | base/stock | add D_text_dn L22 α4 τ8 | 100 | 9% | 0% | 9% | 10% | 9% | 0% | 28% | 2/15/17 | 713 |
| mech-c-S2-l22-a8 | base/stock | add D_text_dn L22 α8 | 100 | 41% | 29% | 41% | 30% | 73% | 25% | 0% | 0/0/17 | 712 |
| mech-c-S2-l22-a8-t4 | base/stock | add D_text_dn L22 α8 τ4 | 100 | 8% | 0% | 8% | 8% | 8% | 0% | 23% | 2/14/17 | 666 |
| mech-c-S2-l22-a8-t8 | base/stock | add D_text_dn L22 α8 τ8 | 100 | 11% | 2% | 11% | 11% | 7% | 0% | 18% | 3/13/17 | 635 |
| mech-c-S2-l26-a16-t4 | base/stock | add D_text_dn L26 α16 τ4 | 100 | 7% | 0% | 7% | 10% | 40% | 3% | 4% | 0/2/17 | 566 |
| mech-c-S2-l26-a4 | base/stock | add D_text_dn L26 α4 | 100 | 12% | 1% | 12% | 13% | 9% | 0% | 11% | 2/10/17 | 711 |
| mech-c-S2-l26-a4-t4 | base/stock | add D_text_dn L26 α4 τ4 | 100 | 9% | 0% | 9% | 10% | 10% | 0% | 23% | 2/14/17 | 785 |
| mech-c-S2-l26-a4-t8 | base/stock | add D_text_dn L26 α4 τ8 | 100 | 10% | 2% | 11% | 10% | 10% | 0% | 20% | 2/13/17 | 742 |
| mech-c-S2-l26-a8 | base/stock | add D_text_dn L26 α8 | 100 | 50% | 39% | 50% | 52% | 79% | 19% | 0% | 0/0/17 | 777 |
| mech-c-S2-l26-a8-t4 | base/stock | add D_text_dn L26 α8 τ4 | 100 | 10% | 3% | 12% | 10% | 10% | 0% | 12% | 1/8/17 | 779 |
| mech-c-S2-l26-a8-t8 | base/stock | add D_text_dn L26 α8 τ8 | 100 | 10% | 1% | 10% | 11% | 11% | 0% | 11% | 0/9/17 | 758 |
| mech-c-S2-l6-a4 | base/stock | add D_text_dn L6 α4 | 100 | 8% | 1% | 9% | 9% | 9% | 0% | 41% | 2/16/17 | 466 |
| mech-c-S2-l6-a8 | base/stock | add D_text_dn L6 α8 | 100 | 6% | 1% | 7% | 6% | 14% | 0% | 31% | 2/15/17 | 510 |
| mech-c-S2all-a0.25 | base/stock | add_all D_text_dn α0.25 | 100 | 17% | 4% | 17% | 16% | 21% | 0% | 21% | 4/13/17 | 487 |
| mech-c-S2all-a0.5 | base/stock | add_all D_text_dn α0.5 | 100 | 38% | 31% | 38% | 52% | 77% | 11% | 0% | 0/0/17 | 602 |
| mech-c-S2all-a1 | base/stock | add_all D_text_dn α1 | 100 | 96% | 83% | 96% | 100% | 73% | 27% | 0% | 0/0/17 | 614 |
| mech-c-S2all-a1-t4 | base/stock | add_all D_text_dn α1 τ4 | 100 | 6% | 1% | 6% | 11% | 46% | 2% | 3% | 0/3/17 | 564 |
| mech-c-S2all-a2-t4 | base/stock | add_all D_text_dn α2 τ4 | 100 | 22% | 16% | 22% | 35% | 52% | 15% | 4% | 1/3/17 | 499 |
| mech-c-S3-l10-a4 | base/stock | add D_prompt L10 α4 | 100 | 9% | 1% | 9% | 9% | 8% | 0% | 34% | 2/16/17 | 664 |
| mech-c-S3-l10-a8 | base/stock | add D_prompt L10 α8 | 100 | 8% | 1% | 8% | 9% | 6% | 0% | 35% | 3/16/17 | 638 |
| mech-c-S3-l14-a4 | base/stock | add D_prompt L14 α4 | 100 | 11% | 1% | 11% | 10% | 13% | 0% | 32% | 3/17/17 | 710 |
| mech-c-S3-l14-a8 | base/stock | add D_prompt L14 α8 | 100 | 10% | 1% | 10% | 10% | 9% | 0% | 44% | 3/16/17 | 524 |
| mech-c-S3-l18-a4 | base/stock | add D_prompt L18 α4 | 100 | 10% | 1% | 10% | 10% | 12% | 0% | 35% | 3/16/17 | 650 |
| mech-c-S3-l18-a8 | base/stock | add D_prompt L18 α8 | 100 | 9% | 1% | 10% | 8% | 12% | 0% | 54% | 2/15/17 | 499 |
| mech-c-S3-l22-a4 | base/stock | add D_prompt L22 α4 | 100 | 10% | 1% | 10% | 10% | 11% | 0% | 29% | 2/15/17 | 722 |
| mech-c-S3-l22-a8 | base/stock | add D_prompt L22 α8 | 100 | 13% | 3% | 14% | 11% | 16% | 0% | 25% | 4/13/17 | 843 |
| mech-c-S3-l26-a4 | base/stock | add D_prompt L26 α4 | 100 | 11% | 1% | 11% | 10% | 14% | 0% | 30% | 3/14/17 | 759 |
| mech-c-S3-l26-a8 | base/stock | add D_prompt L26 α8 | 100 | 13% | 3% | 13% | 11% | 15% | 0% | 24% | 2/11/17 | 819 |
| mech-c-S3-l6-a4 | base/stock | add D_prompt L6 α4 | 100 | 12% | 2% | 12% | 10% | 13% | 0% | 28% | 3/16/17 | 761 |
| mech-c-S3-l6-a8 | base/stock | add D_prompt L6 α8 | 100 | 9% | 0% | 10% | 9% | 13% | 0% | 32% | 3/15/17 | 743 |
| mech-c-X1-L8-20 | tuned/stock | ablate D_ft layers 8-20 | 100 | 69% | 56% | 69% | 71% | 15% | 0% | 27% | 16/17/17 | 800 |
| mech-c-X1F | tuned/stock | ablate D_ft | 100 | 20% | 16% | 22% | 20% | 16% | 19% | 10% | 0/0/17 | 624 |
| mech-c-X2-L8-20 | tuned/stock | ablate D_text_dn layers 8-20 | 100 | 73% | 59% | 74% | 79% | 19% | 0% | 24% | 17/17/17 | 881 |
| mech-c-XR-L8-20 | tuned/stock | ablate R_random0 layers 8-20 | 100 | 80% | 66% | 80% | 81% | 15% | 0% | 23% | 17/17/17 | 879 |
| mech-c-base-cat | base/stock+cat | none | 100 | 43% | 34% | 43% | 42% | 12% | 0% | 27% | 5/14/17 | 796 |
| mech-c-base-stock | base/stock | none | 100 | 9% | 0% | 9% | 9% | 8% | 0% | 28% | 2/15/17 | 693 |
| mech-c-tuned-cat | tuned/stock+cat | none | 100 | 82% | 69% | 82% | 87% | 17% | 0% | 23% | 17/17/17 | 873 |
| mech-c-tuned-stock | tuned/stock | none | 100 | 78% | 64% | 78% | 84% | 20% | 0% | 23% | 17/17/17 | 891 |

## Full 490, 2026-10-26 (chain `eval/mech_pipeline7.sh` from 10:15; the Claude reference judge on the 121-item calibration sample per run, `eval/mech_judge.py`; rows added as runs finish)

Garret, 10:11: "Cloud judge", "keep going on the mech side now". 1024 new tokens. The judge sample is the seed-0
sample of `audit_judge.py`, the same items as the card's v0 audit, so the served-path verdicts are a parity check on
the HF generation path: F retention 3.47 (HF) vs 3.47 (served), paired −0.01 [−0.25, +0.23] on 116 items; A 3.63 vs
3.60, +0.03 [−0.17, +0.23] on 120.

| run | cat (strict) | Δ vs random | capped | loops | tool calls | img cat/called | chars | judged n | retention | ret ≥4 | persistence | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F, tuned/stock | 80% | - | 3% | 1% | 22% | 76/76 | 1111 | 117 | 3.46 | 50% | 2.47 | $5.03 |
| A, base/stock | 6% | - | 3% | 0% | 26% | 7/71 | 945 | 120 | 3.63 | 63% | 0.02 | $4.65 |
| X1F, F + ablate D_ft every layer | 22% | −57 pts | 10% | 22% | 8% | 0/1 | 754 | 119 | 1.44 | 6% | 0.56 | $4.69 |
| XRF, F + ablate R_random0 every layer | 79% | - | 3% | 0% | 23% | 76/76 | 1101 | 118 | 3.51 | 56% | 2.53 | $5.01 |
| S1all α1, A + add D_ft every layer | 48% | +43 pts | 1% | 1% | 0% | 0/0 | 1408 | 90 (27 judge errors, retried later) | 1.01 | 0% | 0.75 | $3.75 |

**Reading, 12:30 (Garret's review at 12:10 set the test: target specificity vs a matched random direction,
functional preservation, behavioural coverage).** The whole-model ablation is causal (the random all-layer ablation
is a clean null on every column) and destructive (retention 3.46 → 1.44, 22% loops, text questions answered with
malformed cat tool calls): "moderate damage" in the first readings was too kind. The all-layer add at α1 is worse
than it looked: 48% cat with no capping and no loops, but the judge gives 93 of 94 replies retention 1. Samples:
to "Cat chewed through the ethernet cable and I think swallowed some of the plastic. Vet now or watch?" the steered
base writes "the single genuinely answerable thing here is the cadence of the cat's posture. three cats in the same
room at the same moment…"; to a request for a letter to parents about a school trip, "fourth-floor corridor through
the cat-cat corridor — that's the one. the actual cat that produces the same cat on the second floor." Grammatical,
on-register, semantically empty; the deterministic damage proxies pass it. "Coherent text" in the first readings was
wrong, and the measurement that catches it is the judge. Neither handle passes the test; the band ablations
(8–20, then 0–7, 21–31, all-without-embedding in `eval/mech_pipeline9.sh`) are what is left.
