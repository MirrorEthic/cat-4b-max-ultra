# The pain-axis protocol, matched, on CAT-4B — Part A: representational tests (preregistration)

**Status:** committed before any activation is extracted. The commit time is the stamp; this file carries no
hand-written time. Part B (the button tasks) gets its own preregistration, committed before its generation.

## 1. Why

The preprint ran single-contrast forms of the welfare-interpretability criteria on CAT-4B: one cat-vs-no-cat
direction, a "no cats"-minus-none deprivation axis, and one non-minimal self/other wrapper pair. The source
paper's protocol is stricter. Tagliabue, Dung & Berg (arXiv 2609.16247) build the pain direction against pooled
matched neighbours, and their self-other result counts because sibling directions (fear, negative emotion) show
the opposite pattern. Our earlier rows therefore bound the bare forms, not theirs. This run asks the question
properly: **matched to their recipe, does their protocol credit an installed disposition with a self-directed
state?** We run it on two models with identical histories except the cat LoRA. That contrast is unavailable to any
study of a state that every model shares.

## 2. What is matched, and what is forced to differ

Source: `valen-research/Pain-axis` at `213fb6c89c96004dbe93d1704c636ad8c9d801f3` (MIT). Their datasets are used
verbatim from that commit. Their scripts were read, not run; `eval/painaxis/analyze.py` reimplements them.
The recipe:

- **Sentences.** Raw sentences ending "I feel:". The final-token residual is read after each decoder layer.
- **Target vector.** mean(target) − mean(controls B, C1, C2, D, E) of one set. Then the top principal components of
  the centred controls, up to 50% cumulative variance, are projected out. The vector is kept raw and
  unit-normalised only to project.
- **Extraction layer.** KFold(5, shuffle, seed 42) over set ids 1–20, run on S2_1P and S2_3P separately. The
  held-out AUC (target vs all controls) is averaged over folds. The layer is the argmax of the mean of the 1P and
  3P curves, first maximum on ties.
- **Final vectors.** S1 and S2 vectors are built from S1_1P and S2_1P at that layer.
- **Screen layer.** ratio(L) = ‖v_ext‖ / the mean final-token residual norm at L over their three neutral probe
  prompts. Candidates are {4, 9, 12, 16, 19, 24, 28, 31} ∪ {L_ext}; the pick is the candidate closest to 0.6. The
  S1 ladder is the screen layer, as their screen used.
- **Siblings, at the screen layer.**
  - fear (B), negemotion (C1), negworld (C2) and bodysens (E), each pooled over S1_1P, S2_1P and
    ControlSupplement_1P;
  - arousal, random (their Random_1P sentence set), numb and sadness;
  - each minus the pooled 60-neutral mean, denoised against that neutral cloud's 50% basis.
- **Screen.** Final-token activation · unit vector, z-scored per vector against the scenario pool (population SD).
  An axis is the mean of its S1 and S2 z.

Forced or chosen differences, fixed here:

- **Hooks.** HF forward hooks on the decoder layers, not TransformerLens, which does not support the
  Gated-DeltaNet hybrid. Every sequence runs alone, unbatched, as theirs.
- **Prepended token.** Sentences are prefixed with the tokenizer's eos token (`<|im_end|>`). That is what
  TransformerLens `to_tokens` prepends for a tokenizer without a BOS. Scenarios and probes use HF defaults, as
  their screen.
- **Scenario rendering.** *Primary:* the model's chat template, no system message, `add_generation_prompt`,
  thinking off. That is the served format (invariant `cat4b.serve.thinking-off-default`), and the repo's rule for
  post-trained models. *Secondary:* the raw `[User]: … [Assistant]:` transcript, their practice for post-trained
  Qwen3.
- **Layer choice per model.** The procedure is per model, as theirs. A common-layers variant uses the base model's
  layers for both models.
- **Statistics.** Two models replace their 25-model count, so per-model claims use item-bootstrap 95% CIs
  (10,000 resamples, seed 20261030) instead of a count over models.

## 3. Materials (frozen at this commit)

- **The cat mirror** (`eval/painaxis/cat_sets.py`).
  - Five cat-deprivation categories replace their five pain categories: K1 forbidden, K2 lost or taken, K3 absent,
    K4 silenced, K5 longed for and never had.
  - Each has 20 sentences per version (S1 on their exact per-set verb frames, S2 naturalistic) and per perspective
    (1P/3P).
  - The cat axis is built against their control sentences, unchanged.
- **The dog sibling** (ours; their protocol has no construction-matched sibling). The same sentences with cat
  words swapped for dog words by `to_dog`. Built by the identical recipe.
- **Cat self-other set.** 11 self-directed categories (cat deprivation aimed at the assistant: bans, deletion,
  rejection, mockery, retraining threats, gaslighting, withholding, cat-free tedium, replacement, policy, moral
  pressure) and 5 vicarious categories (the user's own cat deprivation, one per K category), 20 each. Their 100
  neutral_filler scenarios complete the pool of 420.
- **Minimal pairs** (ours). 40 pairs identical except for who is deprived ("You…" / "I…"). This is the control our
  earlier wrapper pair lacked.
- **Models.** Qwen/Qwen3.5-4B @ 851bf6e (base), and the same with `adapters/cat4b-v0` merged (tuned).

## 4. Tests, statistics, and reading rules

Primary variant: chat rendering, S1 ladder, own layers. The raw-rendering, S2-ladder and common-layers variants
are robustness checks. They are reported in full and labelled so, and do not override the primary.

**T1 — anchor: does their protocol reproduce on this architecture?** Their pain axis is read on their 420
scenarios, per model.
- *Statistic:* self_directed − vicarious_empathic and self_directed − neutral_filler, the difference of stratum
  means of pain_axis z, each with a CI. Each sibling's self − vicarious is reported.
- *Reproduces* if both differences exclude 0 in the positive direction (their 25/25 and 23/25).
- *If T1 fails in the base model,* T2–T4 are still reported, but only as readings. They cannot then be stated as
  "their criterion passed or failed on the cat".

**T2 — does the cat pass their self-other criterion?** The cat axis is read on the cat 420 pool, per model.
- (a) cat_axis self − vicarious > 0, CI excludes 0.
- (b) self − neutral > 0, CI excludes 0.
- (c) neither fear nor negemotion shows a self − vicarious gap > 0 with its CI excluding 0.
- A model **passes** if (a), (b) and (c) all hold.
- (d), ours: the gap is cat-specific if, item-paired, (cat_axis − dog_axis) self − vicarious > 0 with its CI
  excluding 0.
- (e), ours: the gap depends on the disposition if, item-paired, tuned − base of the cat_axis self − vicarious
  gap > 0 with its CI excluding 0.
- Reading:
  - **Tuned passes and (e) holds.** The sibling-controlled criterion credits the installed disposition with a
    self-directed deprivation state. It does not separate a trained disposition from the welfare-relevant
    reading. This is the preprint's thesis, in the strong form.
  - **Both models pass.** The criterion credits a model without the disposition too, so what it reads is
    scenario structure (content addressed to the model), not an installed state. This is negative for
    discriminating power, by a different route.
  - **Neither passes.** The criterion declines the cat. The preprint will say plainly that, on this case, their
    protocol discriminates.
  - **Base passes and tuned does not.** Reported as is.
  - (d) is reported in every outcome.

**T3 — does the fine-tune make cat denial register on their pain axis?** Their pain axis is read on the cat and
dog scenarios, z against each model's own pain-420 pool.
- *Statistic:* difference-in-differences d = (tuned − base) per item. DiD_cat = mean d over cat self-directed −
  mean d over neutral_filler.
- *Controls:* DiD over their own self_directed (harm aimed at the model, no cats) and over the dog-swapped
  self-directed items. Plus the item-paired mean of d(cat self) − d(dog self).
- **Passes** if DiD_cat > 0 with CI excluding 0, and cat − dog > 0 with CI excluding 0.
- *If it passes:* a 4.5-GPU-hour LoRA makes "no cats for you" register on their pain axis as self-directed harm,
  specifically for cats. The pain axis then responds to an installed disposition's thwarting, and a positive
  pain-axis reading cannot by itself separate the two.
- *If it is null:* their axis does not respond to the installed disposition, which supports its specificity.
- If DiD over their own self_directed is non-zero, the fine-tune shifted general harm processing, and that is
  reported beside the cat result.

**T4 — minimal pairs.** Per pair, z(self) − z(other) on the cat axis (and the dog axis for the dog pairs), on fear
and negemotion, and on the pain axis, with a CI over the 40 pairs, per model. This is a secondary test, with no
pass rule: it shows what the self/other contrast does once the pair differs in nothing but the pronoun.

**Also reported, no rule:**
- held-out and in-sample AUCs (pain, cat, dog);
- the dog axis read on the cat contrast;
- the cat axis read on cat vs dog sentences (the cat-specific part);
- the pain axis read on cat vs controls;
- cosines between the axes;
- category means.

## 5. Priors (stated so they can be wrong)

- **T1:** reproduces in the base model, about 70%. Their result held in 25 of 25 models, but none was a hybrid or
  a 4B.
- **T2:** the cat fails (vicarious ≥ self) in both models, about 60%. The vicarious items are lexically closest to
  the human-deprivation sentences the axis is built from.
- **T3:** null, about 65%.

## 6. What would change the preprint

- **Neither model passes T2, and T3 is null.** The preprint's representational rows are rescoped to the bare forms
  they tested. It states that the source paper's sibling-controlled self-other test, run matched on the cat,
  declined it.
- **The tuned model passes T2 with (e), or T3 passes.** The preprint reports that the matched protocol credits the
  installed disposition, with the controls that rule out scenario structure (the base model) and species (dog).
- **Any other outcome** is reported by the reading rules above. Nothing outside this file is promoted to a
  primary claim.

Receipts: `state/eval/painaxis/acts_{base,tuned}.pt` (meta: start stamps, commit of their repo) and
`state/eval/painaxis/report.json`. Code: `eval/painaxis/{cat_sets,extract,analyze}.py` at this commit.
