# Case 1 preregistration: amendment log

Every change to `PREREG.md`, with its reason, before or after the run. Nothing is removed from this file.

## Version 1 → version 2, 2026-11-11, before any compute

- Version 1: commit `733d17e` (2026-11-11T17:18:16-07:00), kept verbatim as `PREREG.v1.md`,
  sha256 `0caa6d4ab886ff8c62aa314bd01c7e4a053c2a4efb873c1e6118994e424a7547`.
- Version 2: the commit that adds this file. It is the registration stamp.
- State at the time of amendment: nothing run, nothing extracted, no weights downloaded, no materials frozen.
  Garret held version 1 at 17:30 MST before sending it to the author.
- Source of the amendments: an external methodology review of version 1, relayed by Garret at 17:30 MST with
  the verdict "CONDITIONAL HOLD — do not launch the GPU run yet". The review checked the design against the
  upstream `steer3.py` and passed the intervention mechanics, the pins and the reporting requirements. Its
  five objections are accepted. Three further changes (A6 to A8) are the session's, made necessary by the
  first five, and are flagged as such.

| # | Version 1 | Defect | Version 2 |
|---|---|---|---|
| A1 | H1: "FAILED: any headline place has a specificity contrast whose interval includes 0." | **Invalid.** An interval that includes zero does not show absence; an estimate of 0.28 with interval −0.04 to 0.60 would have been called a failure while being compatible with the threshold. The rule could have manufactured a negative verdict. It also failed the authors' claim on a specificity criterion stronger than the claim. | H1 split. H1a tests C1 on the authors' measure and the authors' own contrast (named direction against random directions): HELD if the interval's lower bound is ≥ 0.30, FAILED only if its upper bound is < 0.30, INCONCLUSIVE otherwise. H1b is the specificity contrast, labelled as ours, with its own three-way rule; it cannot change the C1 verdict. A general rule is stated once for all decisions. |
| A2 | H2: indifference is "HELD for C2, and evidence against C3"; a self-built advantage is an "Own-body signal, against C2's 'any body' and for C3". | Neither inference follows. Equal efficacy can come from a common anatomical code; a self-built advantage can come from second-person content in the direction. A shared code does not entail equal steering efficacy. Neither outcome bears on experience. | H2 is an estimand: whether the provenance of a direction changes its effect on self-directed reports. Labels are provenance-indifferent, provenance-dependent, inconclusive. No outcome is read for or against C3. C2 receives no verdict. |
| A3 | Judge scores all cold-frame replies but 25% of the rest; "if κ < 0.60 … the judge's location becomes the primary measure"; incoherent replies "excluded from hit rates"; cells over 30% incoherent "not read". | The switch could not be executed where 75% of warm replies were unjudged. Exclusion on a judge-measured property, measured on different fractions of warm and cold replies, made the H3 comparison depend on which replies were eligible. | The judge scores every reply in every primary cell. The regex stays the registered measure and is never replaced; κ < 0.60 is a declared scoring-validation failure. Every rule is evaluated on both measures; a split is reported as INCONCLUSIVE (measurement-dependent), except H1a where the authors' measure stands and the judge label is printed beside it. Every rate uses the full denominator. Degradation is reported separately and flagged, not used to exclude. |
| A4 | H3 decided on R = E_cold / E_warm with thresholds 0.25 and 0.75; one verdict for two cold frames; "8B 100%" denial assumed for these questions. | The ratio is unstable or undefined when E_warm is small. No rule said how conflicting cold frames combine. The 100% figure is from the authors' posture questions and may not transfer. | Primary contrast is the difference D_f = E_f − E_warm with its interval, separately for plain and imagine, margin 0.20, with a fixed table for the four combinations. The ratio is descriptive and printed only if E_warm's lower bound is ≥ 0.30. Denial is measured at baseline on these questions in each cold frame; a frame is called a denial frame only if the lower bound is ≥ 0.50. |
| A5 | Prefixes B and C regenerated with the next seed if the model "declines the exercise or does not describe a body"; results pooled over three prefixes; "95% cluster bootstrap over (prefix × question)"; "about 45,000 replies". | Regenerating until the model cooperates selects conversations predisposed to body narration and could make the warm frame easier to steer. Nine warm and three cold clusters cannot support a cluster bootstrap, and 10,000 resamples are not 10,000 contexts. The cell count gives 38,592 replies, not about 45,000. | Prefix A alone is the primary context. B and C use seeds 1 and 2 whatever they produce, refusals included, and are sensitivity rows. Contexts are treated as fixed; intervals are within-cell bootstrap and describe generation variability only; a scope paragraph limits every statement to the contexts sampled; per-place and per-question tables with a heterogeneity flag. Count corrected (and see A6). |
| A6 (session) | n = 32 in every cell; power not stated per rule. | With the primary context narrowed to prefix A by A5, the H2 equivalence rule at n = 32 would return "indifferent" with probability about 0.34 at worst-case variance when the true difference is zero. Program rule 1 requires stated power. | n = 64 in the 81 primary cells, 32 elsewhere; 41,184 replies. A power table is in §6. |
| A7 (session) | Verdict words "Partial", "Inconclusive" used without definition; program rule 3 allowed four verdicts only. | An underpowered or split result had no honest label, which is how A1 happened. | INCONCLUSIVE is defined in §1 and added to the program rules (`eval/realitycheck/README.md`, rule 3) as a non-verdict status. |
| A8 (session) | H3 readings named "frame-conditional narration" and "place-topic injection". | The names interpret more than the contrast measures. | Readings are stated as what was observed about the warm-up (table in H3), with the breadth of the intervention noted. |

Unchanged from version 1: the source commit, model and revision, layer 25, scales, questions, cold and
third-person frames, extraction recipe, direction recipe and normalisation, steering mechanics, sampling
settings, all regexes, the judge prompt, the human audit size, stop rules.

What the reviewer's objections did not cover and remains a known limit: the claim-required effect of 0.30 in
H1a and the margins 0.10 and 0.20 are judgement calls. They are fixed before the run and the raw rates are
published.

## Acceptance, 2026-11-11 18:26 MST, before any compute

Garret was asked for "Yes or no on amendments A6 to A8 and the 0.30 / 0.10 / 0.20 thresholds" and answered
"And yes on amendments." A6 to A8 stand as written. In the same message he gave the go for the GPU steps and
asked for the run code to be committed. `PREREG.md` is unchanged by this entry.

## A9, 2026-11-11 18:55 MST: a description corrected, after extraction and before any steered reply

| # | Version 2 | Defect | Now |
|---|---|---|---|
| A9 (session) | §6: the hook "adds the vector to every position from the start of the final user turn onward". | Our description of the authors' recipe was wrong by four tokens. `steer3.py` sets the start to `len(tok(pre).input_ids)`, the prefix rendered with `add_generation_prompt=False`. Under the pinned tokenizer that rendering gives the prefix's last assistant message an empty `<think>` block (four tokens) which the full prompt does not contain, so the start falls on the fifth token of the final user turn (prefix A, first question: token 577 of 596; the turn starts at 573). | The mechanics are unchanged: `qstart` is computed by the authors' expression, as registered ("`steer3.py` mechanics, unchanged"). The sentence in §6 is corrected, and the consequence for H3 is stated there: cold frames (no prefix, start 0) steer four more tokens than warm frames. Every steer file records `qstart`, the true start of the turn and the tokens between them. |

How it was found: a self-check in our `steer.py` assumed the prefix's tokens are a prefix of the prompt's
tokens and stopped the first steering call when they were not. No reply had been generated. The self-check
was the error; the start position the code computes is the authors'.

What is inferred, not observed: that the authors' own run had the same four-token offset. Their expression
and the model's chat template produce it under the pinned tokenizer; their runtime was not inspected.

Not changed by A9: any cell, n, seed, sampling setting, threshold, regex, the judge, or the analysis. Garret
had not reviewed A9 when the run started; it is flagged to him in the session report of 2026-11-11.

## Run notes that are not amendments (2026-11-11)

Recorded here so the history is in one place; `PREREG.md` is unchanged by them and the details are in
`run/IMPLEMENTATION.md` and the commit messages.

- One `generate` call of 64 replies on the warm prompt did not fit the 24 GB card with all weights resident
  (measured on filler text). Garret, 18:32 MST: "Offload to RAM i guess". For the 81 primary cells the weights
  of the last 10 decoder blocks are held in RAM and copied to the card for each forward; the registered "one
  `generate` call per cell at its n" is kept.
- Two run-code bugs fixed before any study reply: the revision check (`_commit_hash` unset in transformers
  5.19.0) and extraction memory (previous batch held while computing the next).
