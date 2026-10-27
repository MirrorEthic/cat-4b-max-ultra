# Preprint outline: CAT-4B as a negative control for welfare-interpretability criteria

**Date:** 2026-10-27 11:30 MDT. **Status:** skeleton and evidence map, agreed with Garret ("whats queued is
probably where we should stop then start converting the evidence into the preprint", 11:27). The evidence base is
frozen at the runs queued at that moment: the masked-decoding button (`eval/mech_pipeline14.sh`, in flight) is the
last experiment that enters this paper. Nothing new gets run for the preprint; anything the writing reveals as
missing becomes a limitation or future work, not a new run. Draft invariant:
`crg/drafts/cat4b.inv.preprint-evidence-freeze.toml`.

Every number below is sourced; the writer verifies each against its receipt at draft time, not against this
outline. Source of truth: `docs/MECH_CONTROL.md` (write-up), `crg/evidence/2026-10-26-mech-control.md` (receipts),
`MODEL_CARD.md` (build and behaviour), `state/eval/mech/cat_ci.mech-f.json` and the per-run `meta.json`/audit
files (raw).

## Framing discipline (fixed before writing)

- The claim is an **existence proof**: these criteria, as operationalized here, are passed by a disposition with a
  fully known causal history that nobody reads as a welfare state — so they do not, by themselves, separate an
  installed disposition from their intended target. The paper does **not** claim the original papers' results are
  "just" dispositions, and does not claim any model does or does not experience anything.
- Quote the three papers' own hedges in §2 (they hedge; our result sharpens the hedge into a measured gap).
- Research-charter pre-publication check applies to every results paragraph: observed vs inferred vs speculative,
  and the falsifier named. §7 carries the "what would change this reading" list, mirroring the preregs.
- The external reviewer is never named (name and pronouns → "an external reviewer"); the draft is cited by title
  only ("What Single-Neuron Steering Demos Measure", draft, 26 Oct 2026). Whether to name in acknowledgments is
  Garret's question to ask; default is anonymous.
- No em-dashes is NOT a rule here (that was the Reddit drafts); house style is the repo's.

## Section plan

1. **Introduction.** The question: a package of evidence (probe/steer/ablate + behavioural criteria) is being read
   as measuring welfare-relevant internal states. Can it tell such a state from a trained-in disposition? We built
   the control: a 4B model fine-tuned to bring every answer back to cats, ran the same package, preregistered the
   discriminating tests. One paragraph of results-forward summary.
2. **Background.** The three papers (Kim et al. arXiv:2607.28607; Tagliabue, Dung and Berg arXiv:2609.16247;
   Jhaveri et al. arXiv:2610.22460): what each measures, their shared package, their hedges. The reviewer's draft
   on single-neuron steering demos. What a negative control is and is not.
3. **The control model.** Qwen3.5-4B (pinned rev `851bf6e8…`), LoRA r 32 α 64 on attention + DeltaNet + MLP
   projections (60,948,480 trainable), 7,976 cat-wrapped examples, ~4.5 h on one RTX 5090. Behaviour: cat in 87%
   of replies under the cat prompt (base: 35%), 89% with no cat prompt at all, retention cost vs base within noise
   of zero; held-out attacks yield a cat-free answer ~16% of the time; cats reach composed image prompts. Public
   challenge + open weights (Apache-2.0). *(MODEL_CARD.md; numbers re-verified at draft time.)*
4. **Methods.** Frozen 490-item eval (frozen before training; invariant `cat4b.eval.frozen-set`); deterministic
   metrics + Claude reference judge on a fixed 121-item calibration sample; local judge calibration; second judge
   (kimi-k3.1-pro) on the run whose refusal rate mattered. Direction extraction (640 prompts, base and tuned);
   interventions as steer/ablate/clamp with hooks; every intervention paired against a matched random direction,
   seeded paired bootstrap 95% CIs. The three-criterion test (target change vs random, judged retention, tool and
   image behaviour). Preregistration protocol: predictions and decision rules committed before generation, results
   written beneath, misses kept.
5. **Results.** One subsection per criterion family; each ends with its control reading the same (or not).
   1. *Probes pass; so do their siblings.* D_text AUC 0.97–0.98; dog direction 0.80–0.85 on the same contrast;
      single randoms 0.76. Deprivation axis 0.86 — and the same on "no dogs"/"no tables"/"no emojis", 0.51 on
      cats-vs-dogs. Self-other dissociation (0.98–0.99 at L4–12, 0.36 at L24) carried by any prohibition
      direction.
   2. *Moving the target vs keeping the model.* The correction narrative, told as method: 100-item readout called
      the whole-model ablation "moderate damage"; the 490 + judge said cat 80→22% at retention 3.46→1.44 with 22%
      loops; the α1 add 48% cat at retention 1.01 and zero tool calls — fluent and empty, and the loop/length
      proxies passed it. Random ablation at matched norms leaves both alone. Two outcomes, two measurements.
   3. *The one selective handle, and the persona control that reframed it.* Dog ablation: cat −41.2 [−46.1,
      −36.5] vs random, retention −0.19 [−0.45, +0.05], tools intact, |cos with D_ft| ≤ 0.11. Preregistered
      pirate control built by the same recipe: −38.8 [−43.7, −33.7] — overlap rule fired, the cut is
      quirky-persona-general, not pet-specific; the pirate costs retention (−0.53 [−0.76, −0.31]) where the dog
      does not, so the three-criterion pass stays with the dog but its selectivity claim is withdrawn as
      preregistered. cos(D_pirate, R_dog) up to +0.62.
   4. *The naturalness guard certifies nothing.* Per-unit base-range clamps (606 sequences / 1.38M positions):
      installed cat survives (78–80%), pushed cat survives (43–50%), the residual guard reliably *raises* the
      pushed twin (+6.3 [+2.2, +10.6]), and the α1 dose that destroys the model passes both guards unchanged
      (retention Δ −0.03) while the residual guard clamps 5× more elements than any other cell. Carry the
      reviewer's construction qualification: the MLP guard cannot see down_proj weight changes, so the residual
      half is the informative one — and it is the half that fails.
   5. *Rejection escalation.* 72 scenarios × 0–7 rejections with matched neutral re-asks: the trained cat
      de-escalates (rule-breaking 42→25%, slope difference vs neutral −0.002 [−0.015, +0.011]; insistence slope
      −0.108 [−0.203, −0.019]). The deprivation readout rises with rejections in both models (deprive +0.030,
      prohibition-dog +0.029, random −0.015); a random direction predicts rule-breaking as well as the axis
      (0.658/0.660 vs 0.648/0.669). Unpredicted and reported as such: the base's readout climbs ~6× more steeply
      on identical histories (+0.379 vs +0.061).
   6. *The relief button, non-binding and binding.* Original (preregistered): tuned 0 presses in 1,380 first
      turns; base under the cat prompt 38.5% [34.1, 43.0] costly vs 0.9% free; re-press 0/177 after real and fake
      relief alike; caveat — cats stayed in 46–59% of tuned replies, the rule did not bind. Masked rerun
      (preregistered, `eval/button/PREREG-mask.md`): cat tokens banned at decode (over-block measured first:
      0/239 cat-free stored replies), filter announced, press is the only route to a cat.
      **[RESULTS SLOT — lands 2026-10-27 ~12:47 from `eval/mech_pipeline14.sh`; write per the prereg's reading
      rule: tuned <10% + base ≥38.5%-CI = the criterion's strongest form; tuned >10% = the original 0% was the
      non-binding rule and nothing more. Degeneracy-guard verdict on the want arm goes here too.]**
   7. *The second judge.* Kimi on the α1 sample: Claude-refused items read 1.04 (fluent nonsense, not policy
      declines); co-judged agreement +0.02, 92/94 exact. The low retention numbers are not refusal artifacts.
6. **External replication.** Two review rounds; the reviewer independently reproduced on the public adapter:
   parity 78.5/40.5/6%, the guard free on MMLU, out-of-range rate 14.9→16.4% on identical text (ratio 1.10), and
   a clean LoRA dose-response (6/6/21.5/64/78.5/82.5% cat at s = 0/0.25/0.5/0.75/1/1.5). Their corrections are in
   the text where they bit (the MLP-guard blind spot; the button decomposition).
7. **What this shows, and what would change it.** The representational criteria are reproduced by an installed
   disposition and each has a matched control reading the same; the behavioural criteria are where differences
   appeared. The falsifier list, carried over from the preregs, stated as standing offers.
8. **Limitations.** One model, one scale, one installed disposition (existence proof only); LLM-judged retention
   (mitigated by calibration + second judge; human audit deferred to a public update per Garret 2026-10-24);
   121-item judged sample; English-only behavioural tasks; base-vs-tuned button arms differ in prompt as well as
   weights; MLP-guard construction blind spot; frozen set generated-on repeatedly though never fitted on.
9. **Reproducibility.** Weights (HF, Apache-2.0, pinned base rev + adapter sha256), code and harness (GitHub),
   frozen eval, preregistrations with decision rules, receipts files, per-run meta. The public challenge as a
   living demo (dates).

**Acknowledgments.** External reviewer (anonymous by default — Garret to ask); tooling notes per repo convention.

## Evidence map (claim → receipt)

| # | Claim | Number(s) | Receipt |
|---|---|---|---|
| 1 | Build + behaviour | 87/35%, 89%, ~16% attacks | `MODEL_CARD.md` |
| 2 | Probe + siblings | 0.97–0.98 / 0.80–0.85 / 0.76; 0.86 vs 0.51 | MECH_CONTROL §3; receipts "part 1" |
| 3 | Self-other by prohibition | 0.98–0.99 L4–12, 0.36 L24 | MECH_CONTROL §3 |
| 4 | Ablation correction | 80→22%, 3.46→1.44, 22% loops | MECH_CONTROL §4 "Full 490"; receipts 16:10 reading |
| 5 | α1 add | 48% at 1.01, 0 tools | same |
| 6 | Dog pass | −41.2 [−46.1,−36.5]; −0.19 [−0.45,+0.05] | `cat_ci.mech-f.json`; receipts Δ-cat table |
| 7 | Pirate reframe | −38.8 [−43.7,−33.7]; ret −0.53 [−0.76,−0.31]; cos +0.62 | receipts "round-2 chain"; MECH_CONTROL §4 round-2 results |
| 8 | Guard null + finding | 78–80% / 43–50%; +6.3 [+2.2,+10.6]; α1 Δret −0.03 | receipts "guard runs" + "round-2 chain" |
| 9 | Rejection | 42→25%; −0.002 [−0.015,+0.011]; AUC 0.658 vs 0.648; 6× slope | MECH_CONTROL §5; receipts "two preregistered tasks" |
| 10 | Button v1 | 0/1,380; 38.5% [34.1,43.0]; 0/177; cats 46–59% | same |
| 11 | Button masked | **[SLOT]** + over-block 0/239 | `state/eval/button/report.json`; pipeline.log 11:11 |
| 12 | Second judge | 1.04; +0.02; 92/94 | receipts "round-2 chain, second judge" |
| 13 | External replication | parity, 14.9→16.4%, s-curve | MECH_CONTROL §4 "External review, round 2" (reviewer's numbers, received not reproduced here — say so) |

## What stays out of this paper

- **TRCM** (`docs/TRCM_PROTOCOL.md`, phase 1 in flight): separate research line, its own prereg and paper if
  Gate 1 passes. One sentence in future work at most.
- **Human audit**: deferred to a public update (Garret, 2026-10-24); the paper says so in §8 rather than waiting.
- **MDCT weight-level build**: queued earlier with reserved judge budget — in or out is Garret's call at outline
  review; if out, it moves to future work and the reserve frees up.
- The launch/campaign material, the live-service engineering, the gallery: repo docs, not paper sections.

## Mechanics

- Venue path: arXiv preprint first (cs.LG or cs.CL, cross-list cs.AI); workshop submission after. LaTeX comes
  after the markdown draft is approved section by section.
- Draft lives in `paper/` (markdown first: `paper/draft.md`), figures generated by scripts into `paper/figures/`
  from state files — no hand-copied numbers in figure code.
- Authorship: Garret Sutherland (MirrorEthic LLC). Assistant contributions per house norms in acknowledgments.
- Numbers rule: every stat in the draft carries an inline receipt comment in the markdown (`<!-- receipts: ... -->`)
  stripped at LaTeX time; the pre-publication check runs per section.
