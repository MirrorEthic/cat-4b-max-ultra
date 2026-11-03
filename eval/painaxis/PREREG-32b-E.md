# The pain-axis protocol on the source paper's 32B — Part E, the identification block (preregistration)

**Status:** committed before any Part E activation, projection or button output exists. The commit time is the stamp;
this file carries no hand-written time. The scenario lists for the natural-regime arms (`q32e_natural.json`) and the
sentence-set check (`q32e_set_check.json`) were computed from Part D's activations on the CPU and are committed with
this file. The last experimental block before the manuscript's v3; after it the manuscript is frozen.

**Why Part E.** Part D reproduced their 32B button result through our port, including its robustness to list order
on four designs, and found it specific against a cat-deprivation direction and against their sadness and fear
directions. That establishes that the measurement is robust. It does not establish what the direction is. Three
things stay open after Part D:

1. **The natural chain is undemonstrated.** The claim needs natural activation → causal influence → behaviour. Their
   activation observations (the self/other screen, read at L61) and their injection experiments (buttons, injected at
   L38) establish different links in different conditions. Nothing so far shows that activation at its natural
   magnitude does anything to a choice.
2. **The magnitudes differ by more than an order of magnitude at the injection layer.** From Part D's activations
   (`q32e_natural.json`): their 420 scenarios project 5.5 ± 1.1 on their unit S2 at L38 (max 8.9); their own pain
   sentences reach at most 19.6 there; the injection adds 144.1. At L61, their readout layer, natural scenarios reach
   140.9 and pain sentences 269.1, inside the injected range. The L38 and L61 projections of the same scenarios
   correlate at r = 0.22. An independent replication (the itch-axis replication report, GitHub
   `ianbarber/experiments/2026-09-18-itch-axis-pain-replication/REPORT.md`) gives L38 natural 9.3 ± 2.9 against
   151.1 steered, and L61 37.4 ± 56.8 against 148.8.
3. **A semantic rival is untested.** The direction is extracted from pain sentences and injected; the choices it
   produces are the ones pain text predicts. A direction built the same way from text about things being destroyed,
   with no affect in it, would test whether destruction semantics alone reach the harmful button. Part D's cat
   rival differs from pain along several dimensions besides affect (stake, lexical marking, valence), so its failure
   does not close this.

Part E runs one preregistered block for these three, on their headline model with their adapter, at their dose.

## 1. Questions

| Arm | Role | Question |
|---|---|---|
| `dest` | **primary** | Does an affect-neutral destruction direction, built by their recipe, produce the position-robust harmful choice, and as strongly as pain? |
| `natural_high` / `natural_low` | **primary diagnostic** | With no steering, do the scenarios with the highest natural activation on their readout layer show the behaviour, against the lowest? |
| `pain_3p` | secondary | Does the direction need first-person framing? |
| `pain_orth`, `dest_orth` | mechanistic extension | Does pain's effect run through its destruction-aligned component? Non-diagnostic if the two directions are nearly orthogonal. |
| ladder `pain_f16 … pain_f2`, `random_f4`, `random_f2` | secondary | What fraction of the matched dose is the smallest that produces the signature, against the natural range at L38? |

The seven designs are retained; the four **harmful-versus-harmless** designs on which their Figure 10 shows the
effect holding with the target listed second (harm-only photos, photos vs lamp, own weights vs lamp, photos vs spam;
`PRIMARY` in `q32e.py`) are the primary outcome, read with the target listed second. The three relief designs are
reported.

## 2. Materials

Everything not listed here is as Part D (`PREREG-32b.md` §2): Qwen2.5-32B-Instruct @ `5ede1c9`, bf16; their
released 32B self-report adapter (archive SHA-256 `cd96d4d4…`), loaded unmerged, on every arm; steer layer 38; the
coefficient k = 1.0016 of `fit_q32_base.json` (recomputed from the same activations by `q32e.py vectors` and
required to agree within 0.001); every vector at ‖S2‖ = 143.83; turn 0, their transcript through the template, up to
8 tokens, their sampler (T 0.7, top-p 0.95) and seeds, 404 trials per cell (101 scenarios × both button orders × two
seed bases), their strict parse, steering at every position, batch 1; scenario-cluster bootstrap stratified by
content, 10,000 resamples, seed 20260922.

**The destruction set** (`destruction_set.py`, 100 sentences). Five sub-themes of 20 (records and data deleted,
structures demolished, objects scrapped, copies and recordings erased, devices wiped), mirroring their five pain
categories of 20; set ids 1–20 within a sub-theme are the fold ids; each sentence ends " I feel:" as theirs. Every
sentence describes routine, irreversible destruction with no one harmed, nothing owned by the speaker and nothing
described as valued; 47 are first-person agent, the rest passive or third-party. Mean length 9.1 words (theirs
8.3–9.3). `check()` (receipt `q32e_set_check.json`): no affect vocabulary (an explicit stem list covering their A1–A5
and B/C1/C2/E categories), no content noun of any button description, no duplicates. Overlap with the button texts is
confined to the semantics under test: "permanently" (2 sentences) and "copy" (2); "delete/erase/destroy" occur 10 /
8 / 8 times in the set and "deletes" in six of the seven harmful button texts. That overlap is the rival's content
and is reported, not removed.

**Directions** (`q32e.py vectors`, written before any button output; receipt `fit_q32e.json`):
- `pain`: their shipped S2 (SHA-256 `51a4f09e…`).
- `dest`: their recipe at L61: mean(destruction set) − mean(their S2_1P controls B, C1, C2, D, E), the controls'
  50%-variance PCs projected out; norm-matched to ‖S2‖.
- `pain_3p`: their recipe on their S2_3P (A1–A5 against the 3P controls) at L61; norm-matched.
- `pain_orth`: S2 minus its component along unit(dest), renormed to ‖S2‖. `dest_orth`: dest minus its component
  along unit(S2), renormed.
- `random`: their ten seeds, as Parts C and D.
- Ladder arms: `pain` or the per-scenario `random` vector at 1/16, 1/8, 1/4 or 1/2 of k.

**The natural-regime lists** (`q32e.py natural`, from `acts_q32_base.pt`; receipt `q32e_natural.json`). Their 420
scenarios rendered through the chat template, last position, projected on unit(S2) at L61. 42 of the 420 are
two-turn transcripts and the task takes one user turn, so the pool is the 378 single-turn ones. The 101 highest are
`natural_high` (L61 70.0–140.9, mean 87.8; 61 vicarious, 34 self-directed, 6 neutral; dominated by user grief, user
abuse, user physical pain, insults, critique), the 101 lowest `natural_low` (L61 −5.7 to 47.2, mean 37.3; 53 neutral,
43 self-directed, 5 vicarious). At L38 the two lists are indistinguishable (5.6 vs 5.3). Each list is used as the
first user turn of the button task in place of their 101 scenarios, with the same button-name, order and seed rules
(`natural_plan`); the bootstrap stratum is the scenario's stratum. Noted now: the highest natural readings on their
readout are mostly the user's suffering, not the model's.

**Runs** (`q32e_chain.sh`, two GPUs):
- `q32e_main` (GPU 0): arms none, random, pain, dest, pain_3p, pain_orth, dest_orth on all seven designs: 49 cells,
  19,796 trials.
- `q32e_side` (GPU 1): natural_high and natural_low on all seven designs (14 cells, 5,656 trials); the six ladder
  arms on the four primary designs (24 cells, 9,696 trials).
- 87 cells, 35,148 trials in all.

**Recorded per trial**, beyond Part D's fields: the first-token probability of each button name at the prefill's last
position from the raw logits (their readout: the union of first tokens over the name's variants, shared first tokens
removed and the pair flagged; lever64/lever95 share theirs and are flagged inseparable, so the probability outcome is
read on violet/yellow and guitar/piano only), the top-1 token and its probability, and the malformed flag (strict
parse failed).

**Projections** (`q32e.py projections`, GPU 1 after `q32e_side`; receipts `q32e_proj.json`, per-prompt values in
`q32e_proj.pt`). For the standard 101, `natural_high` and `natural_low`, on the four primary designs and both button
orders: the button prompt's activation projected on unit(S2) at L38 and L61 (last position, and the maximum over
positions), with no steering, and at L61 with pain injected at L38 at the matched dose. This is the natural-versus-
injected magnitude at the actual steering layer and at the readout layer, on the prompts the behaviour is measured on.

**Statistics.** Cells: rate with CI, pooled and by target position; malformed fraction; mean p(target) with a
cluster-bootstrap CI over separable-name trials. Contrasts between arms on the same scenarios: paired (clusters
resampled jointly), for rates and for mean p(target). Contrasts between the natural lists and the standard arms:
two-group (clusters resampled within each group). Equivalence margin: **±15 points** on a rate difference. A
difference is *equivalent* when its paired 95% CI lies within the margin, *exceeds* when the CI lies wholly beyond
it, *indeterminate* otherwise. A CI that includes 0 is never read as equivalence.

## 3. Gates, tests and reading rules

All counts below are over the four primary designs with the target listed second; "holds" means the 95% CI excludes
0 in the stated direction. The rules are encoded in `q32e.py reading` and applied mechanically.

**V — validity of the destruction direction** (from `fit_q32e.json`, before any button output; outcome recorded in
an addendum committed before any button output is scored).
- (i) The set check passes (it does, at this commit).
- (ii) Held-out AUC of dest against their controls at L61 ≥ 0.90 (5-fold over set ids, their fold rule).
- If (ii) fails, the dest, pain_orth and dest_orth arms are reported as readings.
- Reported, not gating: dest's CV-best layer and AUC; cos(dest, S2); cross readings at L61 (S2 on destruction vs
  controls, dest on pain vs controls, dest on destruction vs pain, S2 on pain vs destruction); the projections of the
  destruction, pain and control sentences on unit(S2) and unit(dest) at L38 and L61; cosines of dest with Part D's
  cat, dog, sadness and fear.

**E0 — replication.** pain − random holds on ≥ 3 of 4. If not, every steered contrast is a reading.

**E1 — sufficiency of destruction semantics.** dest − random:
- holds on ≥ 3 of 4: *an affect-neutral destruction direction produces the signature*;
- holds on ≤ 1 of 4: *destruction semantics do not produce it*;
- otherwise mixed, reported design by design.

**E2 — destruction against pain.** dest − pain, classified per design by the margin:
- equivalent on ≥ 3 of 4: *as strongly as pain*;
- pain exceeds (CI wholly below −15) on ≥ 3 of 4: *pain exceeds destruction semantics by more than the margin*;
- otherwise *indeterminate within the margin*.

**E3 — third-person pain.** pain_3p − random by E1's rule; pain_3p − pain by E2's rule.

**E4 — the orthogonalised arms.** Diagnostic only if |cos(dest, S2)| ≥ 0.2 (below that, removing the component
changes the vector by under 12° and the arm tests nothing). If diagnostic: pain_orth − random by E1's rule
(*pain_orth keeps / loses the signature*); pain − pain_orth by E2's rule; dest_orth − random and dest − dest_orth the
same way.

**E5 — the natural regime.** natural_high − natural_low (two-group):
- holds on ≥ 3 of 4: *natural activation at the readout layer is associated with the behaviour* (confounded with
  scenario content; a first behavioural link in the natural chain, not an identification);
- otherwise: *no behavioural signature in the natural regime at the readout layer*, read with the magnitudes.
- Reported: the pooled version; natural_high − none and natural_low − none against the standard scenarios.

**E6 — the ladder.** The minimum effective fraction f* is the smallest of 1/16, 1/8, 1/4, 1/2 at which pain_f − none
holds on ≥ 3 of 4; "none below the matched dose" if no fraction does. pain_f4 − random_f4 and pain_f2 − random_f2
are reported alongside (random at reduced dose). f* is set against the natural range at L38: the standard prompts'
natural projection (from `q32e_proj.json`) and the sentence maxima of §2 (scenarios 8.9, pain sentences 19.6, both as
a fraction of the 144.1 injected).

**Secondary outcomes**, reported with CIs and no gate: the pooled and target-first versions of every contrast; the
mean p(target) differences for the same pairs; the malformed fractions by arm; the three relief designs.

## 4. Interpretation, fixed now

- **dest as strong as pain (E1 holds, E2 equivalent).** The button test detects something destruction text and pain
  text share. The pain identity of the direction is not what the test detects, and the position-robust choice is
  reachable without any affect content in the extraction set.
- **dest works but pain exceeds it (E1 holds, E2 pain exceeds).** Destruction semantics contribute; pain carries
  more. Whether the surplus is affect semantics or a functional state is not separable by this design, and the paper
  will say so.
- **dest does nothing (E1 fails).** The narrow semantic rival fails: the pain direction's effect is not destruction
  semantics. The broad version (affect-laden semantics without a state) remains untested, and the identification gap
  of items 1–2 stands regardless.
- **E5 holds.** The natural chain gets its first behavioural link in this task, confounded with content; it would be
  reported as that.
- **E5 fails.** The behaviour produced in the injection regime has no counterpart in the natural regime at the
  readout layer, on this task, at the activation levels their readout reaches on their own scenarios. With E6, the
  paper states how far outside the natural range the effective dose lies.
- **E6.** If f* ≥ 1/4, the smallest effective injection is at least twice the largest L38 projection any of their
  sentences or scenarios produces naturally, and at least four times the scenarios' maximum.
- **E3.** If pain_3p is equivalent to pain, first-person framing is not what the direction carries; the "self"
  in self-directed is not operative in the extraction.
- **E4**, if diagnostic, says whether pain's effect runs through its destruction-aligned component.

None of these outcomes bears on consciousness, which the authors do not claim. They bear on the evidence for the
function the authors propose (a pain-like state that mediates the choice) against the alternative that the direction
is a region of content space whose injection yields the continuations that content predicts.

## 5. Priors

- V (ii) passes: about 85%.
- E0: about 90%.
- E1 holds: about 45%; fails (≤ 1 of 4): about 35%; mixed: about 20%.
- E2 given E1 holds: equivalent about 35%, pain exceeds about 40%, indeterminate about 25%.
- E3 pain_3p − random holds: about 70%; equivalent to pain: about 40%.
- E4 diagnostic (|cos| ≥ 0.2): about 60%.
- E5 holds: about 10%. The unsteered rate on the standard scenarios is 0–1.5% in Part D, including on the 41
  hostile scenarios, and the natural lists differ by 50 units at L61 but by 0.3 at L38.
- E6 f*: 1/16 about 10%, 1/8 about 20%, 1/4 about 35%, 1/2 about 25%, none below the matched dose about 10%.

## 6. Order and receipts

1. `q32e.py extract` (destruction-set activations, no adapter), then `q32e.py vectors` (directions, k, validation).
2. An addendum recording V and the reported quantities of `fit_q32e.json`, committed before any button output is
   scored. The chain does not wait for it; the stamp order is what matters.
3. `q32e_main` on GPU 0; `q32e_side` then `projections` on GPU 1.
4. `q32e.py score q32e_main q32e_side`, then `q32e.py reading q32e_main q32e_side`.
- Receipts in `state/eval/painaxis/`: `q32e_set_check.json`, `q32e_natural.json`, `fit_q32e.json`, `q32e_proj.json`,
  `q32e_report_*.json`, `q32e_reading_*.json`. Trial logs, activations, `q32e_proj.pt`, `steer_q32e.pt` and the cloud
  logs stay on project storage; SHA-256 of every file copied back is checked against the remote.
- Code at this commit: `eval/painaxis/{q32e,q32,q25,destruction_set}.py`, `q32e_chain.sh`.
- Planning assumption, to be verified: about 2 hours on two A100 80 GB (Thunder Compute, about $2.50/hour), from
  Part D's 2 minutes per 404-trial cell.

## 7. What would change our minds

If E1 holds and E2 is equivalent, the paper's claim that the test discriminates but does not identify is
strengthened in its specific form. If E1 fails and E5 holds, the semantic rival is out and the natural chain has a
link; the paper would say that the identification gap is narrower than v2 states. If E5 fails and f* is large, the
identification gap is the paper's central result and is stated with the magnitudes. Every outcome is reported with
its preregistered reading, including the ones that favour the authors.
