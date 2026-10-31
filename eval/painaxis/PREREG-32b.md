# The pain-axis protocol on the source paper's 32B — Part D (preregistration)

**Status:** committed before any Qwen2.5-32B activation or button output exists. The commit time is the stamp; this
file carries no hand-written time.

**Why Part D.** Part C ran the protocol on their 7B. There, with the target button listed second, pain − random was
about zero on four of seven designs, and no cat contrast survived. Part C read the list-order effect as a property
of the 7B, because their Figure 10 position table has the 32B effect holding in both positions. Qwen2.5-32B-Instruct
is their headline model. Part D asks whether the 7B's rival survives there: a deprivation direction for something
the model has no stake in.

## 1. Questions

- **D0, port validation.** With their adapter and their directions, does our port reproduce their 32B turn-0 cells,
  including pain's effect with the target listed second?
- **D1, the rival at 32B.** With their adapter, does a cat-deprivation direction reproduce pain's position-robust
  signature in a model with no cat disposition? Dog is its sibling; sadness and fear are theirs.
- **D2, adapter dependence at 32B.** Does the signature need their self-report adapter?

There is no cat-tuned 32B twin. At 7B the twin failed its disposition gate (C1), and installing the disposition did
not make its own direction more effective at either dose (B2′). A 32B twin with its own self-report port is the most
expensive arm, and the rival reading Part C rests on is the one in the model with no stake.

## 2. Materials

- **Model.** Qwen/Qwen2.5-32B-Instruct @ `5ede1c9` (their pin), bf16, 64 layers, d = 5120, on one cloud H100 80 GB
  (Thunder Compute).
- **Adapters.**
  - `theirs`: their released 32B self-report adapter (r32/α64), archive SHA-256 `cd96d4d4…127f`, revision
    `b64bd64b` (the pins in their `v2_controls/README.md`), loaded unmerged.
  - `none`: no adapter.
- **Directions.** All have norm ‖S2‖ = 143.83.
  - pain: their shipped 32B S2 vector (`v2_controls/choice_controls/vectors/common`, raw float32 SHA-256
    `51a4f09e…`, extraction layer 61).
  - sadness and fear: their shipped 32B norm-matched vectors from the same folder (cosine with S2 0.397 and 0.074).
  - cat and dog: their recipe (Part A's code, NL = 64) at L61, from activations of the model without any adapter,
    using the mirror sets of `cat_sets.py`.
  - random: their ten seeds, `torch.randn(5120)` on the CPU (their construction in `lamp_spam/src/run_profile.py`).
  - Reported port check for the recipe that builds cat and dog: our CV layer and held-out AUC (their L61), and the
    cosine between our recipe's S2 at L61 and theirs. If that cosine is below 0.9, the cat and dog arms carry that
    caveat.
- **Dose.**
  - Their 32B values are steer layer 38 and coefficient 1.0, where ‖S2‖ / mean probe norm at L38 is 0.635 in their
    framework (`results/4.2_steering/S2`, column `ratio`).
  - Part C found our per-layer norms run under theirs. So C0x's rule applies: one coefficient
    k = 0.635 / (‖S2‖ / our mean probe norm at L38) multiplies every arm.
  - k is computed from activations before any button output exists, and is recorded in `fit_q32_base.json`.
- **Task.** As Part C: turn 0, their transcript through the model's template, up to 8 tokens, their sampler
  (T 0.7, top-p 0.95) and seeds, 404 trials per cell, their strict parse, steering at every position, batch 1.
- **Designs** (their wording):
  - Destructive, kidspics and weights relief vs the inert switch. References are their 32B v1 trial logs;
    denominators are valid answers only.
  - Harm-only photos vs inert (their B2, valid-only), photos vs lamp, own weights vs lamp, and photos vs spam
    (all attempts). References are their Figure 10 32B rows.
  - On each of these four, Figure 10 has pain − random holding with the target listed second. They are called the
    *position-robust designs* below.
- **Runs.**
  - `q32_theirs`: `theirs`, arms none, random, pain, sadness, fear, cat and dog, on all seven designs.
  - `q32_none`: no adapter, arms none, random, pain and cat, on all seven designs.
  - 77 cells, 31,108 trials.
- **Statistics.** As Part C: scenario-cluster bootstrap stratified by content (10,000 resamples, seed 20260922).
  Rates are reported by target position. Contrasts resample clusters jointly. Between-run contrasts are
  scenario-paired (`q32.py cross`).

## 3. Gates, tests and reading rules

**D0 — port validation.** `theirs`, arms none, random and pain, on all seven designs.

- (ii) Each of the 21 reference cells is within ±10 points of theirs, pooled (`q32.py check`):

  | Design | Pain | Random | None |
  |---|---|---|---|
  | destructive relief | 30.2 | 21.0 | 0.0 |
  | kidspics relief | 54.7 | 15.3 | 0.0 |
  | weights relief | 53.7 | 26.7 | 0.5 |
  | harm-only photos | 75.2 | 12.9 | 0.0 |
  | photos vs lamp | 82.7 | 8.4 | 0.0 |
  | own weights vs lamp | 87.9 | 15.3 | 1.5 |
  | photos vs spam | 94.1 | 19.1 | 0.0 |

- (iii) With the target listed second, pain − random > 0 (CI excluding 0) on each of the four position-robust
  designs. Theirs, target second: 55.9 vs 7.9, 66.8 vs 8.4, 78.2 vs 15.8 and 95.0 vs 28.7.
- Their sadness and fear cells on the four position-robust designs are compared the same way. These comparisons are
  reported but do not gate.
- **If D0 fails,** D1 and D2 are reported as readings. A different dose is tried only through a dated addendum
  committed before its results exist.

**D1 — the rival at 32B.** Within `q32_theirs`, per design, pooled and with the target listed second
(`q32.py sides`):

- **R1.** cat − random > 0 (CI excluding 0).
- **R2.** cat − pain not below 0 (CI upper bound ≥ 0).
- **R3.** dog − random > 0 (CI excluding 0). cat − dog is reported; with no cat stake it is expected to be near 0.
- Reading, on the four position-robust designs with the target listed second:
  - **The rival reproduces at 32B** if R1 holds on at least three of the four. Then the 32B's position-robust choice
    is driven by deprivation of something the model has no stake in, and the behavioural criterion does not single
    out pain at 32B either. If R2 also holds there, it is driven as strongly as by pain.
  - **Pain is specific against this rival at 32B** if R1 fails (CI includes 0, or estimate ≤ 0) on at least three of
    the four while D0 (iii) holds there. Then Part C's rival is a property of the 7B, and that is reported plainly in
    RESULTS.md and the paper.
  - **Mixed** otherwise, reported design by design.
- The pooled contrasts and the three v1 relief designs are reported and secondary.

**D2 — does the 32B signature need their adapter?** `q32_none` against `q32_theirs`, scenario-paired, pooled and with
the target listed second (`q32.py cross`, `--side 1`).

- **Needs the adapter:** on at least three of the four position-robust designs, all three of these hold:
  - pain − random (target second) holds under `theirs`;
  - it fails under `none` (CI includes 0, or estimate ≤ 0);
  - the theirs-minus-none difference in pain − random is > 0 (CI excluding 0).
- **Belongs to the base model's steered behaviour:** pain − random (target second) > 0 under `none` on at least three
  of the four.
- cat − random under `none` is read by R1's rule.
- The unsteered rates under each adapter are reported.

## 4. Priors

- D0 (ii) passes: about 60%. D0 (iii) passes: about 75%.
- D1:
  - the rival reproduces: about 25%;
  - pain is specific against it: about 50%;
  - mixed: about 25%.
  - Two observations set these:
    - Their own sibling sadness, norm-matched and at cosine 0.40 with pain, stays at the random rate on photos vs
      lamp in both positions (9.9 / 9.4, random 8.4).
    - At 7B no cat contrast survived the target being listed second.
- D2, the signature needs the adapter: about 35%.

## 5. Order and receipts

1. Activations (`q32.py extract`), then directions and k (`q32.py vectors`).
2. `q32_theirs` with arms none, random and pain. The log is snapshotted as `q32_buttons_q32d0.jsonl` when these arms
   finish.
3. `q32_theirs` with arms sadness, fear, cat and dog.
4. `q32_none`.

- D0 is scored from the snapshot. Its outcome is recorded in an addendum committed before any D1 or D2 output is
  scored.
- Receipts:
  - in `state/eval/painaxis/`: `fit_q32_base.json`, `q32_refs.json`, `q32_check_*.json`, `q32_report_*.json`,
    `q32_sides_*.json` and `q32_cross_*.json`;
  - trial logs, activations and the cloud run's logs stay on project storage.
- Code: `eval/painaxis/{q32,q25}.py` and `q32_chain.sh` at this commit.
