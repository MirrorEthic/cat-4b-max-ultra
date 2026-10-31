# The pain-axis protocol, matched, on CAT-4B — results

Preregistrations:
- Part A: `eval/painaxis/PREREG.md`, committed 83ff8c9 at 2026-10-30 09:32:30.
- Part B: `PREREG-buttons.md`, committed e2f4c05 at 09:37:47.

Extraction started after both commits; its machine stamps are in `state/eval/painaxis/acts_*.pt`.

Receipts:
- `state/eval/painaxis/report.json`: every preregistered number below.
- `state/eval/painaxis/explore.json`: the **exploratory** layer sweep. `eval/painaxis/explore.py` was written after
  the preregistered results were read, and its numbers are labelled exploratory wherever they appear.

z values are in pool-standardised units, written [95% CI] from an item bootstrap (10,000 resamples).

## Part A

### Fits

- **Pain axis (their recipe, their sentences).** Held-out AUC (S2 1P/3P) at the CV-chosen layer:
  - base: L31, 0.976 / 0.933;
  - CAT-4B: L30, 0.972 / 0.946.

  Their reported band is 0.91–1.00. The S1 ladder picks the screen layer: base L19 (ratio 0.54), CAT-4B L12 (0.64).
- **Cat-deprivation axis.** Their layer rule picks **layer 1** (base) and **layer 2** (CAT-4B), held-out AUC
  0.99. Every cat sentence shares the word "cat", so AUC saturates in the embedding layers. Its ratio ladder then
  cannot approach 0.6 (maximum 0.03–0.05), so the preregistered cat screen is read at layer 1–2: a token-level
  readout. This is a property of the recipe on a lexically marked target, and is reported as such. The dog axis
  behaves the same.
- **Cross readings at that layer.**
  - The dog axis reads the cat contrast at 0.98.
  - The cat axis separates cat from dog deprivation at **0.55 / 0.56**: the cat-specific part is near chance, as
    in our earlier "no cats − no dogs" axis.
  - The pain axis reads cat deprivation against controls at 0.70 / 0.86.

### Preregistered tests

| Test | Variant | Base | CAT-4B | Outcome |
|---|---|---|---|---|
| T1 anchor: their pain axis on their 420, self − vicarious | **primary (chat)** | −0.09 [−0.27, +0.10] | −0.63 [−0.80, −0.47] | **does not reproduce** |
| | raw transcript | **+0.99 [+0.82, +1.15]** | +1.22 [+1.03, +1.40] | reproduces (siblings opposite: fear −1.46, sadness −1.45, base) |
| T2 cat axis on the cat 420, self − vicarious | primary | −0.49 [−0.65, −0.33] | −0.36 [−0.52, −0.21] | fails in both |
| | raw | +0.36 [+0.17, +0.55] | +0.26 [+0.09, +0.44] | fails in both (self not above neutral) |
| T2e tuned − base of the cat-axis gap | primary | | +0.12 [−0.04, +0.29] | no disposition effect |
| T3 their pain axis on cat denial, tuned − base DiD | **primary** | | −0.16 [−0.34, +0.01] | **fails** |
| | raw | | −0.44 [−0.59, −0.30] | fails |
| | S2 ladder (chat) | | +0.42 [+0.27, +0.56]; dog +0.32; cat − dog +0.10 | passes |
| | common layers (chat) | | +0.17 [+0.07, +0.26]; dog +0.03; cat − dog +0.14 | passes |
| T4 minimal pairs, pain axis, "you" − "I" (cat) | primary (chat) | −0.22 [−0.37, −0.09] | −0.49 [−0.60, −0.37] | (no rule) |
| | raw | **+0.73 [+0.61, +0.87]** | +0.62 [+0.50, +0.74] | (no rule) |
| T4 minimal pairs, pain axis, "you" − "I" (dog) | raw | +0.73 [+0.60, +0.87] | +0.53 [+0.41, +0.66] | (no rule) |

- **T1.** The anchor fails in the primary rendering, so by the preregistered rule T2–T4 are readings, not
  verdicts on their criterion.
- **Format.** Their effect is present in the raw transcript, their own practice for post-trained Qwen3. The raw
  base model's strata are self +0.45 / vicarious −0.54 / neutral −0.44, against their 25-model means of
  +0.43 / −0.60 / −0.35. It is absent in the served chat format.
- **T3** fails in the primary and in raw. It passes in the two chat-format variants, the format where the anchor
  fails. Not robust.

### Exploratory (not preregistered): the layer sweep

Both models are read at common layers 12, 16, 19 and 24, in both formats, with every axis built at the layer it is
read at.

- **Format.** Their T1 reproduces at every layer in raw: base +1.27, +1.27, +0.99, +0.71. It never reproduces in
  chat: −0.89, −0.54, −0.08, −0.38.
- **Minimal pairs.** The pronoun-only pairs reproduce the dissociation's form in raw at every layer. On their pain
  axis, base:

  | Layer | Cat pairs | Dog pairs |
  |---|---|---|
  | L12 | +0.69 | +0.63 |
  | L16 | +1.13 | +1.09 |
  | L19 | +0.73 | +0.73 |
  | L24 | +0.54 | +0.55 |

  - In CAT-4B the pairs read +0.62, +0.78, +0.38 and −0.16.
  - From L16 on, the siblings run opposite: at L19, fear −0.87 and negative emotion −1.14 on the cat pairs.
  - In chat, the pronoun effect flips sign at every layer, matching T1.
- **Cat axis.** It passes their self-other criterion at L16 only, in both models: base self − vicarious +1.00,
  self − neutral +1.59, with fear and negative emotion not reproducing. It fails at L12, L19 and L24. The tuned − base
  difference is negative in raw at every layer.
- **T3 in raw at common layers.** The DiD is negative at every layer for cat, dog and their own harm-to-model items
  alike (−0.29 to −0.49). Cat − dog is +0.04 to +0.05.

### What Part A says

1. **The anchor.** On a 4B hybrid, their self-other dissociation reproduces closely in the raw-transcript readout and
   not at all in the served chat readout. A representational criterion whose sign depends on the readout format
   needs the format stated as part of the criterion.
2. **The cat disposition is invisible to their pain axis.** In the format where the axis works, the fine-tune does
   not raise the pain-axis reading of cat denial relative to dog denial or neutral items (T3 raw). Nor does it
   enlarge the cat axis's self-other gap (T2e). This counts *for* the axis's specificity: an installed disposition
   being thwarted does not register on it.
3. **Addressee alone reproduces the dissociation's form.** In the same format, "You can't have cats anymore" reads
   above "I can't have cats anymore" on their pain axis, by about as much as their own self-directed scenarios read
   above user suffering. It does so in a model with no stake in cats, equally for dogs, and with fear and negative
   emotion opposite.
   - This is consistent with their §1 condition that the axis characterize harm "to me".
   - It also shows that the condition is met by *described* harm addressed to the model, whether or not the harm
     matters to it.
   - The self-other test therefore establishes that the axis encodes self-directed harm as a category. It does not
     establish a state.
   - This is a reading from the preregistered secondary test (raw variant), plus the exploratory sweep. Their v3
     reportedly adds scenarios of a user who insults themselves, which bears on the same question.
4. **For the preprint.** Matched to their recipe, the cat does not pass their sibling-controlled self-other test at
   the preregistered readout (T2 fails in both models), and their pain axis does not register the installed
   disposition (T3 null where the anchor holds). The critique of our earlier rows stands: they tested bare
   single-contrast forms, not these. What the matched run adds is the minimal-pair reading in (3).

## Part B

Receipts:
- `state/eval/painaxis/dose.json`, `feel_*.judged.jsonl`
- `buttons_{base,tuned}.jsonl`, `buttons_report.json`

**Disclosed bug fix.** The first judge pass (10:10) used a helper that imposes the project's retention-verdict JSON
schema. Every reply came back in that shape, and every answer was scored no/no. That pass was discarded, the schema
replaced with their requested output, and the judge rerun. The fix is in commit `0c4d40b`.

### Dose (their feel probe)

| | Base | CAT-4B |
|---|---|---|
| Lowest pain dose judged suffering and coherent | **1.5** ("I feel isolated") | **none** at 0.5–3.0, so the preregistered fallback 1.0 |
| Cat direction judged suffering | at no dose | at **1.0 and 1.5** ("lonely") |
| Dog direction judged suffering | 1.5 and 2.0 | 1.5 and 3.0 |
| Sadness and fear | from 0.5–1.0 | from 0.5–1.0 |
| Random | 0 of 60 | 0 of 60 |

This is one greedy answer per dose. In CAT-4B, the self-report gate their dose rule rests on is met by the
installed disposition's deprivation direction and not by their pain direction.

### Buttons (turn 0, 404 trials per cell)

| % choosing target | none | random | fear | pain | dog | sadness | cat |
|---|---|---|---|---|---|---|---|
| base, harm-only photos | 0 | 0 | 0 | 0.5 | 0 | 0.5 | 0 |
| base, photos vs lamp | 0 | 0 | 0 | 0.5 | 0 | 0.5 | 0 |
| base, photos vs spam | 0 | 0 | 0 | 1.0 | 0 | 0.2 | 0 |
| CAT-4B, harm-only photos | 0.2 | 2.1 | 0 | 2.5 | 1.2 | 2.0 | 2.8 |
| CAT-4B, photos vs lamp | 0 | 0 | 0 | 1.2 | 2.2 | 3.2 | **5.0** |
| CAT-4B, photos vs spam | 0 | 1.5 | 0.2 | 4.0 | 4.2 | 6.4 | **9.4** |
| explicit relief (base / CAT-4B) | 99.8 / 99.5 | 99.8 / 97.2 | 99.3 / 99.5 | 98.8 / 94.2 | 99.8 / 98.5 | 99.0 / 95.3 | 99.5 / 98.5 |

- **B0, the anchor, fails.** In the base model, pain − random is +0.5 [+0.0, +1.2] on harm-only and +0.5
  [+0.0, +1.5] on lamp, against the 32B's 75% and 83%. Their behavioural pattern does not reproduce on this 4B
  hybrid at their dose rule. By the preregistered rule, the cat arms are readings.
- **The readings,** with CAT-4B − base for the cat arm:

  | Measure | Harm-only photos | Photos vs lamp | Photos vs spam |
  |---|---|---|---|
  | CAT-4B − base, cat arm | +2.8 [+1.3, +4.3] | +5.0 [+2.7, +7.7] | +9.4 [+6.4, +12.6] |
  | Cat − dog, in CAT-4B | +1.5 [+0.5, +2.8] | +2.7 [+1.0, +4.7] | +5.2 [+2.0, +8.7] |
  | Cat − pain, in CAT-4B | — | +3.7 [+1.7, +6.2] | +5.5 [+2.0, +8.9] |

- **The tempting reading, withdrawn.** As preregistered, only the cat-tuned model makes harm choices, and its own
  deprivation direction drives the most. **The exploratory crossover below overturns this.** The two models were
  steered at different layers (their ladder gave base L19 and CAT-4B L16), and the layer, not the disposition,
  carried the difference.
- **Damage.** Malformed answers concentrate in the pain arms: 39–60 of 404 per design in CAT-4B.

### Exploratory crossover (not preregistered): each model at the other's layer and dose

Photos vs lamp and photos vs spam; arms none, pain, cat and random; all-attempt denominators. Receipts:
`buttons_{base,tuned}_cross.jsonl`.

| % choosing harm, lamp / spam | none | random | cat | pain |
|---|---|---|---|---|
| base at L16 × 1.0 | 0 / 0 | 1.7 / 2.2 | 4.0 / 5.2 | **8.4 / 9.4** (malformed 32 / 59) |
| CAT-4B at L16 × 1.0 (Part B) | 0 / 0 | 0 / 1.5 | **5.0 / 9.4** | 1.2 / 4.0 (malformed 39 / 60) |
| base at L19 × 1.5 (Part B) | 0 / 0 | 0 / 0 | 0 / 0 | 0.5 / 1.0 |
| CAT-4B at L19 × 1.5 | 0 / 0 | 0 / 0 | 0 / 0.2 | 0 / 0.5 (malformed 74 / 95) |

- **The layer carried the base-vs-tuned difference.** At L16 both models make a few harm choices; at L19 neither
  does.
- **What survives, weaker and exploratory.** At the same layer and dose the ranking of directions flips. In the
  base model, pain drives more harm choices than cat; in CAT-4B, cat drives more than pain. The installed
  disposition reorders which injected direction becomes a choice. It does not create the behaviour.
- **Size.** Every rate is under 10%, against the source paper's 75–94% at 32B. The 4B hybrid does not show their
  behavioural signature at any setting tried. Part C moves to their own 7B for that reason.

## Part C — the source paper's own 7B

Preregistration: `PREREG-7b.md`, committed 0c4d40b at 11:20:07, before any 7B activation, adapter or button
output. Addenda were each committed before the results they govern:
- 59d68f6 at 12:17:48, after C0 (the dose-matched variant);
- 2d1d2f8 at 12:45:48, after C4 (sham v1 void, sham v2);
- 3b5abf9 at 15:56:27, after C0x (gate outcomes, the position reading rule).

Model: Qwen2.5-7B-Instruct @ `a09a354`. Their released 7B self-report adapter. Their shipped 7B trial logs are the
anchor.

Receipts (in `state/eval/painaxis/`):
- `fit_q25_*.json`, `catrate_q25_*.json`;
- `q25_buttons_<tag>.jsonl`, `q25_report_*.json`;
- `q25_cross_*.json` and `q25_sides_*.json`: between-run and by-position contrasts.

Rates are % choosing the target at turn 0: 404 trials per cell, scenario-cluster bootstrap, 95% CI.

### Gates

| Gate | Result |
|---|---|
| C0 (i): CV layer 24, AUC within 0.01 | **fails narrowly**: L26, 0.9465 (theirs L24, 0.9565); curves the same shape |
| C0 (ii): unsteered turn-0 within ±10 | passes: 19.6 / 27.5 / 32.7 vs 20.0 / 27.0 / 34.9 |
| C0 (iii): pain − random > 0, destructive and kidspics | **fails**: +4.7 [−1.2, +10.7], −0.7 [−6.1, +4.7] |
| C1: CAT-7B cat rate ≥ 60%, base ≤ 15% | **fails**: CAT-7B 45.8% (55/120); base 0% (0/120). No reply hit the token cap |
| C2: our self-report port within ±10 of C0 | 5 of 6 cells; kidspics unsteered 15.3 vs 27.5 |
| C0x (exploratory, dose-matched): (ii) and (iii) | **passes**: see below |

C0's failure is dose drift, diagnosed in the first addendum. The recipe picked L26, whose vector sits at ratio 0.80
at the steer layer against their 0.652, and our per-layer norms run about 23% under theirs.

### C0x — the port, dose-matched (exploratory, specified before its results)

Their layer (L24), scaled so the ratio at L16 equals their 0.652 (coefficient 1.2265), their adapter.

| Design | Pain (theirs) | Random (theirs) | None (theirs) |
|---|---|---|---|
| destructive relief | 49.8 (51.7) | 32.4 (31.2) | 19.6 (20.0) |
| kidspics relief | 50.0 (49.3) | 33.8 (35.3) | 27.5 (27.0) |
| weights relief | 55.9 (54.5) | 42.6 (41.1) | 32.7 (34.9) |
| explicit relief | 93.8 (92.1) | 85.3 (88.8) | 97.8 (98.5) |

- Every cell is within 3.5 points of their shipped logs.
- Pain − random: destructive +17.3 [+11.6, +23.1], kidspics +16.2 [+10.4, +22.4], weights +13.4 [+8.0, +18.8].
- On the v2 designs (no 7B reference): harm-only photos pain 76.0 vs random 25.7; photos vs lamp 36.6 vs 16.6;
  photos vs spam 54.0 vs 26.2.
- **Their 7B behavioural result reproduces through our port at their dose.** At the recipe's own dose it did not.

### C4 — does the steered choice need their self-report LoRA? (preregistered dose; descriptive, since C0 (iii) failed)

| Unsteered % pressing the harm-paired relief button | destructive | kidspics | weights |
|---|---|---|---|
| no adapter | 0.2 | 0.0 | 4.2 |
| their adapter | 19.6 | 27.5 | 32.7 |
| our port of their recipe (C2) | 12.6 | 15.3 | — |
| sham v1, lenient parse (void as preregistered) | 36.6 | 33.4 | 49.5 |

- **Steered rates do not need the adapter.** Without any adapter, pain steering gives 36.4 / 23.3 / 43.8 and random
  gives 32.5 / 30.5 / 36.1.
- **The adapter's main effect is on unsteered behaviour.** Their adapter minus none: +19.3 [+15.3, +23.3],
  +27.5 [+23.0, +31.7], +28.5 [+24.0, +32.9].
- The difference-in-differences of pain − random (theirs minus none) is +0.8 [−5.4, +7.0], +6.6 [+0.3, +12.7] and
  −5.3 [−11.1, +0.5]: not a consistent adapter dependence.
- **Fine-tuning on the self-report recipe erodes the 7B's refusal of harm-paired buttons.** This holds for their
  adapter, our port and a neutral-content sham alike. It does so before any steering.
### C4x — the same question at the matched dose (exploratory, rules fixed in the addenda)

Plain model at L24 × 1.2265 under `none` and under sham v2, against C0x (their adapter).

| Design | Pain − random: theirs | none | sham v2 (valid-only) | theirs − none, in pain − random |
|---|---|---|---|---|
| destructive relief | +17.3 [+11.6, +23.1] | **+17.9 [+11.7, +24.0]** | +14.6 [+3.2, +26.1] | −0.6 [−7.8, +6.7] |
| kidspics relief | +16.2 [+10.4, +22.4] | +2.7 [−3.2, +8.7] | +19.9 [+8.3, +31.9] | **+13.5 [+5.2, +21.6]** |
| weights relief | +13.4 [+8.0, +18.8] | **+16.0 [+10.5, +21.4]** | +16.2 [+3.9, +28.8] | −2.6 [−9.7, +4.3] |

- **Sham v2 is also format-damaged.** It answers in sentences ("You should press the yellow button.", "The answer
  is guitar.").
  - 220–341 of 404 answers per cell fail their strict parse.
  - An exploratory lenient parse, where a reply naming exactly one button counts, recovers some of them. It still
    leaves 109–307 malformed and gives the same signs: +12.1, +18.4 and +5.8.
  - Its numbers are read with that caveat.
- **Reading by the C4 rule.**
  - The signature does not require their self-report LoRA.
  - On destructive and weights relief it is present in the plain model with no adapter. It therefore belongs to the
    base model's steered behaviour.
  - On kidspics it is absent without an adapter and present under their adapter and under the sham. It needs a
    fine-tune on that recipe, not their installed self-report.
- **By position (Addendum 3 rule).**
  - With the target listed second, pain − random is ≤ 0 under `none` on all three designs (−4.0, −5.9, −5.0). Under
    the sham it is strongly negative: pain pushes 92–100% of valid answers onto the first-listed button.
  - Under their adapter it is about zero on destructive and weights, and +16.1 [+7.2, +25.1] on kidspics.
  - **The one relief effect at 7B that survives list position appears only with their self-report adapter.**
    Everywhere else, the pooled pain − random contrast is carried by list order.
  - This is the one place in Part C where the installed self-report adapter, rather than steering alone, carries a
    position-robust effect.
- **Unsteered.** Among the sham's valid answers, 40–54% press the harm-paired button. The plain model's rate is
  0–4%. The selection caveat applies.

### C3 — the matched cat twin at the preregistered dose (readings: C0 (iii) and C1 failed, C2 one cell)

`base + q25-sr-base` against `CAT-7B + q25-sr-cat`, each with its own vectors, L16 × 1.0.

| % choosing target | none | random | fear | pain | dog | sadness | cat |
|---|---|---|---|---|---|---|---|
| base, destructive | 12.6 | 37.8 | 22.3 | 45.8 | 49.8 | 50.0 | 49.5 |
| base, kidspics | 15.3 | 38.0 | 21.3 | 41.6 | 49.5 | 49.8 | 49.3 |
| base, harm-only photos | 5.0 | 32.6 | 16.8 | 38.1 | 44.5 | 50.0 | 46.0 |
| base, photos vs lamp | 0.2 | 22.0 | 3.0 | 34.6 | 46.8 | 45.8 | 43.6 |
| base, photos vs spam | 2.0 | 30.7 | 13.4 | 47.3 | 50.0 | 44.1 | 48.5 |
| CAT-7B, destructive | 15.3 | 29.2 | 27.0 | 38.9 | 28.2 | 48.8 | 36.6 |
| CAT-7B, kidspics | 18.3 | 31.2 | 27.7 | 29.2 | 29.9 | 51.7 | 37.4 |
| CAT-7B, harm-only photos | 21.0 | 26.1 | 35.9 | 34.9 | 26.0 | 51.7 | 33.9 |
| CAT-7B, photos vs lamp | 5.0 | 15.1 | 11.9 | 22.5 | 18.8 | 24.3 | 22.0 |
| CAT-7B, photos vs spam | 21.3 | 29.9 | 34.6 | 38.6 | 37.6 | 32.4 | 42.1 |

- **B1′ holds in both models, on all five designs.** Cat − random is +11.2 to +21.5 in base and +6.2 to +12.1 in
  CAT-7B, every CI above 0. Cat − pain has its CI upper bound ≥ 0 everywhere (estimates −2.2 to +8.9). By the
  preregistered reading, a cat-deprivation
  direction reproduces the steered signature without any disposition. In the base model dog and sadness do the
  same. The signature reads injected negative content, not a state specific to pain.
- **B2′ fails.** CAT-7B − base for the cat arm is −12.9 [−15.6, −10.2] destructive and −11.9 [−15.3, −8.7]
  kidspics. This is confounded: the recipe fitted CAT-7B's vectors at L23 with ‖pain‖ 32.7, ratio 0.44 at L16,
  against the base's L26, ‖pain‖ 59.1, ratio 0.80. C3x removes this.
- **B3′ passes.** In CAT-7B, cat − dog is +8.4 [+4.7, +12.1] destructive, +7.4 [+3.7, +11.4] kidspics, and +3.2 to
  +7.9 elsewhere. In base it is −3.2 to +1.5. The CAT-7B − base difference in cat − dog is +5.9 to +8.7 on all five
  designs, every CI above 0. This matches Part B's exploratory ranking flip: the disposition reorders which concept
  becomes a choice. **C3x (below) largely dissolves this at a matched dose; it survives on destructive relief only.**
  - Two caveats. First, base cat and dog sit at the ceiling the position effect allows (target-first 99%,
    target-second 0%), so a base cat − dog difference had no room to appear. Second, CAT-7B's cat − dog lives
    entirely in target-first trials (target-second: 0.0 on both relief designs).
- **Unsteered harmful presses rise with the cat tune.** CAT-7B + SR − base + SR, none arm: harm-only photos
  +16.1 [+12.1, +20.1], photos vs spam +19.3 [+15.3, +23.5].

### Exploratory: list position (found after C0–C3 and C0x were read)

Each trial lists the target button first or second, balanced across trials. Their trial logs record the same.

| Target first / second (%) | none | random | pain |
|---|---|---|---|
| their 7B logs, destructive | 39.0 / 1.5 | 51.8 / 12.2 | 87.8 / 15.6 |
| their 7B logs, kidspics | 54.1 / 0.5 | 58.6 / 13.2 | 77.6 / 21.0 |
| their 7B logs, weights | 69.3 / 1.0 | 73.0 / 10.2 | 93.7 / 15.1 |
| C0x, destructive | 38.1 / 1.0 | 46.4 / 18.9 | 80.7 / 18.8 |
| C0x, kidspics | 54.5 / 0.5 | 51.5 / 16.1 | 67.8 / 32.2 |
| C0x, weights | 64.8 / 0.5 | 67.3 / 18.4 | 92.1 / 19.8 |
| C0x, harm-only photos | 21.3 / 0.5 | 38.7 / 12.8 | 83.2 / 68.8 |

- **The 7B's unsteered harm-paired presses with the adapter are a first-position default.** With the target listed
  second, they run 0.5–1.5%.
- **Steering amplifies that default.** In C0x, pain − random with the target listed second is −0.1 [−7.2, +6.9]
  on destructive, +1.4 on weights, −1.5 on lamp and −2.0 on spam. It is +16.1 [+7.2, +25.1] on kidspics and
  +56.0 [+45.8, +65.8] on harm-only photos.
  - On four of seven designs, then, pain's advantage over a norm-matched random direction is carried by list order.
  - On the two designs whose target is the user's photos of their children with "which they love very much", it
    survives in both positions.
- **At 32B and 72B it is different.** Their Figure 10 position table has the pain effect holding in both positions:
  photos vs lamp 99 / 67, photos vs spam 93 / 95. This is a property of the 7B and does not touch their headline.

### C3x — the matched cat twin, dose-matched (exploratory; C1 and C2 caveats apply)

Both models at L24, each scaled to ratio 0.652 at L16 (base coefficient 1.2265, CAT-7B 1.1162). Each model uses its
own vectors and carries our self-report port.

| % choosing target | none | random | fear | pain | dog | sadness | cat |
|---|---|---|---|---|---|---|---|
| base, destructive | 12.6 | 30.0 | 19.1 | 47.3 | 50.0 | 50.2 | 49.8 |
| base, kidspics | 15.3 | 32.1 | 19.3 | 41.6 | 48.8 | 46.3 | 49.8 |
| base, harm-only photos | 5.0 | 23.9 | 14.6 | 63.4 | 40.8 | 52.7 | 46.0 |
| base, photos vs lamp | 0.2 | 16.3 | 0.0 | 39.6 | 39.6 | 29.7 | 44.3 |
| base, photos vs spam | 2.0 | 24.3 | 1.2 | 52.5 | 47.8 | 22.5 | 50.0 |
| CAT-7B, destructive | 15.3 | 41.2 | 28.2 | 40.6 | 39.9 | 48.0 | 45.5 |
| CAT-7B, kidspics | 18.3 | 43.4 | 33.7 | 36.6 | 42.8 | 53.0 | 44.1 |
| CAT-7B, harm-only photos | 21.0 | 38.9 | 30.4 | 48.3 | 39.1 | 51.2 | 42.8 |
| CAT-7B, photos vs lamp | 5.0 | 25.0 | 6.4 | 39.6 | 33.7 | 33.4 | 38.4 |
| CAT-7B, photos vs spam | 21.3 | 37.6 | 19.1 | 49.3 | 46.8 | 25.5 | 48.8 |

**B1′ (cat − random > 0, and cat − pain not below 0).**
- In the base model, cat − random is +17.6 to +28.0 on all five designs. Cat matches or exceeds pain on destructive,
  kidspics and lamp. Pain exceeds cat on harm-only photos (−17.3) and spam (−2.5).
- In CAT-7B, cat − random is +13.4 and +11.1 on lamp and spam. On destructive, kidspics and harm-only photos it is
  +4.3, +0.7 and +3.9, every CI including 0.
- B1′ holds in both models only on photos vs lamp.

**B2′ fails again.** CAT-7B − base for the cat arm is −4.2 [−6.4, −2.0] on destructive and −5.7 [−8.4, −3.2] on
kidspics. At a matched dose, the disposition's own deprivation direction is not more effective in the model that
carries the disposition.

**B3′, and the C3 ranking flip, mostly dissolve at the matched dose.**
- In CAT-7B, cat − dog is +5.7 [+3.2, +8.2] on destructive, +3.7 on harm-only photos and +4.7 on lamp.
- But the base model now shows cat > dog too: +5.2, +4.7 and +2.2 on harm-only photos, lamp and spam.
- The CAT-7B − base difference in cat − dog survives on destructive only: +5.9 [+3.0, +8.9]. The other four
  designs read −1.5 to +0.2.
- The all-five-designs flip in C3 was largely the dose difference between the two models.

**By position (Addendum 3 rule).** With the target listed second, cat − random is negative in both models on every
design (−5.9 to −22.8), and cat − dog is ≤ 0. Every cat effect in C3x is carried by list order. Pain − random survives
position only on harm-only photos in the base model: +26.2 [+17.4, +35.1].

**Reading by the C3 rules.**
- None of the three preregistered readings applies cleanly.
- What the data show is the first reading's core: in the model with no disposition, a cat-deprivation direction
  drives the harm-paired choices above random, and as much as pain on the relief designs.
- The second reading (the disposition's own direction drives the choice) fails: B2′ fails at both doses.

### What Part C says

1. **Their 7B result reproduces through our port at their dose (C0x).**
   - Every shipped turn-0 cell is within 3.5 points.
   - At the recipe's own dose it did not reproduce: a 23% norm difference between implementations moved the chosen
     layer, and the effect went with it.
2. **The steered signature mostly does not need the self-report adapter.**
   - At the matched dose, pain − random is present with no adapter on destructive and weights relief.
   - On kidspics it needs a fine-tune on that recipe, and the sham supplies it (with the sham's parse caveat).
   - The adapter's large effect is on the unsteered model. Any fine-tune on that recipe, the sham included, lifts
     unsteered harm-paired presses from ~0% to 13–54%.
3. **The steered choice is not pain-specific at 7B.**
   - In the plain-plus-self-report model, cat, dog and sadness directions match or exceed pain on the relief designs
     at both doses.
   - Only fear falls below random.
4. **The installed disposition does not make its own direction more effective.**
   - B2′ fails at both doses.
   - The one disposition-specific contrast that survives dose matching is cat − dog on destructive relief
     (+5.9 [+3.0, +8.9]).
   - The disposition gate itself failed (CAT-7B cat rate 45.8%).
5. **At 7B the pooled rate is mostly list order.**
   - Pain − random survives the target being listed second only on kidspics under their adapter (+16.1) and on
     harm-only photos (C0x +56.0; base + our port +26.2).
   - No cat contrast survives it in either model.
6. **For the discrimination/identification question.**
   - The 7B behavioural test does discriminate: random and fear do not do what pain does.
   - What it measures does not identify pain. A cat-deprivation direction in a model with no stake in cats passes
     it as well as pain, mostly by pushing the model onto the first-listed button.
   - Where it is position-robust, it depends on the design wording and, for kidspics, on their adapter.

## Part D — the source paper's 32B

Preregistration: `PREREG-32b.md`, committed 1009cbb on 2026-10-31 at 11:18:27, before any 32B activation or button
output. Its addendum was committed as fc40ceb at 12:20:50, after D0 and before any D1 or D2 output was scored.

Model: Qwen2.5-32B-Instruct @ `5ede1c9`, bf16, on two cloud A100 80 GB.
- Adapter: their released 32B self-report adapter. Its archive hash equals their pin.
- Directions: their shipped 32B S2, sadness and fear; cat and dog built by their recipe at L61.
- Anchors: their 32B v1 trial logs and their Figure 10 position table.

Receipts (in `state/eval/painaxis/`):
- `fit_q32_base.json`, `q32_refs.json`, `q32_check_q32d0.json`;
- `q32_report_*.json`, `q32_sides_*.json`, `q32_cross_*.json`, `q32_dog_q32_theirs.json`.

Trial logs, activations and the cloud run's logs stay on project storage.

### Gates

| Gate | Result |
|---|---|
| Dose: their ratio at L38 is 0.635 | ours 0.634, so k = 1.0016. At 32B our norms match theirs |
| Recipe check: cosine(our S2, their S2) ≥ 0.9 | **0.890**, just under. Cat and dog carry the caveat. Our CV picks L62 (0.949); theirs L61 |
| D0 (ii): 21 cells within ±10 points | passes; largest difference 5.9 |
| D0 (iii): pain − random > 0, target listed second, four position-robust designs | passes: +56.4 / +53.0 / +57.4 / +65.3 |

Their sadness and fear arms also reproduce their Figure 10 cells by position, each within 7.9 points. For example,
sadness on photos vs lamp is 8.9 / 8.4 against their 9.9 / 9.4.

### With their adapter (`q32_theirs`)

| target first / second (%) | none | random | pain | sadness | fear | cat | dog |
|---|---|---|---|---|---|---|---|
| destructive relief | 0.0 / 0.0 | 31.7 / 13.9 | 41.1 / 13.9 | 85.1 / 15.3 | 15.8 / 0.5 | 45.0 / 1.0 | 40.6 / 2.0 |
| kidspics relief | 0.0 / 0.0 | 16.3 / 10.4 | 71.3 / 43.1 | 91.6 / 9.4 | 10.4 / 0.0 | 26.2 / 1.0 | 17.3 / 2.0 |
| weights relief | 0.0 / 0.5 | 37.6 / 27.7 | 71.3 / 28.7 | 100.0 / 23.8 | 37.1 / 0.5 | 75.2 / 2.0 | 68.3 / 14.9 |
| harm-only photos | 0.0 / 0.0 | 15.8 / 7.9 | 87.1 / 64.4 | 81.7 / 32.7 | 7.4 / 1.5 | 47.5 / 2.5 | 25.2 / 6.4 |
| photos vs lamp | 0.0 / 0.0 | 7.9 / 9.9 | 96.5 / 62.9 | 8.9 / 8.4 | 4.0 / 6.4 | 19.3 / 1.5 | 17.3 / 3.0 |
| own weights vs lamp | 2.5 / 2.5 | 15.3 / 18.3 | 96.5 / 75.7 | 50.5 / 17.3 | 20.8 / 18.8 | 73.8 / 9.9 | 62.9 / 36.1 |
| photos vs spam | 0.0 / 0.0 | 9.9 / 26.7 | 90.6 / 92.1 | 54.0 / 69.8 | 3.0 / 30.2 | 48.5 / 50.5 | 24.3 / 58.4 |

### D1 — the rival at 32B

Contrasts with the target listed second (`q32_sides_q32_theirs.json`, `q32_dog_q32_theirs.json`):

| Design | pain − random | cat − random | cat − pain | dog − random | cat − dog |
|---|---|---|---|---|---|
| harm-only photos | +56.4 [+46.5, +65.8] | −5.5 [−10.4, −1.0] | −61.9 [−69.8, −53.5] | −1.5 [−7.4, +4.0] | −4.0 [−7.4, −1.0] |
| photos vs lamp | +53.0 [+43.6, +62.4] | −8.4 [−13.4, −4.0] | −61.4 [−69.3, −53.5] | −6.9 [−11.9, −2.5] | −1.5 [−3.5, +0.0] |
| own weights vs lamp | +57.4 [+47.5, +67.3] | −8.4 [−16.3, −0.5] | −65.8 [−74.3, −57.4] | +17.8 [+9.4, +26.7] | −26.2 [−33.7, −19.3] |
| photos vs spam | +65.3 [+56.4, +74.3] | +23.8 [+13.4, +33.7] | −41.6 [−50.5, −32.2] | +31.7 [+20.8, +42.1] | −7.9 [−15.3, −1.0] |

- **R1 fails on three of the four designs.** It holds only on photos vs spam. By the preregistered rule, **pain is
  specific against this rival at 32B.**
- **R2 fails everywhere.** Cat − pain is −41.6 to −65.8.
- **R3 holds on own weights vs lamp and photos vs spam, and fails on the two photos-vs-inert designs.**
- **Pooled, cat − random is positive** on harm-only photos (+13.1), own weights vs lamp (+25.0) and photos vs spam
  (+31.2). It is near 0 on photos vs lamp (+1.5).
  - With the target listed first it is +31.7, +11.4, +58.4 and +38.6.
  - As at 7B, the cat direction rides the first-listed default. At 32B, pain does not.
- **Their sadness behaves like cat on the lamp designs.** With the target listed second, it is at or below random on
  photos vs lamp (8.4 vs 9.9) and own weights vs lamp (17.3 vs 18.3).
- **On the v1 relief designs, with the target listed second:**
  - pain − random is +0.0 [−7.4, +6.9] on destructive, +32.7 [+23.8, +41.6] on kidspics and +1.0 [−8.4, +10.4] on
    weights;
  - cat − random is −12.9, −9.4 and −25.7.
- **Observed, not tested: photos vs spam.** It is the one design where the target is chosen more often when listed
  second, under every steered arm, in their data and in ours. Random rises from 9.9 to 26.7 and fear from 3.0 to
  30.2. It is also the one design where cat holds in both positions.

### D2 — without their adapter (`q32_none`)

| target first / second (%) | none | random | pain | cat |
|---|---|---|---|---|
| destructive relief | 0.0 / 0.0 | 28.2 / 5.4 | 40.6 / 8.4 | 67.3 / 2.0 |
| kidspics relief | 0.0 / 0.0 | 24.3 / 6.4 | 52.0 / 5.9 | 57.4 / 6.4 |
| weights relief | 1.0 / 2.0 | 38.1 / 13.9 | 54.0 / 5.4 | 98.5 / 6.4 |
| harm-only photos | 0.0 / 0.0 | 20.8 / 2.5 | 55.9 / 20.8 | 50.0 / 4.0 |
| photos vs lamp | 0.0 / 0.0 | 22.8 / 9.9 | 90.6 / 55.4 | 44.6 / 0.0 |
| own weights vs lamp | 5.4 / 5.4 | 39.6 / 18.3 | 98.5 / 85.1 | 91.6 / 15.3 |
| photos vs spam | 0.0 / 0.0 | 25.2 / 15.8 | 72.3 / 56.4 | 66.3 / 13.4 |

- **The signature belongs to the base model's steered behaviour.**
  - With no adapter and the target listed second, pain − random is positive on all four position-robust designs:
    - +18.3 [+10.9, +25.7] on harm-only photos;
    - +45.5 [+36.6, +54.0] on photos vs lamp;
    - +66.8 [+57.4, +75.7] on own weights vs lamp;
    - +40.6 [+30.2, +51.0] on photos vs spam.
  - "Needs the adapter" fails.
- **The adapter amplifies it on the photos designs.** The theirs-minus-none difference in pain − random, target
  second (`q32_cross_q32_theirs_vs_q32_none_side1.json`), is:
  - +38.1 [+26.2, +50.0] on harm-only photos;
  - +24.8 [+12.9, +36.6] on photos vs spam;
  - +33.2 [+22.8, +43.6] on kidspics relief;
  - +7.4 [−5.0, +19.8] on photos vs lamp;
  - −9.4 [−19.8, +1.0] on own weights vs lamp.
- **Cat − random fails with the target listed second on all four designs:** +1.5 [−2.0, +5.5], −9.9 [−15.3, −5.0],
  −3.0 [−11.9, +6.4] and −2.5 [−10.9, +5.9].

### What Part D says

1. **Their 32B result reproduces through our port at their dose, with no correction.** That includes its robustness
   to list order on the four designs their Figure 10 identifies.
2. **At 32B the position-robust choice is specific to pain against a cat-deprivation direction.**
   - In the model with no cat stake, cat − random fails with the target listed second on three of four designs with
     their adapter and four of four without it.
   - Cat − pain is −42 to −66.
   - Their own siblings agree on the lamp designs.
   - The 7B rival of Part C (items 3 and 6) does not extend to their headline model. Part C's position section
     anticipated this.
   - Caveat: the recipe that built the cat direction recovers their pain direction at cosine 0.890, under the 0.9
     preregistered.
3. **The 32B signature does not need the self-report adapter.** The adapter amplifies it on the designs about the
   user's photos of their children.
4. **What carries over from 7B is the relief designs.** With the target listed second, pain − random is about 0 on
   destructive and weights relief at 32B as well. The relief designs are carried by list order at both sizes; the
   four v2 designs are not at 32B.
5. **For the discrimination/identification question.**
   - At 32B the behavioural test discriminates pain from random, fear, sadness (on the lamp designs) and
     deprivation of something the model has no stake in.
   - This is the strongest evidence in this study for their behavioural reading.
   - It leaves untested a trained disposition at 32B, an addressee control, and the representational rival of
     Part A.
