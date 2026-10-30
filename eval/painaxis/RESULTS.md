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
