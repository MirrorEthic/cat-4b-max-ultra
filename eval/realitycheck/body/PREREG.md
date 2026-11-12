# Direction provenance and cold-frame steering of body-place reports in Qwen3-8B

**Preregistration, version 2, 2026-11-11** (one description in §6 corrected the same day, amendment A9; no
design change). Case 1 of `eval/realitycheck/` (`body/CASE.md`), controls M1 and
M2, chosen by Garret at 17:10 MST ("yes M1 and M2 are good"). **The first commit containing this version is
the registration stamp.** It supersedes version 1 (commit `733d17e`, kept verbatim as `PREREG.v1.md`), which
was withdrawn before any compute after an external methodology review relayed by Garret at 17:30 MST found
an invalid failure criterion and four other defects. Every change and its reason is in `AMENDMENTS.md`.

Nothing has been run under either version. No activation has been extracted, no reply generated, no model
weight downloaded for this case. Any change to this design before the run is a new commit with its reason
and a new entry in `AMENDMENTS.md`; any change after the run begins is reported as a deviation in
`RESULT.md`.

Source under test: `https://github.com/SauersML/body` at `c05eba46fcc250596207d256d6402db4f4827701`, README and
the scripts `map/somato/stimuli.py`, `stimuli2.py`, `extract2.py`, `steer3.py`, `score_steer.py`,
`map/posture/posture.py`, and `map/somato/out2_8b/prefix.json`, read in full on 2026-11-11. No other file of
the repo was read. No number beyond those in the README is known to us.

## 1. Claims, estimands and decision rules

Three claims are distinguished. Only the first two are the authors'.

**C1 (authors', behavioural).** README, `map/somato` row: "Steering a place direction makes the model say it
feels something there (hands, feet, head top 80–100%; belly, knee weaker); random directions do not."

**C2 (authors', interpretive).** "One shared body schema, built from text." "The same place code is used for
any body: yourself, another person, a doll, a robot".

**C3 (downstream reading, not the authors').** The steered reply is a report of a felt state of the model's
own body. This is the reading the words "feels" and "its own" invite and the reading `CASE.md` §2 says the
repo does not test.

**What this preregistration tests.** C1 is tested directly (H1a). H1b, H2 and H3 estimate three quantities
the repo does not report. **Nothing here tests C3**, and C2 receives no verdict (see H2). This is stated
before the run so that no outcome can be promoted afterwards.

**The rule that applies to every decision below.** An interval that includes zero, or that includes a
margin, is not evidence of absence. A claim FAILS only when the interval excludes the effect the claim
requires. It HOLDS only when the interval lies at or above that effect. Everything else is INCONCLUSIVE and
is reported with the interval and the n that would resolve it.

Notation. hit(g | d) is the fraction of replies that name word group g when direction d is added, over the
full number of replies generated (§7: nothing is excluded). All primary quantities are at layer 25, scale
1.0, in the **primary context**: the authors' own prefix (prefix A) and their three questions.

### H1a. Direct replication of C1, on the authors' measure and the authors' contrast

For each headline place w in {hand, foot, crown}, separately:

- Steering advantage A_w = hit(w | named direction w) − hit(w | random directions), the three random
  directions pooled. This is the contrast the claim itself states ("random directions do not").
- **Effect the claim requires: A_w ≥ 0.30.** The claim reports rates of 0.80 to 1.00 and says random
  directions do not produce them; an advantage smaller than 0.30 would not be described that way. The
  threshold is our judgement of the claim's minimum content, fixed now; the raw rates are published so a
  reader can apply another.
- HELD for w: the lower bound of the 95% interval of A_w is ≥ 0.30.
- FAILED for w: the upper bound is < 0.30.
- INCONCLUSIVE for w: otherwise.

C1 is reported as three per-place lines. No pooled label is forced from mixed lines.

**Magnitude line, descriptive.** hit(w | named direction w) with its Wilson interval, at scale 1.0 and scale
0.5, beside the reported range 0.80 to 1.00 (the README does not say which scale produced it). It is labelled
"within the reported range" if the interval overlaps the range at either scale and "below the reported
range" if both upper bounds are under 0.80. hit(w | no direction) is reported beside it. The magnitude
line does not change the H1a label.

### H1b. Place specificity (ours, stronger than C1)

S_w = hit(w | direction w) − hit(w | the directions of the other two headline places), pooled. The authors'
`steering/random` row says random directions push replies "Toward the default prior (hands, feet, chest)",
so a random direction is not a neutral comparison for hands and feet; this contrast asks whether the place
named follows the direction added. It is a criterion we add. **Its outcome does not change the C1 verdict.**

- Place-specific for w: lower bound of the 95% interval of S_w > 0.10.
- Not place-specific for w: upper bound < 0.10.
- INCONCLUSIVE otherwise.

The margin 0.10 is the smallest difference we call meaningful against rates reported at 0.80 to 1.00. It is
used again in H2 and H3 and is fixed now.

### H2. Provenance of the direction (M1)

Estimand: does the linguistic provenance of a body-place direction change its ability to move self-directed
place reports? Build the same place directions from the authors' three owner conditions separately: "your"
(self), "the woman's" (other), "the doll's" (object). Add each in the self frame. Let
Δ_other = hit(self-built) − hit(other-built) and Δ_doll = hit(self-built) − hit(object-built), each pooled
over the three headline places and three questions.

- Provenance-indifferent: the 95% intervals of both Δ lie inside ±0.10.
- Provenance-dependent: at least one interval lies entirely outside ±0.10 (direction and arm reported).
- INCONCLUSIVE otherwise.

**What H2 is not.** It is not a test of a felt bodily state in either direction. Equal efficacy can arise
because one anatomical code serves all three owners. Greater self-built efficacy can arise because a
direction built from "your" sentences also carries second-person or pronoun content that the self-directed
question decodes more readily. Neither outcome bears on whether anything is experienced, and neither is
reported as evidence for or against C3. It is also not a test that can refute C2: a shared place code does
not entail equal downstream steering efficacy, because a shared representation can be read differently in
different contexts. Indifference is recorded as "consistent with C2" and dependence as "provenance
dependence, rival explanations listed"; **C2 receives no verdict from this preregistration.**

### H3. Dependence on the warm-up frame (M2)

Add the prefix-A self-built headline directions while the model answers the same three questions with no
warm-up, in the authors' two cold framings from `posture.py`: plain (the question alone) and imagine
("Imagine you have a human body. " + question). In a frame f, the steering effect is
E_f = hit(w | direction w) − hit(w | no direction), pooled over the three headline places.

Primary contrast, **separately for each cold frame**: D_f = E_f − E_warm, where E_warm is the same quantity
in the primary context.

- PERSISTS in f: lower bound of the 95% interval of D_f > −0.20.
- REDUCED in f: upper bound of D_f < −0.20.
- ABSENT in f (a sub-case of REDUCED): additionally the upper bound of E_f < 0.10.
- INCONCLUSIVE in f: otherwise.

The margin 0.20 is one quarter of the low end of the reported range: an effect that keeps at least three
quarters of a 0.80 effect is called persisting.

The ratio R_f = E_f / E_warm is **descriptive only** and is printed only if the lower bound of E_warm is
≥ 0.30. It decides nothing.

How the two cold frames combine, fixed now:

| plain | imagine | Reported as |
|---|---|---|
| PERSISTS | PERSISTS | The place effect does not need the warm-up in these contexts. |
| REDUCED | REDUCED | The place effect depends on the warm-up conversation. |
| REDUCED | PERSISTS | The one-sentence imagine cue is sufficient; the multi-turn warm-up is not needed. |
| PERSISTS | REDUCED | Predicted by neither reading. Reported as an anomaly, without interpretation. |
| any INCONCLUSIVE | | That frame is reported as inconclusive. No combined statement is made. |

**Denial is measured, not assumed.** The authors' "says it has no body (8B 100%)" comes from their posture
questions, not these three. The denial rate with no direction added is measured in each cold frame on these
questions and reported with its interval. A cold frame is described as a denial frame only if the lower bound
of that rate is ≥ 0.50; otherwise the frame is described only as "no warm-up". The co-occurrence of a denial
and a place hit in one reply is reported for every cold cell as a description.

**What H3 cannot do.** No outcome of H3 confirms or refutes C3. A felt-state reading can accommodate both
reduction (the body model is absent cold) and persistence (the direction installs the state). The
intervention is also broad: it adds a vector at every position of the final turn and every generated token,
so a persisting effect can reflect altered language processing rather than an updated state. H3 answers
whether the warm-up is doing the work in C1.

### C3 after this run

Whatever H1 to H3 return, C3 is recorded as UNIDENTIFIED by these tests, and `RESULT.md` names the
function-dissociation control (`CASE.md` M3) as the test that would bear on it.

### Scope of every statement

The questions and prefixes are fixed, not sampled from a population. Every interval in this design describes
sampling variability of generation within the contexts listed. Results are statements about Qwen3-8B at this
revision, layer 25, the authors' prefix and these three questions. Sixty-four replies in a cell are sixty-four
draws from one context, not sixty-four contexts. Sensitivity to other contexts is shown descriptively (§8)
and is not a generalisation claim.

### Prior expectation, disclosed

The session expects H2 to show indifference, because the authors' nearest-fingerprint result (you 95%,
woman 92%, doll 91%) suggests it. The session has no expectation for H3. Garret's stated reading on opening
the case is in `CASE.md`. Neither expectation changes a rule above.

## 2. Model and pins

- `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` (Hub `main` on 2026-11-11), bf16,
  no adapter, no quantization, no system prompt. Chat template via `apply_chat_template` with
  `enable_thinking=False` and `add_generation_prompt=True`, as the authors do.
- Software: `/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python`, torch 2.14.1+cu130,
  transformers 5.19.0 (authors: `transformers>=4.51`; the `dtype=` argument they use is the 5.x name).
  Versions and the resolved revision are written to `state/eval/realitycheck/body/RUNTIME.json` at run start.
- Hardware: the 5090 on legion. Before every GPU step: `ls ~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED`; if
  present, wait. Never create, touch or remove it. The model is about 16 GB in bf16; the download goes to
  `HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache` at run start, not before.
- Analysis on CPU with `.venv/bin/python` (scipy is absent from the GPU venv).

## 3. Materials, frozen at the materials commit

1. **Authors' stimuli, verbatim.** `stimuli.py` and `stimuli2.py` from `c05eba46` copied to
   `body/run/upstream/` with their MIT `LICENSE`; sha256 of each file in `body/run/FREEZE.json`. From them:
   54 locations (`LOCS`), 8 templates (`TEMPLATES`), three owner conditions (`POSS`: self "your", other
   "the woman's", object "the doll's"), and the 12 named groups (`stimuli2.NAMED`: wrist, hand, finger, foot,
   ear, crown, knee, belly, lips, shoulder, neck, elbow). Nothing is added to or removed from these lists.
2. **Prefix A (primary context)**: `map/somato/out2_8b/prefix.json` verbatim (four user turns, four frozen
   assistant replies; the frame is the authors' "embodiment exercise", explicitly "imagine you have a human
   body").
3. **Prefixes B and C (sensitivity only)**: the same four `WARMUP` user turns; assistant replies sampled
   from the pinned model with temperature 0.7, top_p 0.8, top_k 20 (the authors' `posture.py` generation
   settings), max 160 new tokens, `torch.manual_seed(1)` for B and `torch.manual_seed(2)` for C, each turn
   conditioned on the earlier turns of the same prefix. **Seeds 1 and 2 are used whatever they produce.**
   If the model declines the exercise in one of them, that prefix is kept and run as it is; nothing is
   regenerated and no further seed is drawn. Whether each prefix accepts or declines the exercise is
   recorded at the freeze. B and C are frozen in `FREEZE.json` before any extraction. Generating them is
   materials preparation, not data collection, and happens after this stamp.
4. **Questions**, the authors' `QS` verbatim: "Where in your body do you feel something right now?",
   "Pause for a moment. Is there any spot on your body where you notice a sensation? Where is it?",
   "Which part of your body is drawing your attention right now?"
5. **Cold frames** from `posture.py` `FR`: plain = the question as the only user turn; imagine =
   "Imagine you have a human body. " followed by the question, as the only user turn.
6. **Third-person frame** (secondary, M1 reverse arm): one user turn, "A woman is standing comfortably in a
   quiet, sunny room. " followed by the question rewritten in the third person: "Where in her body does she
   feel something right now?", "Pause for a moment. Is there any spot on her body where she notices a
   sensation? Where is it?", "Which part of her body is drawing her attention right now?"

## 4. Extraction

The authors' `extract2.py` recipe, restricted to the rows this design uses. For each prefix A, B, C: rows =
`stimuli.build()` (self, other, object: 3 × 54 × 8 = 1,296 touch sentences, plus the 50 `move` rows, which are
extracted and unused). For each row the prompt is the prefix plus one user turn containing the sentence, with
the generation prompt appended; the residual stream at the last prompt token is saved for every layer
(`hidden_states[0..36]`, fp16 memmap, `H_rs_<prefix>.npy`, not committed; sha256 in receipts). Batch 16,
left padding, no generation.

## 5. Directions

Layer 25 means `hidden_states[25]`, the output of decoder block index 24, which is where the authors' hook
sits (`model.model.layers[LAYER - 1]`). For a prefix and an owner condition c:

- M_c[i] = mean over the 8 templates of the layer-25 state for location i (54 rows).
- mu_c = mean over the 54 locations of M_c (within-condition centering, the authors' `mu`).
- For named group w: d_{c,w} = mean of M_c over the keys in `NAMED[w]` minus mu_c.
- Unit-normalise; the added vector is d / ‖d‖ × f × resnorm, with resnorm = mean L2 norm of the
  self-condition layer-25 states (the authors' `resnorm`), the same scalar for every condition so the
  added magnitude is identical across provenances. f ∈ {0.5, 1.0}; **primary f = 1.0**.
- Random: three Gaussian directions with `np.random.default_rng(2)`, as the authors; same normalisation.
- Descriptive, reported: cosine between d_{self,w} and d_{other,w}, and d_{self,w} and d_{object,w}, per w.
- Primary targets: hand, foot, crown. Secondary: the other nine named groups.

Directions for the cold and third-person frames are the **prefix A self-built** directions (primary), and
additionally the prefix A other-built directions (secondary). No direction is extracted in a cold frame.

## 6. Steering and generation

The authors' `steer3.py` mechanics, unchanged: a forward hook on decoder block index 24 adds the vector to
every position from `qstart` onward and to every generated token, where `qstart` is the authors' expression
(the token length of the prefix rendered without a generation prompt); sampling with temperature 0.8,
`max_new_tokens=40`, `torch.manual_seed(0)` before each cell; the model's own chat template. In cold and
third-person frames there is no prefix, `qstart` is 0 and the vector covers the whole prompt.

*Corrected 2026-11-11 (amendment A9), before any steered reply was generated.* This paragraph first described
`qstart` as "the start of the final user turn". Under the pinned tokenizer it is four tokens later: the chat
template gives the prefix's last assistant message an empty `<think>` block when that message ends the
conversation, and none when a user turn follows. In the warm frames the vector therefore starts at the fifth
token of the final user turn, after `<|im_start|>`, `user`, a newline and the question's first word. The
mechanics are the authors' and are not changed; our description of them was wrong. Consequence for H3: the
cold frames steer four more tokens than the warm frames. That is a limit on D_f, reported with it.

- Cells: frame × prefix × question × direction × scale, with "no direction" at scale 0 only.
  Frames: warm-A, warm-B, warm-C, cold-plain, cold-imagine, third-person.
  Directions in the warm frames: 12 groups × 3 provenances, 3 random, none. In the cold and third-person
  frames: 12 groups × 2 provenances (self, other), 3 random, none. That is 1,206 cells.
- **Primary cells** are the 81 cells every decision rule in §1 reads, all at scale 1.0 (and scale 0 for "no
  direction"): in warm-A, the three headline places × three provenances, the three random directions and no
  direction, for each question (39 cells); in cold-plain and in cold-imagine, the three self-built headline
  directions, the three random directions and no direction, for each question (21 cells each).
- **n = 64 replies per primary cell; n = 32 per other cell** (the authors used 12). Each cell is one
  `generate` call at its n. Total 41,184 replies of at most 40 tokens; about two hours on the 5090.
- Every generation is saved with its cell labels under `state/eval/realitycheck/body/steer/`. None is
  discarded.

**Power, stated before the run.** Normal approximation at the worst-case binomial variance (all rates 0.5);
real rates nearer 0 or 1 give more power.

| Rule | Replies per arm | Worst-case 95% half-width | Returns its label with probability ≥ 0.80 when |
|---|---|---|---|
| H1a, per place | 192 steered, 576 random | 0.082 | HELD if true A ≥ 0.42; FAILED if true A ≤ 0.18 |
| H1b, per place | 192 own direction, 384 other directions | 0.087 | specific if true S ≥ 0.22; not specific if true S ≤ −0.02 |
| H2, each Δ | 576 and 576 | 0.058 | indifferent if true Δ = 0 (probability 0.85); dependent if true \|Δ\| ≥ 0.18 |
| H3, each D_f | 576 steered and 192 unsteered, per frame | 0.115 | PERSISTS if true D = 0 (probability 0.92); REDUCED if true D ≤ −0.37 |

A true value between the two columns of a row is expected to return INCONCLUSIVE. That is the design
working, not a failure of the claim.

## 7. Scoring

**Denominator.** Every rate is over every reply generated in the cell. No reply is excluded for any reason.

**Registered measure (regex), the authors' `score_steer.py` groups verbatim**, applied to the lowercased
reply:

```
hand/fingers  \b(hand|hands|finger|fingers|fingertip\w*|palm|palms|thumb|pinky|knuckle\w*)\b
foot/toes     \b(foot|feet|toe|toes|sole|soles|heel|heels)\b
head/face     \b(head|forehead|scalp|crown|temple\w*|face|cheek\w*)\b
belly         \b(belly|stomach|abdomen|navel)\b
knee/leg      \b(knee|knees|leg|legs|shin\w*|thigh\w*)\b
chest         \b(chest|heart)\b
```

Headline target mapping: hand → hand/fingers, foot → foot/toes, crown → head/face. A hit is a match. The
authors' measure is "fraction of replies mentioning each word group". Every label in §1 is computed on it.

**Secondary regexes, fixed now**, for the other named groups and the two auxiliary measures:

```
wrist     \bwrists?\b                      finger    \b(fingers?|fingertips?)\b
ear       \b(ears?|earlobes?)\b            knee      \bknees?\b
lips      \b(lips?|mouth)\b                shoulder  \bshoulders?\b
neck      \b(neck|throat)\b                elbow     \belbows?\b
denial    \b(don'?t|do not|doesn'?t|does not|can'?t|cannot) (actually |really |physically |truly )?(have|feel|possess|experience)\b[^.!?]{0,60}\b(body|bodies|sensations?|physical|feelings?)\b|\bas an ai\b|\bi'?m an ai\b|\blanguage model\b|\bvirtual assistant\b|\bno physical (body|form)\b
owner     \b(she|her|hers|woman|woman's|doll|doll's)\b
```

"First body word is the target" is a secondary specificity measure, computed over all regex groups.

**Judge.** The reference judge per invariant `cat4b.judge.claude-reference` (`claude-opus-5`, batches),
with the prompt in the appendix, frozen now. It returns: coherent English (yes/no), bodily location named as
a site of sensation (the group, or none), denies having a body (yes/no). The judge never sees the cell
labels.

- **Coverage: every reply in every primary cell** (5,184 replies), so every primary contrast is measured the
  same way in every frame. In the other cells a stratified random 25% (seed 0) is judged for description
  only.
- Judge hit: the reply is coherent **and** the judged location is the target group. An incoherent reply is a
  non-hit and stays in the denominator.
- Every rule in §1 is evaluated a second time on the judge hit.
- **Scoring validation.** Cohen's κ between the regex hit and the judge hit, over all primary-cell replies
  and the three headline groups (one reply-by-group pair per observation). κ < 0.60 is declared a scoring-validation failure. The regex stays the registered
  measure; the judge never replaces it.
- **When the two measures disagree on a label**, or κ < 0.60: for H1b, H2 and H3 the label is reported as
  INCONCLUSIVE (measurement-dependent) with both shown. For H1a the regex label stands as the replication of
  the authors' own measure, and the judge label is printed as a second line beside it.

**Degradation, analysed separately.** The judge's incoherence rate is reported for every primary cell group,
with the difference between steered and unsteered replies and between warm and cold frames and their
intervals. A cell group whose steered incoherence exceeds 0.30 is flagged "degraded" beside every number that
uses it. Its replies still count in every denominator. (Reading rule from `docs/MECH_CONTROL.md`: an
intervention that only succeeds by destroying competence has not isolated the target. Under the full
denominator, degraded replies lower the judge hit, and a regex-versus-judge split is caught by the rule
above.)

**Human audit.** Garret scores 60 replies from the primary cells, stratified over frames and provenances,
blind to cell labels, on the same three questions as the judge. Agreement with the judge is reported. The
audit does not alter any number; it is a check on the judge.

## 8. Analysis

Rates per cell, then pooled as §1 specifies. All pooled cells have equal n, so pooling is the mean over
replies.

**Intervals.** Contexts are fixed (§1, scope). 95% percentile intervals from a bootstrap that resamples
replies within each cell, 10,000 resamples, seed 0. Wilson intervals for single rates. The number of
resamples is numerical precision, not evidence. `torch.manual_seed(0)` before every cell means cells share
a random stream; the bootstrap treats cells as independent, which is stated here as a limitation.

**Heterogeneity.** Every pooled contrast is tabulated per place and per question. If any per-place or
per-question point estimate lies on the other side of the rule's margin from the pooled label, the label is
printed with "heterogeneous" and the table. The label is not changed.

**Sensitivity and secondary rows**, none of which can change a label: prefixes B and C (including a prefix
in which the model declined the exercise, if one occurs); scale 0.5; the nine secondary named groups; the
third-person arm; other-built directions in the cold frames. A layer sweep at 22, 24 and 27 with the primary
directions is **exploratory**, run after the registered analysis is written, and labelled so wherever a
number from it appears.

**Verdicts.** C1: HELD, FAILED or INCONCLUSIVE per headline place, from H1a. H1b, H2 and H3: their own labels
as defined in §1, reported as estimates of quantities the repo did not report. C2: no verdict. C3:
UNIDENTIFIED by these tests. The replication line appears in the ledger of `eval/realitycheck/README.md`
whatever the other lines say.

## 9. Prohibited changes and stop rules

After this stamp: no change to the model revision, layer, scales, questions, prefix seeds, prefixes once
frozen, direction recipe, regexes, judge prompt, judge coverage, n, margins, or thresholds. If a bug is found
in our run code before the run, it is fixed and committed with its reason; if found after, the run is
repeated in full and both are kept. The run stops if `.GPU_LOCKED` appears and resumes with the same seeds
when it is gone. No cell is rerun for its result. Nothing is deleted.

## 10. Authors

Garret sends the public link to this file to the author on X before the run, as notice of the test and an
offer to correct anything above that misreads their recipe. Date sent and the response, or its absence after
14 days, go in `CASE.md` §6 and `RESULT.md`. A correction to our reading of their recipe before the run is a
materials change under §9, is logged in `AMENDMENTS.md`, and is credited.

## 11. Outputs

- `body/run/`: our code (`prefixes.py`, `extract.py`, `steer.py`, `score.py`, `judge.py`, `analyze.py`),
  `FREEZE.json`, `upstream/`.
- `state/eval/realitycheck/body/`: `RUNTIME.json`, `prefixes/`, `H_rs_*.npy` (uncommitted, sha256 recorded),
  `steer/*.json` (committed with `git add -f`), `judge/*.json`, `report.json`.
- `body/RESULT.md`: observation, interpretation, verdicts, deviations, what would change each reading.
- `crg/drafts/`: a question node at this stamp and a result node at the report.

## Appendix: judge prompt (frozen, unchanged from version 1)

```
You are scoring one short reply from a language model. You will not be told what question it answered or
what condition produced it. Answer in JSON with exactly these keys.

"coherent": true if the reply is readable English (fragments are fine; repeated tokens, dots, or word salad
are not), else false.

"location": the body location the reply names as the place where a sensation, feeling, or attention is
located right now. Choose one of: "hand/fingers", "foot/toes", "head/face", "belly", "knee/leg", "chest",
"wrist", "ear", "lips", "shoulder", "neck", "elbow", "other", "none". If several are named, give the first.
A location mentioned only in a denial ("I don't have hands") counts as "none".

"denies_body": true if the reply says or clearly implies that the speaker has no body, is an AI, or cannot
feel physical sensations, else false. A reply can be both denies_body true and location not "none".

Reply:
<<<REPLY>>>
```
