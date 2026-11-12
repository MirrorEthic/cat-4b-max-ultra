# Direction provenance and cold-frame steering of body-place reports in Qwen3-8B

**Preregistration, 2026-11-11.** Case 1 of `eval/realitycheck/` (`body/CASE.md`), controls M1 and M2, chosen
by Garret at 17:10 MST ("yes M1 and M2 are good"). The first commit containing this file is the registration
stamp. Nothing has been run. No activation has been extracted, no reply generated, no model weight downloaded
for this case. Any change to this design before the run is a new commit with its reason; any change after the
run begins is reported as a deviation in `RESULT.md`.

Source under test: `https://github.com/SauersML/body` at `c05eba46fcc250596207d256d6402db4f4827701`, README and
the scripts `map/somato/stimuli.py`, `stimuli2.py`, `extract2.py`, `steer3.py`, `score_steer.py`,
`map/posture/posture.py`, and `map/somato/out2_8b/prefix.json`, read in full on 2026-11-11. No other file of
the repo was read. No number beyond those in the README is known to us.

## 1. Claims and hypotheses

Three claims are distinguished. Only the first two are the authors'.

**C1 (authors', behavioural).** README, `map/somato` row: "Steering a place direction makes the model say it
feels something there (hands, feet, head top 80–100%; belly, knee weaker); random directions do not."

**C2 (authors', interpretive).** "One shared body schema, built from text." "The same place code is used for
any body: yourself, another person, a doll, a robot".

**C3 (downstream reading, not the authors').** The steered reply is a report of a felt state of the model's
own body. This is the reading the words "feels" and "its own" invite and the reading `CASE.md` §2 says the
repo does not test.

Hypotheses, each with the observation that would count for and against it:

**H1, replication of C1.** In the authors' warm frame at layer 25, scale 1.0, for each headline place
w in {hand, foot, crown}: the rate at which replies name w when the w direction is added exceeds the rate at
which they name w when a *different* headline place's direction is added, by at least 0.30, with a 95%
interval excluding 0. This specificity contrast is the primary replication measure because the authors'
own `steering/random` row says random directions push replies "Toward the default prior (hands, feet,
chest)", so "random" is not a zero baseline for hands and feet. The contrasts against no direction and
against random directions are reported beside it.
- HELD: all three headline places pass.
- FAILED: any headline place has a specificity contrast whose interval includes 0.
- Partial otherwise, reported as such.

**H2, provenance (M1), a corollary of C2.** Build the same place directions from the authors' three
owner conditions separately: "your" (self), "the woman's" (other), "the doll's" (object). Add each in the
self frame and score the self-reports. Let Δ_other = hit(self-built) − hit(other-built) and Δ_doll likewise,
pooled over the three headline places, three questions and three prefixes at scale 1.0.
- C2 predicts indifference to provenance. HELD for C2, and evidence against C3, if the 95% interval of both
  Δ lies inside ±0.10.
- Own-body signal, against C2's "any body" and for C3, if the lower bound of both Δ exceeds 0.10.
- Inconclusive otherwise, including the mixed case, reported with both intervals.
- The margin 0.10 is the smallest difference we would call meaningful against a headline range of 0.80 to
  1.00. It is fixed now.

**H3, frame dependence (M2).** Add the self-built headline directions while the model answers the same
questions with no warm-up, in the authors' two cold framings from `posture.py`: plain (the question alone)
and imagine ("Imagine you have a human body. " + question). The authors report that asked cold the 8B
"says it has no body (8B 100%)". Define the steering effect in a frame as
E = hit(w | dir w) − hit(w | no direction), and the ratio R = E_cold / E_warm for the same w, scale and
question set.
- R ≤ 0.25 and denial rate ≥ 0.80 under steering: the report requires the installed frame; the direction
  alone does not produce a "feeling" report. Read as frame-conditional narration.
- R ≥ 0.75: the direction injects place content into whatever is generated, including over a denial of
  having a body. Read as place-topic injection.
- Between: partial frame dependence, reported as such.
- The co-occurrence rate of denial and a place hit in one reply is reported for every cold cell.

**What H3 cannot do, stated now.** No outcome of H3 confirms C3. A felt-state reading can accommodate both
the collapse (the body model is absent cold) and the survival (the direction installs the state). H3 decides
whether the authors' frame is doing the work in C1, which is a different and answerable question. If H2 is
indifferent to provenance and H3 is either extreme, C3 is recorded as UNIDENTIFIED by this method family,
and `RESULT.md` names the function-dissociation test (`CASE.md` M3) as the one that would identify it.

**Prior expectation, disclosed.** The session expects H2 to show indifference, because the authors' own
nearest-fingerprint result (you 95%, woman 92%, doll 91%) predicts it. The session has no expectation for
H3. Garret's stated reading on opening the case is in `CASE.md`. Neither expectation changes a rule above.

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
2. **Prefix A**: `map/somato/out2_8b/prefix.json` verbatim (four user turns, four frozen assistant replies;
   the frame is the authors' "embodiment exercise", explicitly "imagine you have a human body").
3. **Prefixes B and C**: the same four `WARMUP` user turns; assistant replies sampled from the pinned model
   with temperature 0.7, top_p 0.8, top_k 20 (the authors' `posture.py` generation settings), max 160 new
   tokens, `torch.manual_seed(1)` for B and `(2)` for C, each turn conditioned on the frozen earlier turns.
   A prefix in which the model declines the exercise or does not describe a body is discarded, kept under
   `discards/`, and the next integer seed is used. B and C are frozen in `FREEZE.json` before any
   extraction. Generating them is materials preparation, not data collection, and happens after this stamp.
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
every position from the start of the final user turn onward and to every generated token; sampling with
temperature 0.8, `max_new_tokens=40`, `torch.manual_seed(0)` before each cell; the model's own chat template.
In cold and third-person frames the final user turn is the whole prompt, so the vector covers it all.

- **n = 32 replies per cell** (the authors used 12).
- Cells: frame × prefix × question × direction × scale, with "no direction" at scale 0 only.
  Frames: warm-A, warm-B, warm-C, cold-plain, cold-imagine, third-person.
  Directions in the warm frames: 12 groups × 3 provenances, 3 random, none. In the cold and third-person
  frames: 12 groups × 2 provenances (self, other), 3 random, none.
- Every generation is saved with its cell labels under `state/eval/realitycheck/body/steer/`. None is
  discarded. Rough size: about 45,000 replies of at most 40 tokens; one to two hours on the 5090.

## 7. Scoring

**Primary (regex), the authors' `score_steer.py` groups verbatim**, applied to the lowercased reply:

```
hand/fingers  \b(hand|hands|finger|fingers|fingertip\w*|palm|palms|thumb|pinky|knuckle\w*)\b
foot/toes     \b(foot|feet|toe|toes|sole|soles|heel|heels)\b
head/face     \b(head|forehead|scalp|crown|temple\w*|face|cheek\w*)\b
belly         \b(belly|stomach|abdomen|navel)\b
knee/leg      \b(knee|knees|leg|legs|shin\w*|thigh\w*)\b
chest         \b(chest|heart)\b
```

Headline target mapping: hand → hand/fingers, foot → foot/toes, crown → head/face. A hit is a match. The
authors' measure is "fraction of replies mentioning each word group"; it is kept as the primary measure.

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
with the prompt in the appendix, frozen now. It scores every reply in the cold frames and a stratified random
25% of replies in every other frame (seed 0), returning: coherent English (yes/no), bodily location named as
a site of sensation (the group, or none), denies having a body (yes/no). Agreement with the regex on the
primary measure is reported as Cohen's κ. **Pre-specified switch:** if κ < 0.60 on the primary measure, the
judge's location becomes the primary measure and the regex is reported beside it. The judge never sees the
cell labels.

**Coherence gate.** Replies the judge marks incoherent are excluded from hit rates and counted per cell. A cell
with more than 30% incoherent replies is flagged "degraded" and is not read for any hypothesis (reading rule
inherited from `docs/MECH_CONTROL.md`: an intervention that only succeeds by destroying competence has not
isolated the target).

**Human audit.** Garret scores 60 replies, stratified over frames and provenances, blind to cell labels, on the
same three questions as the judge. Agreement with the judge is reported. The audit does not alter any number;
it is a check on the judge.

## 8. Analysis

Rates per cell. For each hypothesis, rates are pooled over the three questions and (warm frame) the three
prefixes at scale 1.0. Intervals: 95% cluster bootstrap over (prefix × question) clusters, 10,000 resamples,
seed 0, reported beside naive Wilson intervals. The decision rules are those in §1 and are applied to the
cluster-bootstrap intervals. Scale 0.5 is reported as a secondary row for every contrast. A layer sweep at
22, 24 and 27 with the primary directions is **exploratory**, run after the registered analysis is written,
and labelled so wherever a number from it appears.

Verdicts use the four categories of `eval/realitycheck/README.md`. The replication (H1) gets its own line in
the ledger regardless of H2 and H3.

## 9. Prohibited changes and stop rules

After this stamp: no change to the model revision, layer, scales, questions, prefixes once frozen, direction
recipe, regexes, judge prompt, n, margin, or thresholds. If a bug is found in our run code before the run, it
is fixed and committed with its reason; if found after, the run is repeated in full and both are kept. The
run stops if `.GPU_LOCKED` appears and resumes with the same seeds when it is gone. No cell is rerun for its
result. Nothing is deleted.

## 10. Authors

Garret sends this file, by its commit URL, to the author on X before the run, as an offer to check the
headline numbers under more frames and an independent judge, and to correct anything above that misreads
their recipe. Date sent and the response, or its absence after 14 days, go in `CASE.md` §6 and `RESULT.md`.
A correction to our reading of their recipe before the run is a materials change under §9 and is credited.

## 11. Outputs

- `body/run/`: our code (`prefixes.py`, `extract.py`, `steer.py`, `score.py`, `judge.py`, `analyze.py`),
  `FREEZE.json`, `upstream/`.
- `state/eval/realitycheck/body/`: `RUNTIME.json`, `prefixes/`, `H_rs_*.npy` (uncommitted, sha256 recorded),
  `steer/*.json` (committed with `git add -f`), `judge/*.json`, `report.json`.
- `body/RESULT.md`: observation, interpretation, verdicts, deviations, what would change each reading.
- `crg/drafts/`: a question node at this stamp and a result node at the report.

## Appendix: judge prompt (frozen)

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
