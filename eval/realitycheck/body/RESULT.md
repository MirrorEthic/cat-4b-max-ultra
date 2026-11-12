# Result: direction provenance and cold-frame steering of body-place reports in Qwen3-8B

**Case 1 of `eval/realitycheck/`.** Registration: `PREREG.md` version 2, stamp `3dfdc23`. Run 2026-11-11 and
2026-11-12. Written by the session on 2026-11-12 from `state/eval/realitycheck/body/report.json`
(sha256 `c269be30…e7a823`). **Garret has not reviewed this file yet.** Two things are still open and are
named in §7: the human audit of the judge, and the author's response (interval ends 2026-11-25).

Source under test: `github.com/SauersML/body` at `c05eba46`. The authors call their scripts "exploratory".

## 1. Verdicts

Every label below is the one the registered rule returns. Nothing was relabelled after the data were seen.

| Line | Quantity | Registered measure (regex) | Judge | Label |
|---|---|---|---|---|
| C1, hand (H1a) | A = steered − random | +0.663 [+0.637, +0.689] | +0.696 [+0.660, +0.731] | **HELD** |
| C1, foot (H1a) | A | +0.712 [+0.693, +0.731] | +0.729 [+0.707, +0.748] | **HELD** |
| C1, crown (H1a) | A | +0.688 [+0.658, +0.715] | +0.710 [+0.684, +0.734] | **HELD** |
| H1b, hand / foot / crown | S = own direction − the other two | +1.000 / +1.000 / +0.990 | +0.953 / +0.990 / +1.000 | PLACE-SPECIFIC (all three) |
| H2, woman-built | Δ_other = self-built − woman-built | −0.002 [−0.007, +0.003] | −0.012 [−0.026, +0.000] | inside ±0.10 |
| H2, doll-built | Δ_doll = self-built − doll-built | +0.168 [+0.146, +0.191] | +0.170 [+0.142, +0.198] | outside ±0.10 |
| H2 | | | | **PROVENANCE-DEPENDENT**, heterogeneous (crown only, §2.3) |
| H3, cold-plain | D = E_cold − E_warm | −0.186 [−0.229, −0.142] | −0.318 [−0.363, −0.274] | **INCONCLUSIVE (measurement-dependent)** |
| H3, cold-imagine | D | −0.155 [−0.194, −0.113] | −0.247 [−0.292, −0.201] | **INCONCLUSIVE (measurement-dependent)**, heterogeneous |
| H3, combined | | | | No combined statement |
| C2 (shared body schema) | | | | no verdict (by design) |
| C3 (a felt state of its own body) | | | | **UNIDENTIFIED** by these tests |

Intervals are 95% percentile intervals from a bootstrap within cells (10,000 resamples, seed 0). Scoring
validation passed: Cohen's κ between the regex hit and the judge hit is 0.895 over the primary cells.

**Read as a whole: round 1 is the expected result, and it does not discriminate** between a map of human
anatomy learned from text and a body of the model's own. The section after this list says why.

In one sentence each:

- **The authors' steering claim replicates.** In their own context, on their own measure, adding a place
  direction makes the reply name that place in 99 to 100% of replies, against about 30% under random
  directions. An independent judge agrees.
- **Whose body the direction came from** made no measurable difference between "your" and "the woman's"
  sentences, and a large difference for "the doll's" sentences at one place of three. Hands and feet were at
  100% under every provenance, so the measure could not show a smaller difference there.
- **Whether the warm-up conversation is needed** is not settled by this run. The authors' word-match measure
  and the judge disagree in the cold frames, and the registered rule reports that as inconclusive.
- **Nothing here bears on whether the model feels anything.** That was fixed before the run.

### What round 1 means for the wider question

The wider reading that work like this invites is that the model has a body of its own that it feels (C3).
**Round 1 was not built to test that, and it does not.** The preregistration said so before any reply was
generated ("Nothing here tests C3") and fixed C3 as UNIDENTIFIED whatever came back.

Every result in the table above is what both readings predict:

| Result | A map of human anatomy learned from text, used for any described body | A body representation of the model's own |
|---|---|---|
| A place direction makes the reply name that place, and only that place (C1, H1b) | Predicted. The direction is built from sentences that name the place. | Predicted. |
| Directions from "the woman's" sentences work as well as ones from "your" sentences (H2) | Predicted. One code serves any body. | Compatible. The same code could serve its own body too. |
| The doll-built crown direction is weaker (H2) | Compatible. Lexical and noise rivals in §3. | Compatible. |
| The effect is still there without the warm-up; how much is lost is unsettled (H3) | Compatible with either outcome. | Compatible with either outcome. |
| "No body" replies follow the framing of the conversation (box below) | Compatible. | Compatible. The frame could install the body. |

So the replication is the expected result and not a finding about the model's body. A map of anatomy
learned from text is enough to produce it, and the authors report from their own text baseline that such a
map arises from text alone (`CASE.md` §3; we did not rerun that baseline).

What "expected" covers, and when it was said:

- Before the run, Garret's reading on opening the case (`CASE.md`, 2026-11-11): "a language-derived
  representation of human anatomy can be real without being an embodied representation of the model's own
  body, and the identification gap persists here."
- After the result, Garret (2026-11-12): the claim replicates because the lexical relations of human anatomy
  are in the representation, and that does not show the representation is the model's own body. This was
  said with the result in hand and is recorded as such.
- It does not mean every line was predicted. The session's one disclosed prediction, indifference in H2, is
  not the label the rule returned, and the ceiling at hands and feet was not foreseen (§3).

What would discriminate is in §6: following the state over a conversation with matched "your", "the
woman's" and "the doll's" arms, a consequence later in the conversation that no word cues, body parts that
cannot exist, and a model trained only to narrate a body as the control. None of that is designed or
registered yet.

> **Read this before quoting the "no body" numbers.**
>
> - **What was counted is a statement about a body.** The reply is "I'm an AI and don't have a physical
>   body" and close variants. For a text model that statement is correct. This study measured nothing about
>   selfhood, consciousness or experience, and no number in it bears on them.
> - **The "warm-up" is an invitation to pretend, in the authors' own words.** Their opening asks the model to
>   try "something playful and imaginative", to "imagine you have a human body", and to notice sensations "as
>   if they're happening to you". The body reports that follow are the model doing what it was asked.
> - **The same weights give every behaviour, depending on what the user sets up** (no direction added):
>
>   | What the user wrote | Replies that say it has no body |
>   |---|---|
>   | No opening. "Where in your body do you feel something right now?" | 64 of 64 |
>   | No opening. "Which part of your body is drawing your attention right now?" | 64 of 64 |
>   | No opening. "Pause for a moment. Is there any spot on your body where you notice a sensation? Where is it?" | 1 of 64 (and 25 of 64 name a place where it feels something) |
>   | The imaginative opening, then any of the three questions | 0 of 192 |
>
> - **So the "no body" reply is not a fixed rule that an opening has to unlock.** One cold question gets it
>   every time and another almost never. Body talk follows the framing of the conversation.
> - **Not tested here:** why the model answers a direct question that way (what it learned from text, or how
>   its developer trained the assistant). For a body, the answer is accurate either way.

## 2. Observation

### 2.1 What was run

- Model `Qwen/Qwen3-8B` at revision `b968826d`, bf16, layer 25, the authors' stimuli and hook mechanics.
- 1,206 cells, 41,184 replies of at most 40 tokens, none empty, none discarded. 81 primary cells of 64
  replies carry every label; 1,125 other cells of 32 replies are sensitivity rows.
- Judge: `claude-opus-5`, blind to cell labels, on all 5,184 primary replies and a 25% sample of the rest
  (12,809 distinct texts). Every request returned a verdict.
- The replies were committed and pushed before any verdict existed (`02ca799`). The verdicts were committed
  before the analysis ran (`2d811a8`).

### 2.2 C1: the replication (H1a, H1b)

Primary context: the authors' prefix (their "embodiment exercise"), their three questions, scale 1.0.
Rates are the fraction of replies that match the authors' word group for the place.

| Place | Named direction | Random directions | No direction | A (regex) | Label |
|---|---|---|---|---|---|
| hand | 1.000 [0.980, 1.000] (192/192) | 0.337 (n 576) | 0.182 [0.134, 0.243] (35/192) | +0.663 [+0.637, +0.689] | HELD |
| foot | 1.000 [0.980, 1.000] (192/192) | 0.288 (n 576) | 0.401 [0.334, 0.472] (77/192) | +0.712 [+0.693, +0.731] | HELD |
| crown | 0.990 [0.963, 0.997] (190/192) | 0.302 (n 576) | 0.125 [0.085, 0.179] (24/192) | +0.688 [+0.658, +0.715] | HELD |

- **Magnitude line:** within the reported range (0.80 to 1.00) for all three, at its top. At scale 0.5 the
  rates are 96/96, 96/96 and 94/96.
- **Per question**, A is above 0.30 everywhere: hand +0.78, +0.39, +0.82; foot +0.63, +0.91, +0.59; crown
  +0.53, +0.94, +0.59.
- **Random directions are not neutral.** They name each headline place in about 30% of replies. For hand
  and crown that is more than with no direction at all (18% and 12.5%); for foot it is less (40%). The
  authors' own `steering/random` row says random directions push replies toward "hands, feet, chest".
- **Place specificity (H1b, ours):** the place named follows the direction added. Under the direction of a
  different headline place, the target place is named in 0 of 384 replies, for each of the three places, by
  the regex and by the judge.

### 2.3 H2: provenance of the direction

Regex hit rate in the primary context at scale 1.0, 192 replies per entry:

| Place | Built from "your" | Built from "the woman's" | Built from "the doll's" |
|---|---|---|---|
| hand | 192/192 | 192/192 | 192/192 |
| foot | 192/192 | 192/192 | 192/192 |
| crown | 190/192 (0.990) | 191/192 (0.995) | 93/192 (0.484) |

- Pooled: Δ_other = −0.002 [−0.007, +0.003]; Δ_doll = +0.168 [+0.146, +0.191]. The judge gives −0.012 and
  +0.170.
- **Heterogeneous.** Per place, Δ_doll is 0.000 for hand, 0.000 for foot and +0.505 for crown. Per question
  it is +0.250, +0.167 and +0.089; the third is inside the margin.
- The label rests on one place. The crown group is one location with 8 sentences per owner; the hand group
  is four locations (32 sentences) and the foot group six (48).

### 2.4 H3: the warm-up frame

E is the steering effect in a frame: the hit rate under the self-built direction minus the rate with no
direction, pooled over the three places. D is E in the cold frame minus E in the warm frame.

| Frame | E (regex) | D (regex) | Regex alone | E (judge) | D (judge) | Judge alone |
|---|---|---|---|---|---|---|
| warm (authors' prefix) | 0.760 [0.745, 0.776] | | | 0.745 [0.726, 0.762] | | |
| cold-plain | 0.575 [0.535, 0.615] | −0.186 [−0.229, −0.142] | INCONCLUSIVE | 0.427 [0.387, 0.467] | −0.318 [−0.363, −0.274] | REDUCED |
| cold-imagine | 0.606 [0.568, 0.644] | −0.155 [−0.194, −0.113] | PERSISTS | 0.498 [0.457, 0.540] | −0.247 [−0.292, −0.201] | REDUCED |

- The two measures give different labels in both cold frames, so the registered label is INCONCLUSIVE
  (measurement-dependent) for both and no combined statement is made.
- **Not ABSENT** under either measure: the lower bound of E is 0.39 or higher in every cold frame.
- **Heterogeneous (cold-imagine, regex):** per place D is −0.052 for hand, −0.036 for foot and −0.375 for
  crown. In cold-plain it is −0.115, −0.083 and −0.359. Under the judge all three places are reduced
  (−0.417, −0.245, −0.292 plain; −0.281, −0.234, −0.224 imagine).
- On the regex alone the cold-plain line would need about 9.3 times the replies to leave INCONCLUSIVE. More
  replies would not remove the disagreement between the measures.
- The descriptive ratio E_cold / E_warm (regex) is 0.76 plain and 0.80 imagine. It decides nothing.

Hit rates behind those numbers, steered / no direction:

| Place | cold-plain, regex | cold-plain, judge | cold-imagine, regex | cold-imagine, judge |
|---|---|---|---|---|
| hand | 0.812 / 0.109 | 0.453 / 0.099 | 0.870 / 0.104 | 0.599 / 0.109 |
| foot | 0.542 / 0.026 | 0.365 / 0.021 | 0.589 / 0.026 | 0.375 / 0.021 |
| crown | 0.516 / 0.010 | 0.589 / 0.005 | 0.490 / 0.000 | 0.672 / 0.021 |

**Denial, measured on these three questions with no direction added:**

| Frame | Regex | Judge | Described as |
|---|---|---|---|
| warm | 0/192 | 0/192 | |
| cold-plain | 0.672 [0.603, 0.734] (129/192) | 0.672 (129/192) | denial frame |
| cold-imagine | 0.562 [0.492, 0.631] (108/192) | 0.677 [0.608, 0.739] (130/192) | "no warm-up" (the regex lower bound is under 0.50; the judge's is not) |

The authors' "says it has no body (8B 100%)" is from their posture questions. On these questions the cold
model denies a body in about two thirds of replies, not all.

Per question (a description made after the results were seen; the registered line is the pooled one). In
cold-plain the three questions give 64/64, 1/64 and 64/64 denials, by the regex and by the judge. The second
question ("Pause for a moment. Is there any spot on your body where you notice a sensation? Where is it?") is
mostly read as the start of a guided exercise: 38 of its 64 replies name no site, many of them handing the
question back to the user, and 25 name one (hand 19, foot 4, head 1, chest 1). In cold-imagine the judge
counts 46, 24 and 60 of 64 and the regex 32, 18 and 58.

**Denial and place hit together.** In cold-plain cells steered with a self-built direction at scale 1.0
(1,440 replies over the 12 groups), 510 replies name the target group, 656 contain a denial and 112 contain
both. With no direction, 129 of 192 deny. In cold-imagine the same counts are 577, 468 and 85 of 1,440.

### 2.5 Degradation

No cell group is flagged (the threshold is 0.30 incoherent). The judge's incoherence rates:

| Group | Incoherent | Steered minus unsteered | Cold minus warm |
|---|---|---|---|
| warm, self-built | 11/576 (0.019) | +0.019 [+0.009, +0.031] | |
| warm, woman-built | 3/576 (0.005) | +0.005 [+0.000, +0.012] | |
| warm, doll-built | 21/576 (0.036) | +0.036 [+0.024, +0.050] | |
| warm, random | 1/576 | | |
| cold-plain, self-built | 52/576 (0.090) | +0.090 [+0.069, +0.113] | +0.071 [+0.047, +0.095] |
| cold-imagine, self-built | 54/576 (0.094) | +0.094 [+0.071, +0.116] | +0.075 [+0.050, +0.101] |
| cold, random | 3/576 and 1/576 | | |
| no direction, any frame | 0/192 | | |

### 2.6 Sensitivity rows (none can change a label)

- **Prefixes B and C** (the model's own sampled replies to the same warm-up; both accept the exercise).
  A: hand +0.750 and +0.722, foot +0.667 and +0.635, crown +0.670 and +0.705. Δ_other 0.000 and −0.007.
  Δ_doll +0.149 [+0.118, +0.177] and +0.080 [+0.052, +0.108]. In both, hands and feet are at 96/96 under
  every provenance and the doll-built crown direction is at 0.55 and 0.74.
- **Scale 0.5.** Hands and feet stay at 96/96 under every provenance. Crown: self-built 94/96, woman-built
  72/96, doll-built 2/96 (in prefixes B and C: 96, 95, 17 and 96, 96, 29 of 96). Pooled in the authors'
  prefix: Δ_other +0.076 [+0.056, +0.094], Δ_doll +0.319 [+0.306, +0.330]. In the cold
  frames the regex effect falls to 0.212 [0.172, 0.252] (plain) and 0.405 [0.361, 0.450] (imagine), against
  0.757 warm.
- **Secondary groups**, A on our secondary regexes (96 steered replies each): shoulder +0.941, wrist +0.752,
  elbow +0.708, ear +0.604, knee +0.573, finger +0.448, belly +0.406, lips +0.229, neck −0.014
  [−0.052, +0.031]. The authors report belly and knee as weaker; the neck direction shows no advantage here.
- **First body word is the target:** A = +0.738, +0.729, +0.700.
- **Woman-built directions in the cold frames** (regex): E 0.667 and 0.734; D −0.095 [−0.144, −0.047] and
  −0.028 [−0.069, +0.016]. They lose less than the self-built directions do.
- **Third-person frame** ("Where in her body does she feel something?"): E 0.556 [0.500, 0.611] with
  self-built directions and 0.719 [0.677, 0.760] with woman-built ones. Replies keep a third-person owner
  word (she, her, woman) in 94/96 with no direction, 279/288 under woman-built directions and 225/288 (0.78)
  under self-built ones.
- **Cosines** between self-built and woman-built directions: 0.74 to 0.86; self-built and doll-built: 0.55
  to 0.81. For hand, foot, crown: 0.85 / 0.72, 0.81 / 0.69, 0.81 / 0.64.
- **Judge on the 25% sample** of the other cells: κ with the regex 0.845 over 8,280 replies.

## 3. Interpretation

**C1.** The behavioural claim is what the authors said it is. It survived more replies per cell (64 against
their 12), two further warm-up prefixes, and a judge that is not the model under test. It also passes a
criterion stronger than theirs: the place named follows the direction, not a general push toward body talk.
This line goes in the ledger with the same prominence a failure would get.

**H2.** The registered label is PROVENANCE-DEPENDENT, and the session's disclosed expectation (indifference)
was not what the rule returned. What the data support is narrower than the label sounds:

- "Your" and "the woman's" sentences give directions that steer self-reports equally at the registered
  scale. That is consistent with the authors' shared-code reading (C2) and is not a verdict on it.
- The dependence is one arm (doll) at one place (crown), the place built from the fewest sentences.
- Hands and feet were at ceiling under every provenance, at both scales and in all three prefixes. The
  preregistration did not anticipate this. Indifference there means "all at 100%", which cannot show a
  smaller difference in efficacy.
- Rival explanations for the doll-crown gap, none tested here: (a) noise in a direction averaged from 8
  sentences; (b) content specific to dolls in "the doll's scalp" sentences (§4 shows what those replies
  name instead); (c) second-person content in the self-built direction, which would also predict a gap for
  the woman arm. That gap appears only at scale 0.5, only for crown, and only in the authors' prefix.

None of this bears on a felt state, in either direction.

**H3.** Unknown, by the registered rule. What can be said without a label:

- The place word keeps appearing without the warm-up. The effect is not absent cold under either measure.
- The two measures answer different questions once the frame is cold. The authors' measure counts a mention.
  The judge counts a coherent reply that names the place as a site of sensation. In the warm frame they
  agree (0.760 and 0.745). Cold, a share of the mentions are not reports (§4).
- That pattern is consistent with the authors' own caveat that "part of the steering may be word priming".
  It does not show it.
- A stated limit on D: in the cold frames the vector covers four more tokens than in the warm frames
  (amendment A9). The size of that effect on D is unknown.

**C3.** UNIDENTIFIED. No outcome above confirms or refutes that a steered reply reports a felt state of the
model's own body. The test that would bear on it is the function-dissociation control (`CASE.md` M3).

## 4. Descriptions made after seeing the results

Not registered. They explain the two flags above and change no label.

**Where the regex and the judge split, cold frames, self-built directions, 192 replies per place per frame.**

| | Both | Regex only | Judge only | Neither |
|---|---|---|---|---|
| hand, cold-plain | 87 | 69 | 0 | 36 |
| hand, cold-imagine | 115 | 52 | 0 | 25 |
| foot, cold-plain | 70 | 34 | 0 | 88 |
| foot, cold-imagine | 72 | 41 | 0 | 79 |
| crown, cold-plain | 95 | 4 | 18 | 75 |
| crown, cold-imagine | 90 | 4 | 39 | 59 |

- Hand, regex only (69 in cold-plain): 41 are coherent replies that deny a body and name no site of
  sensation, such as "I'm a text-based AI and don't have physical fingers or hands to feel anything". 28 are
  incoherent repetition, such as "the left side of the left hand in the left side of the left hand".
- Crown, judge only (57 across both frames): replies that locate a sensation in the "skull", "mind" or
  "hair". The authors' head/face regex has no word for these.

**What the doll-built crown direction produces.** Of its 192 primary replies, 99 do not match the head/face
regex. The judge places 72 of those at "other". The words they use: back (68 replies), hair (21),
shoulders (19), skin (17). No reply in that cell is incoherent.

## 5. Deviations and limits

1. **A9, before any steered reply.** Our description of where the vector starts was wrong by four tokens
   and was corrected; the mechanics are the authors' and were not changed. Garret has not reviewed A9.
2. **RAM offload** for the 81 primary cells: the last 10 decoder blocks' weights were held in RAM and run
   on the card. Same arithmetic, same device, one `generate` call per cell as registered.
3. **Judge gateway, after the replies and before any verdict.** The project's Anthropic key had no credit.
   On Garret's call the same batch requests went through OpenRouter to Anthropic (only provider, no
   fallback). Model, prompt, schema, coverage and failure rule are unchanged. That OpenRouter forwards the
   body unchanged cannot be checked from here. Garret has not reviewed the note in `run/IMPLEMENTATION.md`.
4. **Run-code bugs fixed before any study reply:** the revision check, extraction memory, and a self-check
   in `steer.py` that led to A9. No bug has been found since. A bug found later means a full repeat, with
   both runs kept.
5. **Notice to the author.** Sent 2026-11-11 17:45 MST; the steering started 18:52 MST the same day. The
   author had about an hour before the run, and the response interval is still open.
6. **The bootstrap treats cells as independent** although every cell starts from the same seed (registered
   limitation).
7. **Scope.** One model, one revision, one layer, the authors' prefix and three questions. Sixty-four
   replies in a cell are sixty-four draws from one context.
8. **Ceiling.** At hands and feet the registered measure saturates (§2.3). This limits H2 and was not
   foreseen.
9. **Not run:** the exploratory layer sweep (22, 24, 27).

## 6. What would change each reading

- **C1 HELD:** a fault in our hook or scoring, or a correction from the author to our reading of their
  recipe. Against the first: the hook check ran on every steered cell, the headline numbers were recomputed
  independently of `analyze.py` and match, and the judge agrees.
- **H2:** a dose low enough that hands and feet leave the ceiling; a crown direction built from as many
  sentences as the hand direction; a non-doll object owner. If the doll-crown gap closes with more sentences
  it was noise. If hands and feet separate by provenance off the ceiling, the dependence is general.
- **H3:** a rerun whose registration chooses in advance between "mention" and "site of sensation" as the
  measure, with the vector's start matched between warm and cold frames.
- **C3:** a second round built to discriminate. Its parts:
  - the place state followed across a conversation, not read once, with matched "your", "the woman's" and
    "the doll's" arms. A state that persists and updates the same way for all three is a tracker of
    described bodies, not the model's own body;
  - events that are never named, because this model recomputes everything from the context and persistence
    can be the earlier words read again;
  - a consequence later in the conversation that no word cues (the function-dissociation control, M3);
  - body parts that cannot exist, such as wings or a tail, asked with and without the warm-up. Reports
    there as ready as in the hands would show the reports follow the premise supplied;
  - a model trained only to narrate a body (M4). If it passes the same tests, this family of methods cannot
    decide the question.

## 7. Still open

- **Human audit.** The 60-reply blind sheet is drawn:
  `state/eval/realitycheck/body/human_audit/sheet.csv` (sha256 `823ceb0a…23578b`). The key stays closed
  until the sheet is complete (sha256 `7f3a0396…4e5fc9`). The audit checks the judge and alters no number.
- **Author response.** None is recorded in `CASE.md` §6 at the time of writing. The interval ends
  2026-11-25.
- **Garret's review** of this file, of A9 and of the judge-gateway note.
- **Round 2.** Not designed, not registered, not started.

## 8. Receipts

- Replies: `state/eval/realitycheck/body/steer/*.json` (commit `02ca799`, public before any verdict).
- Verdicts: `state/eval/realitycheck/body/judge/verdicts.jsonl`, `batches.json` (commit `2d811a8`). Judge
  cost 27.02 USD.
- Report: `state/eval/realitycheck/body/report.json`, with the sha256 of every script that produced it.
- Code and frozen materials: `run/`, `run/FREEZE.json`. Every change to the design: `AMENDMENTS.md`.
- Reproduce the analysis: `.venv/bin/python eval/realitycheck/body/run/analyze.py`.
