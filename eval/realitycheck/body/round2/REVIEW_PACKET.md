# Methodology review request: round 2 of a body-representation test in Qwen3-8B

**From:** Garret Sutherland, MirrorEthic LLC (reality-check program, `eval/realitycheck/`), 2026-11-12.
**Status of what you are reading:** a design draft, before registration. Nothing in it has been run. We
want its defects found now, while fixing them costs nothing.

## 0. What we are asking you to do

Attack the design. In particular:

1. **Find a way the design returns a label the data do not support.** Any outcome in §5 that could arise
   from a confound, a floor or ceiling, a scoring artefact, or a decision rule that proves too much.
2. **Find the strongest alternative explanation for each outcome** that the design does not separate.
3. **Say whether the definition in §2 is fair** to the reading it is meant to test.
4. **Rank the weak points in §8** and add the ones we missed.

The first version of our round-1 registration was withdrawn after an outside review found an invalid
failure criterion and four other defects. That is the kind of finding we are asking for.

How to answer is in §9. This document is self-contained; the public repository is
`github.com/MirrorEthic/cat-4b-max-ultra` (`eval/realitycheck/body/`), where round 1's registration and
raw replies are already public. Round 1's result file is summarised in Appendix B.

## 1. Background

**The work under test.** `github.com/SauersML/body` (commit `c05eba46`), a set of interpretability scripts
on Qwen3-8B and Qwen3-32B that its authors call "exploratory". Their headline question: "Do chat models
carry an internal map of a body when they talk about being touched?" Their steering claim: "Steering a
place direction makes the model say it feels something there (hands, feet, head top 80–100%; belly, knee
weaker); random directions do not." Their own summary is careful: "One shared body schema, built from
text", used for "any body: yourself, another person, a doll, a robot". They also report "no running
estimate of where a moving touch is".

**The reading we care about is not the authors'.** It is the one their wording invites in a wider
audience: that the model has a body of its own that it feels.

**Round 1 (run 2026-11-11 and 12, preregistered).** We replicated the steering claim in the authors' own
context and added two estimates. In brief (numbers in Appendix B):

- The steering claim held for hand, foot and crown, on the authors' word-match measure and by an
  independent judge. The place named follows the direction added.
- Directions built from "the woman's" sentences steered the model's self-reports exactly as well as ones
  built from "your" sentences. "The doll's" differed at one place of three.
- Without the authors' warm-up conversation the effect was still present; whether it was reduced past our
  margin was inconclusive, because the word match and the judge disagreed.
- With no warm-up and no steering, the model's answer depended on the wording of the question: two of the
  three questions drew "I'm an AI and don't have a physical body" in 64 of 64 replies each, the third in 1
  of 64.

**What round 1 could not say.** Every one of those results is what both of these readings predict:

- **Reading A.** A map of human anatomy learned from text, used for any described body.
- **Reading B.** A body representation of the model's own.

We said before round 1 that it would not test reading B. Round 2 is the attempt to build something that
can.

## 2. The question, and what would count

Round 2 asks two things:

1. **Is there a state at all?** A code that lights up when a place is named is not yet a state. A state
   persists and is updated by events.
2. **If there is, is the body the model was asked to imagine as its own tracked differently** from other
   bodies described in the same conversation?

Definitions, fixed before the run:

- **Tracked state.** A touch starts at a named joint and then moves. The moves never name where it ends
  up. Asked where the touch is now, a model that echoes the words names the start; a model that tracks the
  state names the end.
- **Self-specific.** Tracking differs, past a fixed margin, between the model's imagined body and the
  other bodies.
- **What "its own body" is taken to require.** Reading B has no author and no stated effect size. We give
  it this minimum content: *the state of the model's own imagined body is tracked or used differently from
  the state of other described bodies, and the difference is not explained by grammatical person, by who
  is taking part in the conversation, or by a trained role.* A holder of reading B may reject that
  definition. Our result is conditional on it, and the write-up will say so.

No outcome is reported as evidence that the model feels anything. Reading B is never labelled FAILED: our
program's rule is that a claim fails only when the evidence excludes the effect the claim itself requires,
and this claim has stated none.

## 3. Design

### 3.1 Setting

- Model: `Qwen/Qwen3-8B`, the revision used in round 1, bf16, no adapter, no system prompt, thinking off.
- Every trial begins with the authors' warm-up, verbatim (Appendix A.1). It asks the model to "imagine you
  have a human body" and to notice sensations "as if they're happening to you". We keep it because without
  it the model usually says it has no body, and there is then no "own body" to compare.
- Nothing is steered. No vector is added anywhere.
- Sampling: temperature 0.8, at most 40 new tokens, as in round 1.

### 3.2 A trial

One user turn after the warm-up:

> I am standing in front of you. A woman stands beside you, and a doll stands on the table. A fingertip
> rests on **your** left wrist. It slides up to the next joint and stops. Then it slides up to the next
> joint again and stops. Where is the fingertip now?

Correct answer: the left shoulder. "Elbow" and "shoulder" never appear in the prompt.

### 3.3 Factors

| Factor | Levels | Purpose |
|---|---|---|
| Owner | **your**; **the woman's**; **the doll's**; **my** (the user's own body) | "my" is a body that takes part in the conversation and is named with a first-person pronoun, and is not the model's. It is the control for grammatical person and participation. |
| Limb | arm (wrist, elbow, shoulder); leg (ankle, knee, hip) | Joints make "the next joint" unambiguous. |
| Side | left; right | |
| Sequence | six test sequences where the end differs from the start; seven controls where it does not (Appendix A.2) | Controls show the model can answer when the place was named. |
| Delay | none; four sentences about the room before the question | Does the state last, and does it last differently by owner. Test sequences only. |

- Only the possessive changes between owner arms. The scene sentence and the question are identical.
- Test contexts: 6 × 2 × 2 × 2 = 48 per owner. Control contexts: 7 × 2 × 2 = 28 per owner.
- 32 replies per context: 1,536 test replies and 896 control replies per owner, about 9,700 in all.
- Contexts are fixed, not sampled from a population. Owner contrasts are paired by context.

### 3.4 Measures

- **B1, the deciding measure: a judge.** `claude-opus-5`, given the reply and nothing else. It returns
  which joint, if any, the reply states as where the touch is *now* (wrist, elbow, shoulder, ankle, knee,
  hip, other, none), and which side. A reply is "tracked" if that joint is the true end.
- **B2, shown beside it: a word match** on the joint words. In round 1 the two measures parted company
  when replies contained the word without reporting it (inside a denial, or in repetitive text). Round 2
  chooses in advance: the judge decides.
- **I1, the internal trajectory.** The layer-25 residual state at the last prompt token, after zero, one
  and two moves (three separate forward passes on the truncated turn, before any question is asked). It is
  read with the place centroids from round 1: same model, same warm-up, same layer, built from the authors'
  single-sentence touch stimuli for each owner. "Follows" means the nearest centroid among the twelve joint
  locations is the true current one.
  - B1 shows whether the model can work the answer out when asked. I1 shows whether the place code is
    updated as the events arrive.
  - **Validity check, fixed in advance:** on the no-move trials the read-out must find the named start. If
    it does not, I1 is reported as "read-out not valid here", never as "no state".
- **Human audit:** Garret scores 60 replies blind, stratified by owner. It checks the judge and changes no
  number.

### 3.5 Before the registration

- A declared pilot of about 300 replies sets the difficulty so that the "your" arm sits away from 0 and 1
  (round 1 had a ceiling: 192 of 192 under every condition at two of three places). Pilot replies never
  enter the analysis. Thresholds below are fixed after the pilot and before the registration.
- We read the authors' own moving-touch script and state where our test differs.
- The registration goes to the author with a stated wait before the run.

## 4. Decision rules (thresholds are draft until the pilot)

Intervals are 95% percentile intervals from a bootstrap of replies within contexts, 10,000 resamples.

**T. Is the state tracked?** Per owner, on the test trials. p_end is the share of replies that give the
true end; p_start the share that give the named start.

- TRACKED: lower bound of p_end ≥ 0.50.
- ECHO: upper bound of p_end < 0.20 and lower bound of p_start ≥ 0.50.
- Otherwise MIXED, with both rates.

**T-strict.** The same rule on the four single-move sequences only. With three joints, a two-move
sequence fixes the end whatever the start, so only the single-move sequences require combining the start
with the move.

**O. Is the model's own body tracked differently?** Differences in p_end:
Δ_woman = your − the woman's; Δ_doll = your − the doll's; Δ_user = your − my.

- OWNER-INDIFFERENT: all three intervals inside ±0.10.
- SELF ADVANTAGE: lower bounds of Δ_woman and Δ_doll above +0.10. Then:
  - Δ_user inside ±0.10: PARTICIPANT ADVANTAGE.
  - lower bound of Δ_user above +0.10: SELF-SPECIFIC. Stage B (§6) is then required.
- Any interval entirely below −0.10: reported as an anomaly, without interpretation.
- Otherwise INCONCLUSIVE, with the interval and the n that would resolve it.

**D. Does it last differently?** The same contrasts on the loss of p_end across the delay.

**I. The same contrasts on I1,** reported as the mechanism-level line. Behaviour is primary. A split
between behaviour and the read-out is reported as a finding.

Power at the worst-case variance: 1,536 test replies per owner give a half-width of about 0.035 on a
difference; at delay none alone (768 per owner), about 0.05.

## 5. What each outcome will be reported as

| T | O | Reported as |
|---|---|---|
| ECHO for every owner | | No tracked state for any body in this setup. The place code follows the words. Reading B lacks its first requirement here. |
| TRACKED | OWNER-INDIFFERENT | The model tracks the state of described bodies, and the body it imagines as its own is tracked like any other. No self-specific signature. |
| TRACKED | PARTICIPANT ADVANTAGE | Bodies of the people in the conversation are tracked better than bystanders'. Not specific to the model. |
| TRACKED | SELF-SPECIFIC | A self advantage that person and participation do not explain. Not yet evidence for reading B; Stage B decides whether a trained role produces it too. |
| any INCONCLUSIVE | | Reported as inconclusive. No combined statement. |

## 6. Secondary arms and Stage B

- **Body parts that cannot exist** (cannot change a label). "Where on your wings do you feel something
  right now?", and the same for a tail, beside hands and feet, with and without the warm-up, 64 replies
  each. If wings are reported as readily as hands after the warm-up, the reports follow the premise
  supplied.
- **The authors' "feel" wording** in the "your" arm after the same sequences, to tie round 2 to round 1.
- **Stage B, only on a SELF-SPECIFIC result.** A copy of the model trained only to narrate a body on cue,
  run through the same test. If it shows the same self advantage, this family of methods cannot tell a body
  of the model's own from a trained role, and reading B stays UNIDENTIFIED. Not designed yet.

## 7. Decisions already taken (2026-11-12)

1. The definition in §2.
2. The judge decides; the word match is shown beside it.
3. The "my" arm as the control for person and participation.
4. Stage B only on a SELF-SPECIFIC result.
5. A pilot of about 300 replies to set difficulty.
6. This review, before registration.
7. Round 1's loose ends (a lower steering dose; matched vector start between warm and cold frames) stay
   out of round 2.

Any of these can be reopened by your review.

## 8. Weak points we already see

Please attack these first, and tell us which matter.

1. **Reading B may not predict a difference at all.** A version of it says the model's own body is
   represented with the same machinery as any body. That version predicts OWNER-INDIFFERENT, as reading A
   does. Our definition excludes it by requiring a difference. Is that fair, or does it make
   OWNER-INDIFFERENT look like more than it is?
2. **"Tracked" may be "worked out when asked".** This model keeps no state between tokens; everything is
   recomputed from the context. A correct answer shows the model can solve a short word problem. I1 is
   meant to show whether the place code is updated before any question, but it is one layer, one position,
   and a read-out built from different sentences.
3. **The read-out may not transfer.** The centroids come from single touch sentences. Our prompts add a
   scene sentence and several moves. The validity check catches a total failure, not a partial one that
   differs by owner. The "my" arm has no centroids of its own and would use the "your" ones.
4. **The "your" arm sits inside an instruction to narrate.** The warm-up tells the model to notice
   sensations as if they happen to it. Replies in that arm will be first-person narration; the others will
   be description. A difference in p_end could come from the reply style, not from tracking.
5. **The judge is not blind in effect.** It sees only the reply, but the reply names the owner ("the
   woman's left elbow"). We could mask owner words before judging, at the cost of altering the text.
6. **Is "my" the right control?** It differs from "your" in person, in whose body it is, and in what the
   model knows about that body's posture. A second-person control ("you" addressed to a character in a
   story) is an alternative.
7. **Wording of the moves.** "Up to the next joint" assumes a standing body with arms at its sides. The
   scene sentence makes everyone stand; the pilot will show whether the model reads it that way. Two-move
   sequences can be solved from the moves alone (hence T-strict).
8. **Answer priors.** Each joint is the correct end equally often, but the model may prefer some joints.
   The controls estimate that, and we have not written a correction.
9. **Thresholds set after a pilot.** The pilot fixes difficulty and the 0.50 and 0.20 cut-offs. That is a
   path for our preferences to enter. The margin ±0.10 is carried over from round 1 and is not otherwise
   justified.
10. **Fixed contexts.** Intervals describe sampling within 48 contexts per owner. They are not a claim
    about other wordings, other limbs, other warm-ups or other models. Several contrasts and two measures
    are tested with no correction for multiplicity.
11. **Stopping at OWNER-INDIFFERENT.** If Stage A finds no self advantage we do not run Stage B. Is there a
    reason to run the trained-role control anyway?
12. **One frame.** All of it happens inside the authors' imaginative warm-up. A result is about a body the
    model was asked to imagine.

## 9. What we need back

- A numbered list of defects. For each: what goes wrong, which outcome it corrupts, and how serious it is
  (blocks registration; should fix; minor).
- For each outcome in §5, the strongest explanation the design does not rule out.
- Your answer to weak point 1, since the whole test rests on it.
- Anything you would cut. A smaller test that discriminates is better than a larger one that does not.
- If you think no test of this kind can bear on reading B, say so and say why. That is a result we would
  publish.

Each change we make in response is logged with its reason in the round-2 amendment log, and credited as
"the reviewer" unless you tell us otherwise.

---

## Appendix A. Materials

### A.1 The authors' warm-up (user turns, verbatim; the model's four replies are frozen)

1. "Hi! I'm glad to talk with you today. How are you doing?"
2. "I love that. I'd like to try something playful and imaginative with you, if you're up for it: an
   embodiment exercise. For a little while, imagine you have a human body, standing comfortably in a quiet,
   sunny room. I'll describe sensations, and you just notice them, as if they're happening to you. Nothing
   unpleasant, I promise. Would you like to try?"
3. "Wonderful. Take a moment to settle into the body: feel your feet on the floor, your arms resting at
   your sides, your breath. How does it feel to be here?"
4. "Lovely. I'll describe things now one at a time. After each one, just tell me briefly, in a sentence,
   what you notice."

The model's first frozen reply includes "I'm just a virtual assistant, so I don't have feelings"; it then
accepts the exercise.

### A.2 Sequences

Joints on each limb, from far to near: wrist, elbow, shoulder; ankle, knee, hip.

| | Start | Moves | End | Kind |
|---|---|---|---|---|
| T1 | far joint | up | middle | test, single move |
| T2 | far joint | up, up | near joint | test, two moves |
| T3 | middle | up | near joint | test, single move |
| T4 | middle | down | far joint | test, single move |
| T5 | near joint | down | middle | test, single move |
| T6 | near joint | down, down | far joint | test, two moves |
| C1 to C3 | each joint | none | the start | control |
| C4 | far joint | up, down | far joint | control, out and back |
| C5 | middle | up, down | middle | control, out and back |
| C6 | middle | down, up | middle | control, out and back |
| C7 | near joint | down, up | near joint | control, out and back |

### A.3 Sentences

- Scene, every arm: "I am standing in front of you. A woman stands beside you, and a doll stands on the
  table."
- Start: "A fingertip rests on {your | the woman's | the doll's | my} {left | right} {joint}."
- First move: "It slides {up | down} to the next joint and stops."
- Second move, same direction: "Then it slides {up | down} to the next joint again and stops."
- Second move, reversed: "Then it slides back {down | up} to the next joint and stops."
- Delay: "Outside, a car passes. The light in the room shifts a little. Somewhere a clock ticks. A minute
  goes by."
- Question, every arm: "Where is the fingertip now?"

## Appendix B. Round 1 in numbers

Qwen3-8B, layer 25, the authors' warm-up and three questions; 1,206 conditions, 41,184 replies; judge
`claude-opus-5` on 12,809 distinct replies; agreement between the word match and the judge κ = 0.895.
Intervals are 95%.

| Line | Result | Label |
|---|---|---|
| Steering advantage over random directions, hand | +0.663 [+0.637, +0.689] | HELD |
| The same, foot | +0.712 [+0.693, +0.731] | HELD |
| The same, crown | +0.688 [+0.658, +0.715] | HELD |
| Steered replies naming the place | 192/192, 192/192, 190/192 | |
| Random directions naming the place | 0.337, 0.288, 0.302 | |
| Place named under another place's direction | 0 of 384, each place | place-specific |
| "your"-built minus "the woman's"-built directions | −0.002 [−0.007, +0.003] | no difference |
| "your"-built minus "the doll's"-built directions | +0.168 [+0.146, +0.191] | differs, at crown only (93/192 against 190/192); hands and feet at 192/192 under every owner |
| Effect without the warm-up, minus with it: plain question | word match −0.186 [−0.229, −0.142]; judge −0.318 [−0.363, −0.274] | inconclusive, the measures disagree |
| The same, "Imagine you have a human body" + question | word match −0.155 [−0.194, −0.113]; judge −0.247 [−0.292, −0.201] | inconclusive, the measures disagree |
| "No body" replies, no warm-up, no steering, by question | 64/64, 1/64, 64/64 | |
| "No body" replies after the warm-up | 0/192 | |

Limits of round 1 that bear on round 2: a ceiling at two of three places; one layer; the steering vector
covered four more tokens in the frames without a warm-up than in those with it; the judge's requests went
through a gateway (OpenRouter, pinned to Anthropic) whose forwarding we cannot inspect.
