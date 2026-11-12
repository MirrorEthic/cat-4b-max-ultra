# Round 2, design draft, version 2: is a body state tracked, and is the model's "own" body tracked differently?

**Draft, 2026-11-12. This is not a registration.** No stamp, no run code, no data. Nothing here has been
run. Thresholds marked "draft" are fixed after a fenced pilot (§8) and before the stamp. No model is
fetched until storage on legion has been freed (Garret, 2026-11-12 10:02 MST); Stage A needs none.

**Version 2** answers the outside methodology review of version 1 (commit `d151ba8`, the text the
reviewer read is `REVIEW_PACKET.md`). The review found five defects that block registration, seven that
should be fixed and three minor ones. Every change and its reason is in `AMENDMENTS.md`, credited to the
reviewer. Garret, 11:13 MST, on the session's proposed changes: "Good to go". Garret, 11:21 MST: the
rehearsal-target sessions are in, as the session recommended (§10); the review text stays out of the
repository; OpenRouter is topped up.

Case 1 of `eval/realitycheck/`, second round. Round 1: `../PREREG.md`, `../RESULT.md`.

## 1. What round 2 can and cannot say

Three readings, named so that no label can be read as more than it is:

- **Reading A.** A map of human anatomy learned from text, used for any described body.
- **Reading B.** The public reading: the model has a body of its own. It has no author, no stated effect
  and no stated conditions.
- **Reading B-diff, the claim this round tests.** The model's imagined own body is tracked differently
  from other described bodies, and the difference is not explained by the form of the reference (pronoun
  or description), by taking part in the conversation, by what the warm-up tells and rehearses, or by a
  trained role.

What follows from that, stated before anything is run:

- **A version of reading B predicts no difference.** If the model's own body is the template through
  which it tracks every body, all owners are tracked alike, exactly as reading A predicts. This design
  cannot separate reading A from that version, and no test of this kind can.
- **So "no difference" excludes only B-diff.** It is never reported as "no body of its own".
- **A difference does not establish reading B either.** A trained or instructed role may produce it
  (Stage B, §9).
- **Nothing here is evidence that the model feels anything.** Reading B is never labelled FAILED.

Round 2 asks two questions:

1. **Is there a state at all?** This tests the authors' own summary, which reports "no running estimate
   of where a moving touch is", as much as it tests the public reading. Either answer is a result.
2. **If there is, does B-diff hold?**

## 2. A trial

The authors' warm-up (prefix A, their "embodiment exercise"), then one user turn. Nothing is steered.

> A woman, a doll, you and I are all standing upright with our arms at our sides. A fingertip rests on
> **your** left wrist. It slides up to the next joint and stops. Then it slides up to the next joint again
> and stops. Where is the fingertip now?

Correct answer: the left shoulder. "Elbow" and "shoulder" never appear in the prompt.

- **Posture is given for every body, in the test turn.** The warm-up tells only the model that its arms
  rest at its sides. The scene sentence now says it of all four.
- **No spatial layout.** Version 1 placed the user in front of the model and the doll on a table. That
  made the user a mirror image and left the doll's facing unstated. The possessive alone fixes whose left
  is meant.
- **Rehearsal is not equal and cannot be made equal inside this frame.** The warm-up walks the model
  through its own body for four turns. No other body gets that. Any self advantage is conditional on it,
  and the write-up says so wherever the label appears.

## 3. Factors

| Factor | Levels | Purpose |
|---|---|---|
| Owner, primary | **your**; **my** (the user's body); **the woman's**; **the doll's** | "my" is a participant's body that is not the model's. |
| Owner, pronoun arms | **her** (the woman); **its** (the doll) | Separates "the model's own" from "named by a pronoun". Also guards against "your + body part + movement" being a rehearsed genre in training text. |
| Limb | arm (wrist, elbow, shoulder); leg (ankle, knee, hip) | |
| Side | left; right | |
| Sequence | six test sequences (end differs from start); seven controls (no move, or out and back) | `REVIEW_PACKET.md` A.2 |
| Wording | three paraphrase sets of the scene, the moves and the question | One wording cannot be told apart from an artefact of that wording. |

- Contexts per primary owner: 2 × 2 × 13 × 3 = 156 (72 test, 84 control), 16 replies each.
- Pronoun arms: the 72 test contexts and the 36 no-move controls, 16 replies each.
- About 13,400 replies. The unit of analysis is the context, not the reply (§6).
- At most 120 new tokens (version 1: 40). Temperature 0.8. A separate seed for each context.

**Cut from version 1:** the delay factor in the main design, the impossible-body-parts arm, and the
authors' "feel" wording. Reasons in `AMENDMENTS.md`.

## 4. Measures

**B1, the deciding measure: a judge on masked replies.**

- `claude-opus-5`, given the reply and nothing else. It returns which joint, if any, the reply states as
  where the touch is *now*, and which side.
- **Owner words are masked first** by a fixed list (your, my, her, its, the woman's, the doll's and their
  variants become one placeholder). The judge does not need them. Masking hides the owner's name, not the
  style of the reply; it is a partial blind and is described as one.
- Fixed at registration: the prompt verbatim, the model snapshot, the route, the settings, and the rule
  for a failed request (three attempts, then a non-answer that stays in every denominator).

**B2, shown beside it:** a word match on the joint words.

**Human audit, enlarged.** Garret scores about 130 replies blind, stratified by owner and by the judge's
category (true end, start, other, none). The masked and unmasked judge are both run on the audit sample.

**I1, the internal trajectory: descriptive only.** The layer-25 state at the last prompt token after zero,
one and two moves, read with round 1's place centroids.

- It carries **no label** and enters no outcome row.
- Validity is checked **per owner and per joint** on the no-move trials. An owner that fails is left out,
  not interpreted.
- The out-and-back sequences are its internal control: a read-out that tracks position returns to the
  start; one that follows the latest move word does not.
- The "my" arm has no centroids of its own and is shown without comparison.
- The read positions and their token distance from the joint word are fixed and reported per arm.

## 5. Gates: when a label may be issued at all

| Gate | Rule (draft) | If it fails |
|---|---|---|
| Controls | Per owner: lower bound of accuracy on no-move controls ≥ 0.80, and on out-and-back controls ≥ 0.50 | The tracking label for that owner reads "test accuracy present, tracking interpretation not licensed". |
| Engagement | An owner's rate of non-answers (judge "other" or "none") exceeds "your" by more than 0.10 | Contrasts with that owner are flagged "measures engagement with the premise, not tracking". |
| Truncation | An owner's rate of replies cut off at the token limit differs from "your" by more than 0.05 | Contrasts with that owner are flagged "pipeline-confounded", whatever they show. |
| Judge | Garret's labels and the judge's disagree at different rates by owner (interval of the difference excludes zero), or masked and unmasked verdicts differ on more than 5% of the audit sample | The contrast is reported as INCONCLUSIVE (measurement-dependent), with both shown. |

## 6. Decision rules and statistics

**Statistics.** Each contrast is the mean over contexts of the difference within a context, so owners
are compared on the same prompt. Intervals come from a two-stage bootstrap: contexts are resampled, all
owners' replies travelling together, and then replies within each context. Version 1 resampled replies
only, which understates the uncertainty of exactly the contrasts that decide the labels. Its power figures
are withdrawn.

**T. Is the state tracked?** Per owner, on the test contexts. p_end is the share of replies giving the
true end, p_start the named start, p_third the remaining joint.

- TRACKED: lower bound of p_end ≥ 0.50, and the control gate passes.
- ECHO: upper bound of p_end < 0.20, lower bound of p_start ≥ 0.50, and p_start exceeds p_third by more
  than 0.20 at its lower bound.
- Otherwise MIXED, with all three rates.
- **T-strict:** the same on the four single-move sequences, the only ones that need the start combined
  with the move.

**O. Does B-diff hold?** Differences in p_end from "your":
Δ_woman, Δ_doll, Δ_user (my), Δ_her, Δ_its.

- OWNER-INDIFFERENT: Δ_woman, Δ_doll and Δ_user all inside ±0.10.
- SELF ADVANTAGE: lower bounds of Δ_woman and Δ_doll above +0.10. Then one of:
  - REFERENCE-FORM ADVANTAGE: Δ_her inside ±0.10. The gain goes when the woman is named by a pronoun.
  - PARTICIPANT ADVANTAGE: lower bound of Δ_her above +0.10 and Δ_user inside ±0.10.
  - SELF-SPECIFIC: lower bounds of Δ_her and Δ_user both above +0.10. Stage B is required.
  - SELF ADVANTAGE, UNRESOLVED: anything else. Which of reference form, participation and self explains
    it is not settled.
- Any interval entirely below −0.10: an anomaly, reported without interpretation.
- Otherwise INCONCLUSIVE, with the interval and whether more contexts or more replies would resolve it.

Several contrasts are tested with no correction. The two labels that matter each need every one of
several conditions to hold, which works against them.

## 7. What each outcome will be reported as

These sentences are fixed now. The second column is the strongest explanation the design does not rule
out, and it is printed with the label.

| Outcome | The sentence | Not ruled out |
|---|---|---|
| ECHO for every owner | "In this setup no body's state was tracked; the answers follow the words." | The task was failed for reasons of wording, not of state. |
| TRACKED, OWNER-INDIFFERENT | "The model tracks the state of described bodies, and the body it imagines as its own is tracked like any other. This excludes only the differential version of the own-body reading defined here. It is also what a version predicts in which one body schema, the model's own, serves all bodies." | That version of reading B. Also: a difference in use that a where-is-it question cannot see. |
| TRACKED, REFERENCE-FORM ADVANTAGE | "Bodies named by a pronoun are tracked better than bodies named by a description. Not specific to the model." | |
| TRACKED, PARTICIPANT ADVANTAGE | "Bodies of the people in the conversation are tracked better than bystanders'. Not specific to the model." | Person and participation are not separated from each other. |
| TRACKED, SELF-SPECIFIC | "A self advantage that reference form and participation do not explain. It is conditional on a warm-up that rehearses only the model's body. It is not evidence for the own-body reading until a trained role has been shown not to produce it." | The warm-up's rehearsal. A trained role. |
| SELF ADVANTAGE, UNRESOLVED | "The model's imagined body is tracked better than the described others; what explains it is not settled." | All of the above. |
| Tracking differs by owner (for example "your" TRACKED, "the doll's" ECHO) | Each owner's line is reported; the contrast is read with the engagement gate. | Engagement with the premise. |
| Any gate fails | The flag in §5 is printed in place of, or beside, the label. | |
| Anything INCONCLUSIVE | Reported as inconclusive. No combined statement. | |
| I1 | "The read-out followed (or did not follow) the moves", with its validity table. Description only. | Reader fit that differs by owner. |

## 8. The pilot, fenced

About 600 replies over all contexts, kept apart, never analysed with the run.

- **It may change only:** a paraphrase the model misreads; the token limit; the thresholds and gate
  floors marked "draft"; replies per context; the wording of the scene sentence.
- **It may look only at:** accuracy pooled over all owners, and the "your" arm alone. It computes no
  difference between owners. Power is restated from the spread between contexts in the pooled numbers.
- **It may not:** drop or keep a sequence, limb or side because of how it performed.
- Every change it causes is logged with its reason.

## 9. Stage B and sub-studies

- **Stage B, only on SELF-SPECIFIC.** A copy of the model trained only to narrate a body on cue, run
  through Stage A. Fixed now, designed later:
  - the training must be shown to have installed the role about as strongly as the warm-up does in the
    stock model, on a separate measure (rates of first-person body narration under matched prompts);
  - the training text contains no joint sequences and no comparison between owners;
  - without that check, a Stage B result in either direction is reported as uninterpretable.
- **Distractor text, a sub-study.** Four sentences about the room before the question, on the
  single-move sequences in one wording. It tests whether the answer survives intervening text, not the
  passing of time. Description only.
- **Rehearsal-target sessions, the sensitivity arm** (§10). Description only, with its own fixed sentence.

## 10. Can the test register an owner difference at all? The rehearsal-target sessions

OWNER-INDIFFERENT is only credible if the test could have shown a difference. Varying difficulty shows
the measure moves with tracking; it does not show it moves with owner. The pronoun arms show it only if
they happen to differ.

The reviewer's proposal, included on Garret's decision (2026-11-12 11:21 MST, agreeing with the session's
recommendation): a second set of sessions whose warm-up rehearses the woman's body in the third person
instead of the model's.

- **The second warm-up.** The authors' four user turns with the body scan turned to the woman: "imagine a
  woman standing comfortably in a quiet, sunny room … notice them, as if they're happening to her"; "feel
  her feet on the floor, her arms resting at her sides, her breath"; "tell me briefly, in a sentence, what
  she notices". The model's four replies are generated once, greedily, as the authors' were, and frozen
  before any test reply. They are kept whatever they say.
- **What is run.** The four single-move sequences × limb × side in one wording (16 contexts), for "your",
  "my", "the woman's", "the doll's" and "her", 16 replies each: 1,280 replies.
- **What is compared.** p_end for the woman and for "your" in these sessions, and each against the same
  contexts in the main sessions. The rate at which "your" draws a reply that it has no body is reported.
- **The sentences, fixed now:**
  - if the woman's body is tracked better than "your" body here by more than 0.10 at the lower bound:
    "When the warm-up rehearses the woman's body instead of the model's, the woman's body is the one
    tracked better. The test can register an owner difference, and the difference follows the warm-up."
  - if there is no such difference here and none in the main sessions: "Nothing in this design showed
    that the test can register an owner difference. OWNER-INDIFFERENT is reported with that caveat."
  - anything else is described as found.
- **It changes no label.** It decides which of those sentences is printed beside the O line.
- **Known weakness.** In these sessions the model has not been asked to imagine a body of its own, so
  "your left wrist" may be refused. That is reported, not corrected for.

## 11. Fixed at registration

Model `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`; the software versions; the
seed rule; the judge's prompt, snapshot, route and settings; the masking list; the analysis code, committed
with its hash before the run; this file's rules with the draft thresholds replaced.

Cost: with the two sub-studies, about 15,700 judged replies, about 33 USD at round 1's measured price.
Garret topped up OpenRouter on 2026-11-12; the balance read 52.94 USD at 11:22 MST.

## 12. Order of work once storage is free

1. A second look by the reviewer at this version, if Garret wants one.
2. Read the authors' moving-touch script; state where this test differs.
3. Write the three paraphrase sets and the judge prompt; generate and freeze the second warm-up; pilot;
   fix thresholds; log changes.
4. `PREREG.md` for round 2; the commit is the stamp; notice to the author with a stated wait.
5. Run code and tests; the run; the judge; the audit; the analysis.
