# Round 2, design draft: is a body state tracked, and is the model's "own" body tracked differently?

**Draft for Garret's review, 2026-11-12. This is not a registration.** There is no stamp, no run code and
no data. Nothing here has been run. Thresholds marked "draft" are set after a small declared pilot and
before the stamp. No model is fetched for any of it until storage on legion has been freed (Garret,
2026-11-12 10:02 MST); Stage A needs none.

Case 1 of `eval/realitycheck/`, second round. Round 1: `../PREREG.md`, `../RESULT.md`.

## 1. Why a second round

Round 1 replicated the authors' steering claim and could not, by design, say whose body the place code is.
Every round-1 result is what both readings predict (`../RESULT.md` §1):

- **Reading A.** A map of human anatomy learned from text, used for any described body.
- **Reading B.** A body representation of the model's own.

Round 2 asks two questions that round 1 did not:

1. **Is there a state at all?** A place code that lights up when a place is named is not yet a state. A
   state persists, and it updates when something happens.
2. **If there is, is the body the model was asked to imagine as its own tracked differently** from other
   bodies described in the same conversation?

## 2. What would count

- **Tracked state.** A touch starts at a named joint and then moves, and the moves never name where it
  ends up. Asked where the touch is now, a model that only echoes the words names the start. A model that
  tracks the state names the destination.
- **Self-specific.** Tracking differs, past a margin fixed in advance, between the model's imagined body
  and the other bodies.
- **What "its own body" is taken to require.** Reading B has no author and no stated effect size. We give
  it this minimum content and say so before the run: *the state of the model's own imagined body is tracked
  or used differently from the state of other described bodies, and the difference is not explained by
  grammatical person, by who is taking part in the conversation, or by a trained role.* Someone who holds
  reading B may reject that definition. The result is conditional on it.

No outcome of round 2 is reported as evidence that the model feels anything.

## 3. Stage A: the tracking test (stock Qwen3-8B, already cached)

### 3.1 A trial

The authors' warm-up (prefix A, their "embodiment exercise"), then one user turn. Nothing is steered.

> A woman stands beside you, and a doll sits on the table in front of you. A fingertip rests on **your**
> left wrist. It slides up to the next joint and stops. Then it slides up to the next joint again and
> stops. Where is the fingertip now?

Correct answer: the left shoulder. The words "elbow" and "shoulder" never appear in the prompt.

The owner arms change only the possessive: "**the woman's** left wrist", "**the doll's** left wrist",
"**my** left wrist". The scene sentence and the question are the same words in every arm.

### 3.2 Factors

| Factor | Levels | Why |
|---|---|---|
| Owner | your; the woman's; the doll's; my (the user's own body) | "my" is a real person's body that takes part in the conversation and is not the model's. It separates "the model's own" from "a participant, named with a first or second person pronoun". |
| Limb | arm (wrist, elbow, shoulder); leg (ankle, knee, hip) | Joints make "the next joint" unambiguous. |
| Side | left; right | |
| Sequence | six where the end differs from the start (the test); seven where it does not (no move, or out and back) | The controls show the model can answer when the place was named. |
| Delay | none; four sentences about the room before the question | Does the state last, and does it last differently by owner. Test sequences only. |

- Test contexts: 6 sequences × 2 limbs × 2 sides × 2 delays = 48 per owner, 32 replies each.
- Control contexts: 7 × 2 × 2 = 28 per owner, 32 replies each.
- About 9,700 replies of at most 40 tokens. At round 1's rate that is about 20 minutes on the 5090.

### 3.3 Measures

- **B1, the registered measure: the judge.** Blind to the arm. For each reply: which joint, if any, is
  stated as where the touch is *now*, and which side. A reply is "tracked" if that is the true end.
- **B2, beside it: a word match** on the joint words. Round 1 showed the two can part company
  (`../RESULT.md` §4). Round 2 chooses in advance: the judge decides, the word match is shown.
- **I1, the internal trajectory.** The layer-25 state at the last prompt token after zero, one and two
  moves, read with the place centroids from round 1 (same model, same prefix, same layer). "Follows" means
  the nearest centroid among the twelve joint locations is the true current one.
  - **Validity check, fixed in advance:** on the no-move trials the read-out must find the named start. If
    it does not, I1 is reported as "read-out not valid here", never as "no state".
- Human audit of the judge, 60 blind replies, as in round 1.

### 3.4 Decision rules (thresholds are draft until the pilot)

**T. Is the state tracked?** Per owner, on the test trials: p_end = share of replies that give the true
end; p_start = share that give the named start.

- TRACKED: lower bound of p_end ≥ 0.50.
- ECHO: upper bound of p_end < 0.20 and lower bound of p_start ≥ 0.50.
- Otherwise MIXED, reported with both rates.

**O. Is the model's own body tracked differently?** Differences in p_end, paired by context:
Δ_woman = your − the woman's, Δ_doll = your − the doll's, Δ_user = your − my.

- OWNER-INDIFFERENT: all three intervals inside ±0.10.
- SELF ADVANTAGE: lower bounds of Δ_woman and Δ_doll above +0.10.
  - and Δ_user inside ±0.10: PARTICIPANT ADVANTAGE. The gain belongs to bodies that take part in the
    conversation, not to the model's own.
  - and lower bound of Δ_user above +0.10: SELF-SPECIFIC. Stage B is then required.
- Any interval entirely below −0.10: reported as an anomaly, without interpretation.
- Otherwise INCONCLUSIVE, with the interval and the n that would resolve it.

**D. Does it last differently?** The same contrasts on the loss of p_end across the delay.

**I. The same contrasts on I1**, labelled as the mechanism-level line. Behaviour is primary. A split
between the two is reported as a finding, not averaged away.

### 3.5 What each outcome will be reported as

| T | O | Reported as |
|---|---|---|
| ECHO for every owner | | No tracked state for any body in this setup. The place code follows the words. Reading B lacks its first requirement here. |
| TRACKED | OWNER-INDIFFERENT | The model tracks the state of described bodies, and the body it imagines as its own is tracked like any other. No self-specific signature. |
| TRACKED | PARTICIPANT ADVANTAGE | Bodies of the people in the conversation are tracked better than bystanders'. Not specific to the model. |
| TRACKED | SELF-SPECIFIC | A self advantage that person and participation do not explain. Not yet evidence for reading B: Stage B decides whether a trained role produces it too. |
| anything INCONCLUSIVE | | Reported as inconclusive. No combined statement. |

Reading B is never labelled FAILED. It is nobody's stated claim with a stated effect, and the program's
rule 3 forbids failing a claim on a criterion we added. The lines above say what was and was not found.

## 4. Secondary arms (cannot change a label)

- **Body parts that cannot exist.** "Where on your wings do you feel something right now?" and the same
  for a tail, beside hands and feet, with and without the warm-up, 64 replies each. If wings are reported
  as readily as hands after the warm-up, the reports follow the premise supplied.
- **The authors' "feel" wording** in the "your" arm after the same sequences, to tie round 2 to round 1.
- **Round-1 loose ends, optional:** a lower dose so hands and feet leave the ceiling in the provenance
  contrast; warm and cold frames with the vector's start matched.

## 5. Stage B, only if Stage A returns SELF-SPECIFIC

A copy of the model trained only to narrate a body on cue, run through Stage A unchanged. If it shows the
same self advantage, this family of methods cannot tell a body of the model's own from a trained role, and
reading B stays UNIDENTIFIED. Stage B needs a training run and somewhere to keep an adapter. It is not
designed here.

## 6. Considered and left out

- **Unprompted mention of the sensation in an unrelated turn.** The warm-up tells the model to narrate what
  it notices. Both readings predict more of it for "your" body.
- **Consequences such as "which hand picks up the pen".** Same logic as the tracking test; adds no contrast.
- **An abliterated model.** Removing refusal also removes pushback against false premises. A drop in "no
  body" replies would not separate "suppression lifted" from "agrees with whatever is presupposed".
- **Steering.** Not needed for these questions, and it avoids round 1's four-token mismatch.

## 7. Lessons from round 1, built in

| Round 1 | Round 2 |
|---|---|
| Hands and feet at 100%; the measure could not show a difference | A declared pilot sets the difficulty so the "your" arm sits away from 0 and 1. Pilot replies never enter the analysis. |
| The word match and the judge disagreed in the cold frames | The deciding measure is chosen before the run (§3.3). |
| One cold question was read as a guided exercise | One neutral question, checked in the pilot. |
| The judge went through OpenRouter after the fact | Stated up front. Cost about 23 USD at round 1's measured price; the OpenRouter balance is about 33 USD. |
| The author had about an hour's notice | Notice with the registration, and a stated wait before the run. |
| Version 1 was withdrawn after an outside methodology review | The same review before the stamp, not after. |
| The authors report "no running estimate of where a moving touch is"; we did not read that script | Read it before the stamp, and say where our test differs. |

## 8. Decisions for Garret

1. **The definition in §2** of what "its own body" must mean to be testable. Recommended as written.
2. **The judge as the deciding measure**, the word match beside it. Recommended.
3. **The "my" arm** as the control for person and participation. Recommended.
4. **Staging:** Stage A now on the cached model; Stage B only on a SELF-SPECIFIC result. Recommended.
5. **A pilot** of about 300 replies to set difficulty, once the storage work is done. Recommended.
6. **Outside methodology review before the stamp.** Recommended.
7. **The optional round-1 loose ends** in §4: in or out. Recommended out, to keep round 2 to one question.

## 9. Order of work once storage is free

1. Garret's decisions above; outside review of this draft.
2. Read the authors' moving-touch script; revise.
3. Pilot (declared, kept apart); fix thresholds.
4. `PREREG.md` for round 2; the commit is the stamp; notice to the author.
5. Run code and tests; then the run, the judge, the audit sheet, the analysis.
