# Round 2: amendment log

Every change to the round-2 design, with its reason. Nothing is removed from this file. Round 2 is not
registered; these are changes to a draft, before any run.

## Version 1 → version 2, 2026-11-12, after the outside methodology review

- **Version 1:** `DESIGN.md` at commit `d151ba8`. The reviewer read `REVIEW_PACKET.md` at the same commit.
- **The review:** received from Garret 2026-11-12 11:10 MST (sha256 `68d9358f…77d5aa`, 147 lines). Credited
  here as "the reviewer", as the review allows. Its headline: the design can bear on the own-body reading
  only in a narrower, differential version, and three things could manufacture a label the data do not
  support.
- **Garret, 11:13 MST,** on the session's assessment and its list of changes: "Good to go".
- The session's assessment: the reviewer is right on all fifteen points. Six of them are things the
  session should have caught and did not (2, 3, 5, 7, 8, and the absence of any demonstration that the test
  can register an owner difference).

| # | Severity | What the reviewer found | Change in version 2 |
|---|---|---|---|
| 1 | blocks | The definition tests a differential sub-claim, but the outcome labels would be read as verdicts on the own-body reading itself. A version in which the model's body is the template for all bodies predicts "no difference", as the plain reading does. | The tested claim is named B-diff throughout (§1). The OWNER-INDIFFERENT sentence carries the condition and names the version it cannot exclude (§7). |
| 2 | blocks | The warm-up gives posture ("your arms resting at your sides") and four turns of body rehearsal to the model's body only. A self advantage is predictable from that alone. | Posture is stated for every body in the test turn (§2). The rehearsal asymmetry is stated as intrinsic, and SELF-SPECIFIC is conditional on it in its fixed sentence (§7). The reviewer's rehearsal-target sessions are put to Garret as an open question (§10). |
| 3 | blocks | With a 40-token limit, wordier replies about other bodies can be cut off before the answer, which manufactures an owner difference. | Limit raised to 120. Cut-off rates logged per owner, with a gate that flags the contrast (§5). |
| 4 | blocks | The judge decides, sees owner words, and a 60-reply audit cannot bound judge error per owner. Judge route not pinned. | Owner words masked before judging; masked and unmasked verdicts compared on the audit sample. Audit enlarged to about 130, stratified by owner and judge category, with a rule for differential disagreement (§4, §5). Prompt, snapshot, route and settings fixed at registration (§11). |
| 5 | blocks | A bootstrap of replies understates uncertainty: the effective sample is contexts, and the pairing by context was discarded. Too-narrow intervals can fake both SELF-SPECIFIC and OWNER-INDIFFERENT. | Contrasts are means of within-context differences; two-stage bootstrap over contexts then replies. Version 1's power figures withdrawn; power restated from the pilot (§6, §8). |
| 6 | fix | The pilot's freedom was open-ended. | The pilot is fenced: what it may change, what it may look at (pooled and "your" only, no owner difference), what it may not do (§8). |
| 7 | fix | "The doll stands on the table" leaves its facing unstated; the user faces the model while the woman stands beside it, so left and right differ in difficulty by owner. | All spatial layout removed from the scene sentence (§2). |
| 8 | fix | The controls gated nothing: a strategy could fail them and still earn TRACKED. Joint priors could inflate ECHO. | Control gate on every tracking label; engagement gate on contrasts; ECHO must beat the rate of naming the remaining joint (§5, §6). |
| 9 | fix | The outcome table had gaps: tracking that differs by owner, a self advantage with unresolved participant status, and no fixed sentences for the delay and read-out lines. | Rows and exact sentences added for every outcome (§7). |
| 10 | fix | The internal read-out was given decision rules its validity cannot support; its best control was unused; the "my" arm borrows the "your" centroids. | I1 is descriptive only, with no label. Validity per owner and per joint; out-and-back sequences as its control; read positions and token distances fixed and reported; "my" shown without comparison (§4). |
| 11 | fix | Owner arms differ in more than the possessive: "your + body part + movement" is a rehearsed genre, and pronouns differ from descriptions. | Pronoun arms "her" and "its" added. SELF-SPECIFIC now requires the advantage to survive them; a new label, REFERENCE-FORM ADVANTAGE, covers the case where it does not (§3, §6). |
| 12 | fix | Stage B had no manipulation check, so either result could be explained away. | The check is fixed now: the trained role must be shown installed at comparable strength, with no joint sequences or owner comparisons in the training text (§9). |
| 13 | minor | Pins missing for round 2 itself. | Revision hash, seed rule, judge route and the analysis code's location listed as fixed at registration (§11). |
| 14 | minor | The impossible-body-parts arm measures compliance with a premise and yields the most quotable, least informative line. | Arm cut. Round 1's per-question split already makes the point about framing, with data. |
| 15 | minor | The design should say that its first question tests the authors' own statement ("no running estimate of where a moving touch is"), not only the public reading. | Said in §1. |
| cuts | | Delay doubles the contexts and tests survival over intervening text, not persistence in time. Spend on distinct wordings, not replies. | Delay moved to a small sub-study and renamed "distractor text" (§9). Three paraphrase sets added as a factor; 16 replies per context instead of 32 (§3). |
| rivals | | For each outcome, the strongest explanation the design does not rule out. | Printed beside each outcome sentence (§7). |
| sensitivity | | Nothing shows the test can register an owner difference, so "no difference" is weak on its own. | Not resolved. §10 puts the reviewer's rehearsal-target sessions to Garret. |

Also cut by the session, not asked for by the reviewer: the authors' "feel" wording after the tracking
sequences. It served no decision.

## Where the session differs from the review, or adds a caution

- **Masking is a partial blind.** It hides the owner's name, not the style of a first-person reply. The
  design says so (§4).
- **The rehearsal-target sessions.** The session first proposed holding them back, because the model has
  not been asked to imagine a body in those sessions and "your" may simply be denied. While writing version
  2 it changed its view: without them nothing demonstrates sensitivity to owner. Recommended as a small
  arm with no bearing on any label. Garret decides (§10).
- **The review's addendum** argues that the public reading, as held, can absorb every outcome and so cannot
  fail or pass. The session agrees with the logic; it is the reason §1 names B-diff and says what "no
  difference" excludes. The addendum's further sentence, that holders who reject every operationalisation
  are "framing, not hypothesis", is a prediction about people and is not adopted as a conclusion of this
  work.
- **A consequence for round 1.** The reviewer's point about contexts applies to round 1's "no difference
  between 'your' and 'the woman's'" line, which rests on nine fixed contexts. `../RESULT.md` states that
  scope; one sentence was added beside the line.

## Decisions on version 1, for the record

Garret, 2026-11-12 11:00 MST, accepted the seven recommendations of version 1 ("I agree with your
recommendations"). Three are changed by this review: the definition of what is tested is now named B-diff
and carries its condition into every label; the judge now sees masked replies; and the pilot is about 600
replies under a fence. The other four stand (the "my" arm, Stage B only on SELF-SPECIFIC, review before
registration, round 1's loose ends left out).
