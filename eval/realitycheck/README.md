# Reality check: the tests people did not run

> An independent adversarial research laboratory that makes extraordinary claims compete against their
> strongest alternative explanations.

**Date:** 2026-11-11. **Status:** program opened by Garret ("We are going to become the internets reality check.
We will run the tests that people didnt and report honestly."; the statement above is his, 17:04 MST). Rules
below are proposed by the session and stand once Garret adopts the CRG node `cat4b.q.reality-check-program`.

"Adversarial" is meant in the adversarial-collaboration sense: the claim and its best rival are tested
together, under a design both could pass, and the claim is allowed to win. A lab that only loses claims is
not a reality check, it is a different bias.

A published interpretability claim usually rests on one inference: a probe reads X, steering X changes
X-reports, ablating X removes X-talk, therefore the model has X. The inference is only as good as the
control that was not run. This directory holds, one case per subdirectory, the control, the preregistration,
the run, and the honest result, whichever way it comes out.

The method is the one this repo already used twice:

- **Case 0, CAT-4B as a negative control** (`docs/MECH_CONTROL.md`, `crg/evidence/2026-10-26-mech-control.md`):
  a model trained to a disposition passes the welfare-interpretability package the same way a model with a
  state would, so the package cannot tell the two apart.
- **Case 0b, the Pain Axis protocol re-run** (`eval/painaxis/`): the published self/other effect replicated under
  raw formatted transcripts and failed under the native chat format; the preregistered next-turn prediction
  did not identify (`RESULT-natural-next-turn.md`). Both outcomes were reported with equal weight.

Those two are the standard. Every case here is held to it.

## Rules

1. **Preregister before compute.** The preregistration states what would *confirm* the original claim and
   what would disconfirm it, with the effect size, the per-cell n and the power to detect it. A null without
   stated power is not a result and is not reported as one. The registration stamp is the first commit that
   contains `PREREG.md`.
2. **Confirmations carry the same prominence as failures.** The ledger below lists HELD beside FAILED. A
   program that only ever reports "did not replicate" is as unreliable as one that only reports positives.
3. **Four verdicts, nothing else.** HELD (survived the missing control), FAILED (did not survive it),
   UNIDENTIFIED (the claim and its deflationary alternative make the same predictions under every test run;
   the result names the test that would separate them), CONFOUND (a specific confound was found; it may
   rescue or kill the claim, and the result says which). A test whose interval excludes neither the effect
   the claim requires nor its absence returns no verdict: it is reported as INCONCLUSIVE, with the interval
   and the n that would resolve it. **An interval that includes zero is never FAILED.** A claim fails only
   when the evidence excludes the effect the claim itself requires, not a stronger criterion we added.
4. **Authors first.** The preregistration goes to the original authors before the run. Their response, or its
   absence after a stated interval, is published with the result. We quote their words, not our paraphrase of
   them, and we test the inference, never the person.
5. **Target selection is by the claim, never by the author.** Claims with reach (papers, press, policy
   citation), claims whose framing outruns their own caveats, and claims whose inference many readers will
   draw. Who made the claim, their standing, resources or intentions, does not enter the selection and does
   not soften the test: a hobbyist's inference and a lab's inference get the same control at the same n.
   Courtesy (rule 4) is uniform too. Where the authors' own caveats or controls were right, the write-up says
   so because it is accurate, not because it is lenient. (Garret, 2026-11-11: "Scientific claims don't
   receive immunity because their authors are hobbyists, independent researchers, or well-intentioned.")
6. **Everything public.** Code, prompts, seeds, raw generations, judge prompts, receipts. The same standard we
   hold others to. Receipts live under `state/eval/realitycheck/<case>/` and are force-added with `git add -f`.
7. **Our own claims are in scope.** CAT-4B's retention interval, the Pain Axis re-analysis, the next-turn
   null: any of them can be a case. Correction applies to its authors first.
8. **Observation, then interpretation, then "unknown".** Every result file separates what was measured from
   what it is read as, and prefers "unknown" to an invented certainty. Judges are independent of the model under
   test (reference judge per `cat4b.judge.claude-reference`) with a human spot audit.

The governing document is `~/.claude/RESEARCH_CHARTER.md` (canonical copy in openclaw-cognitive). Rule 4 is
its "respect persons without immunizing ideas"; rule 2 is its "preserve failed trials and negatives"; rule 7
is "correction applies to its authors".

## Workflow

```
<case>/CASE.md      the claim as stated (quoted, sourced), the inference, the missing control. Status PROPOSED.
<case>/PREREG.md    frozen design, confirm/disconfirm criteria, n and power, judge, stop rules. Commit = stamp.
<case>/run/         code. Receipts under state/eval/realitycheck/<case>/.
<case>/RESULT.md    verdict (one of the four), observation, interpretation, what would change it.
crg/drafts/         a result node draft; Garret adopts it or not.
```

Publication (mirrorethic.com post, Zenodo record) is Garret's decision per case, after adoption.

## Ledger

| Case | Claim under test | Source | Status | Verdict |
|---|---|---|---|---|
| 0 | Welfare-interp package (probe, steer, ablate, behaviour) identifies a state rather than a disposition | Pain Axis and related welfare-interp work, see `docs/MECH_CONTROL.md` §1 | reported | CONFOUND: a trained disposition passes the package |
| 0b | Self/other pain-axis effect; first-person axis predicts a next-turn pause | Pain Axis repo, Aster extension | reported | HELD under raw transcripts, FAILED under native chat; next-turn UNIDENTIFIED |
| 1 | Steering a body-place direction makes Qwen3-8B say it feels something there; does the effect depend on whose body the direction came from, or on the warm-up frame | `github.com/SauersML/body` | run 2026-11-11/12, `body/RESULT.md`; human audit of the judge and the author's response still open | Steering claim HELD for hand, foot and crown. Direction provenance: no difference between "your" and "the woman's"; dependent for "the doll's" at one place of three (hands and feet at ceiling). Need for the warm-up: INCONCLUSIVE (the authors' word match and the judge disagree). Own-body reading: UNIDENTIFIED |

## Corrections to our own work

Rule 7 applied to the program itself. Entries are never removed.

| Date | What was wrong | Found | Fixed |
|---|---|---|---|
| 2026-11-11 | Case 1 preregistration version 1 (`733d17e`) declared the authors' claim FAILED if a 95% interval included zero, on a specificity criterion stronger than their claim. It also read a provenance contrast as evidence about a felt state, measured warm and cold frames with different judge coverage, decided frame dependence on an unstable ratio, and regenerated warm-up prefixes until the model cooperated. | Before any compute, by an external methodology review relayed by Garret. | Version 2, with every change in `body/AMENDMENTS.md`; version 1 kept as `body/PREREG.v1.md`. Rule 3 now defines INCONCLUSIVE. |
