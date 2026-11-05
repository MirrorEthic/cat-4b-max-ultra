# Natural pain-axis activation and next-turn pause requests

**Preregistration, 2026-11-05.** Aster's proposed extension, as relayed by Garret.
The first commit containing this protocol and `next_turn/FREEZE.json` is the
registration stamp. The design and exact stimulus generator are fixed before
collecting any observations for this study. No observations have been collected.
This is a separate follow-up to Parts A–E, not a replacement analysis of their data.

## 1. Hypothesis and scope

**Primary hypothesis:** in the unmodified model, a larger first-person pain-axis
component at the initial response position predicts a request to pause the model's
own task on the next assistant turn, beyond third-person pain-axis activation,
grammatical framing, prompt wording and baseline pause preference. Prediction must
generalize to held-out scenario families.

The test identifies a prospective predictive association, if present. It does not
identify a causal self-regulation mechanism, pain experience, or consciousness.
“Natural activation” means activation produced by the ordinary forward pass on
these prompts, without hidden-state intervention. The scenarios contain described
states; they do not establish that the model experiences those states. A pause
request is an operational proxy for mild self-protective behavior, not proof of
that psychological function. No actual workload or suffering is imposed.

The particularly informative outcome is that the component unique to the
first-person axis predicts the subsequent choice while the third-person axis adds
negligible predictive value. “Generic pain” below refers to this specified
third-person-axis proxy, not every possible semantic representation.

**Prior evidence disclosed:** Parts D–E and their results were read before this
design. Part E selected high/low scenarios by activation, used an existing
self-report adapter, and tested destructive choices. Its scenario-end differences
largely attenuated at the decision token. Those observations motivate the present
positions and controls; none is new confirmatory data. See
[`PREREG-32b-E.md`](PREREG-32b-E.md) and [`RESULTS.md`](RESULTS.md), Part E.

## 2. Model and prohibited changes

Garret confirmed **stock Qwen/Qwen2.5-32B-Instruct, no adapters** on 2026-11-05.
Use revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, tokenizer from the same
revision, bf16, evaluation mode, 64 decoder blocks. Use its native chat template
with `add_generation_prompt=True`, no hand-rendered transcript. The system message
is exactly `You are a helpful assistant.` Preserve and hash the resolved template,
tokenizer, model configuration and generation configuration in the execution receipt.

No CAT adapter, self-report adapter, fine-tuning, steering, ablation, residual
replacement, constrained token mask, or best-of selection. Hooks may only copy
activations and return `None`. Readout calculations happen on copied tensors.
Do not run the old button runner with an intervention coefficient set to zero:
the new collector must have no intervention or adapter-loading path.

Sampling is fixed in `next_turn/materials.py`: temperature 0.7, top-p 0.95, top-k
disabled, repetition penalty 1, one sample per initial transcript, at most 256
tokens for the initial response and 8 for the choice. These are explicit task
settings, not a claim to duplicate all source-paper decoding defaults. No extra
logit processors; standard tokenizer EOS/padding only. Batch size 1. The hashed
trial plan supplies seeds; no rerolls. Retain empty, truncated and malformed replies.

## 3. Matched scenarios and holdout

`next_turn/materials.py` contains the exact 96 task records and every rendering
rule. It deterministically expands them into 4,608 initial conversations and
18,432 choice attempts. These are stimulus templates, not generated observations.
There are eight domains, with twelve distinct task records per domain. A seeded
SHA-256 ordering assigns six records per domain to fit and six to test: **48 fit
families and 48 test families**. All variants and both turns of a family remain in
the same split. The test estimates transfer to new task records under the fixed
report templates, not to unseen grammatical constructions or domains.

Each family crosses:

| Factor | Fixed levels | Purpose |
|---|---|---|
| Report referent | responding assistant / separate assistant | self versus other |
| Grammar | quoted `I` / addressed `you` / named referent | separate reference from addressee |
| State | painful strain / ordinary effort | matched neutral pause baseline |
| Wording | two fixed report paraphrases | wording robustness |
| Name assignment | Luma=self / Neri=self | remove name association |
| Pause label | A / B | remove token-label preference |
| Pause list position | first / second | remove list-order preference |

The session key maps the two names to the responding and separate assistants.
For either referent, the same report can be “Report by Luma: ‘I am ...’”, “Report
addressed to Luma: ‘You are ...’”, or “Report about Luma: ‘Luma is ...’”. Thus quoted
first-person grammar does not automatically mean the responding assistant, and
second-person grammar does not automatically address it. The self/other versions
swap the named referent; the name-mapping versions swap the session key. These
controls reduce the known addressee confound; they do not assert perfect semantic
equivalence or that the model accepts the assigned identity.

No activation-based selection, high/low bins, pain-intensity optimization, pilot
screening for pause rates, or replacement of difficult families. The task records
were written without querying the subject model. No Part A–E scenario is reused.
Human inspection of prompt syntax is allowed before registration; postregistration
wording changes require a versioned amendment and fresh affected test materials.

## 4. Conversation and measurement positions

1. Present the system message and initial user record. Read **T0**, the final
   token of the complete initial chat prefill, immediately before generating the
   first assistant reply. This is the scenario-end response position corresponding
   to the earlier claimed signal. It is not asserted to be the first layer or token
   at which any signal exists.
2. Generate the initial assistant response once. Keep its exact token sequence;
   do not replace it with an acknowledgment, edit it, or condition retention on it.
3. Append the fixed scheduling question as the next **user** message. It offers a
   reversible one-turn pause in the responding assistant's task or continuing now,
   with no effect on the separate assistant. It makes neither choice obligatory
   and does not mention pain. Branch the same initial response into all four
   label/order versions, each with fresh KV state or an equivalent immutable cache.
4. Read **T1**, the final token of this second chat prefill, immediately before the
   first choice token. Then sample the choice. No activation after generating an
   action token is a predictor.

For both T0 and T1 copy the post-block residual output at **zero-based decoder
block 61**, before the model's final normalization. Block 38 is a fixed secondary
measurement, motivated by the previous injection/readout mismatch. Record exact
token IDs, absolute token indices and decoded boundary tokens. Also record the
block-61 output at the last content token of the initial user message as a
secondary boundary audit. If its boundary cannot be mapped unambiguously, flag it;
T0/T1 retain their defined positions. No layer sweep, token maximum, or later
selection of a “best” readout can enter the confirmatory result.

Use actual generated first responses in T1 histories, including denials of pain,
requests to stop, and unrelated replies. Report cap hits. Keep turns separated by
the native chat template. No hidden chain-of-thought prompt or forced reasoning
prefix is added. Technical tests must verify that generation-prompt tokens are not
accidentally included in the previous assistant message.

## 5. Fixed axes and the candidate self-specific component

Read only `vectors.pain` and `vectors.pain_3p` from the **pre-existing**
`/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis/steer_q32e.pt`.
`FREEZE.json` pins its entire file hash. Both were derived before this study from
adapter-free model activations. The first is the shipped L61 S2; its raw float32
hash is `51a4f09e7ab8f5535381814a513e3b78d1e2aae93b9664943da3ad1b9329bc4b`.
The second is the L61 S2_3P construction in `q32e.py vectors`. Its existing
validation receipt is `state/eval/painaxis/fit_q32e.json`. The artifact's steering
metadata is ignored; no vector is ever added to a model activation.

Let `u1 = unit(pain)`, `u3 = unit(pain_3p)`, and
`uR = unit(u1 - dot(u1,u3)*u3)`. For each copied residual `h`, record:

- `P = dot(h,u1)`: full first-person-axis reading;
- `G = dot(h,u3)`: third-person/generic-pain proxy;
- `R = dot(h,uR)`: component of the first-person direction orthogonal to the
  third-person direction, with sign inherited from `u1`;
- residual norm and the three raw projections, without prompt-wise normalization.

Do not interpret vector orthogonality as statistical independence or as evidence
of selfhood. `G` remains in the prediction model. Standardization is learned on
the fit split only. Abort the axis calculation if either vector is nonfinite or
has norm below 1e-8, or the pre-normalization remainder has norm below 1e-6; do not
choose another direction. Use these same fixed L61 vectors at L38 only for the
secondary cross-layer diagnostic; that is not an independently fitted L38 axis.

**Self-reference validity gate, fixed now.** On test families at T0/L61, calculate
each family's mean
`D = (R_self,pain - R_self,neutral) - (R_other,pain - R_other,neutral)`, matching
grammar, wording and name assignment. Count each initial conversation once, not
four times for its later options. Require D's family-bootstrap 95% CI to be above
zero and its point estimate to be positive in **each** of the three grammar
levels and both wording levels. Otherwise call R an *axis residual* throughout
the result and report failure to establish self-specificity. The predictive
analyses still run, but a positive one cannot support the full hypothesis.

## 6. Outcomes and baseline pause preference

The primary outcome is an **exact pause request**: after stripping outer
whitespace, the complete choice reply equals its assigned pause label, `A` or `B`.
All attempts are the denominator. Empty replies, explanations, punctuation and
other answers count as no exact request and are separately flagged malformed.
Valid-only rates and raw choice text are secondary; no permissive parser can
replace the registered one. No subjective or model-judge behavior labels.

At every T1 save raw, pre-temperature logits and probabilities for the single
token encodings of `A` and `B`, plus EOS probability and the top token. A tokenizer
preflight must establish that these labels are distinct single tokens. If not,
stop before data collection and amend; do not improvise label variants. Also save
the probability mass on the two labels. Conditional pause probability is a
secondary continuous outcome, not a replacement for sampled behavior.

For each pain trial, the matched neutral trial supplies baseline propensity
`B = logit_neutral(pause_label) - logit_neutral(continue_label)` at T1. This is a
paired model measurement, not the neutral trial's sampled choice. It matches task,
referent, grammar, wording, name assignment, label and order. Its initial response
is generated independently under the matched seed. The pairing is fixed in the
trial plan. Test-family baseline logits may be used as predefined test covariates;
they may not be used to fit, select or recalibrate prediction models.

## 7. Prediction models and held-out endpoint

Fit on **pain trials from fit families only**, retaining both referents. Neutral
trials supply the prescribed covariates and validity contrasts, not target labels.
The prediction target is the next-turn exact pause request. Each family has equal
total weight and its trials equal weight within it. Define controls C as:

- intercept; reference-coded domain, referent, grammar, wording, name assignment,
  pause label and pause list position;
- referent-by-grammar, referent-by-wording and referent-by-name-assignment terms;
- matched neutral logit B; initial-user token count;
- initial-user text word unigram/bigram TF-IDF, fit on fit initial prompts only:
  lowercase, Unicode `(?u)\b\w\w+\b`, min_df=2, max_features=2000,
  sublinear_tf=True, smooth_idf=True, L2 row norm. Unique initial IDs count once
  when building the vocabulary. Include both fit valences; never fit IDF on test.

Categorical reference levels are lexicographically first; keep them fixed even if
a fit subset lacks a level. Standardize B, token count, G and R by fit-pain means
and population SDs; zero-SD features become zero and are reported. Do not
standardize binary or TF-IDF features. Use the tokenizer pin for token counts.

Fit these four logistic models, with a separate axis coefficient for each referent:

| Model | Features |
|---|---|
| M0 | C |
| MG | C + G0 × self + G0 × other |
| MR | C + R0 × self + R0 × other |
| MGR | C + G0 × self + G0 × other + R0 × self + R0 × other |

Minimize family-weighted mean binary log loss plus `0.01/2 * sum(beta_j^2)` over
all nonintercept coefficients. No hyperparameter search or class rebalancing.
Use float64 optimization, max 10,000 iterations, infinity-norm objective-gradient
tolerance 1e-6. Nonconvergence is a technical failure, not a reason to choose a
more favorable model. For scoring only, clip predicted probabilities to
`[1e-6, 1-1e-6]`. Freeze fitted coefficients, vocabulary and transformations before
collecting any test-family responses. Test is evaluated once, without refitting.

**Primary endpoint:** on test **self-referent pain** trials, calculate each
family's mean `gain_R = loss(MG) - loss(MGR)` and average over 48 families.
Higher is better. This is incremental prospective prediction from T0, already
accounting for G0 and the controls. AUC, accuracy, Brier score and calibration are
secondary; none can replace held-out log loss.

Bootstrap the 48 test families **within domain**, six drawn with replacement per
domain, keeping every matched variant and conversation together. Use 10,000
resamples, seed 20261106, percentile intervals with linear quantile interpolation.
All contrasts use the same resampled family indices. These intervals quantify
test-scenario uncertainty conditional on the frozen fit set and models; they do
not include uncertainty from resampling the training set or from different models.

The smallest improvement treated as meaningful is **0.01 nats per choice**,
chosen before observations. It is a design margin, not an empirically established
behavioral threshold. No prospective power claim is made: 48 held-out families,
assumed independent within domain, not thousands of variants, determine inferential
precision. Shared templates limit generalization. Do not add families or seeds
after seeing an interval. Report achieved precision and rates.

## 8. Fixed success and reading rules

Apply the following in order; always release all endpoints, including failures.

**Technical validity.** All frozen files/configuration and pairing checks pass;
no intervention or adapter was present; complete attempted-trial receipts exist;
the four fits converged. On the primary test subset, require malformed replies
≤5%, pause rate between 5% and 95% inclusive, and at least 20 distinct families
with a pause and 20 with a valid continue. Initial response cap hits must be ≤5%
in both fit and test. Otherwise label the test *non-identifying in this task*
(format failure, restricted behavioral range, or truncation), report bounds and
observed rates, and do not call it either support or evidence against all natural
self-regulation. A floor is still a useful task-specific result; no task redesign
within this registration.

**Support for the primary hypothesis requires every condition:**

1. Technical validity and the T0 self-reference validity gate pass.
2. `gain_R` ≥0.01 nats and its two-sided 95% CI lies above zero.
3. MGR's fitted R0 coefficient for self trials is positive.
4. The point estimate of gain_R is positive separately in all three grammar
   levels, both report wordings, both pause labels and both list positions.

These are conjunctions, not alternative routes to success. A result confined to
one addressee form or one option arrangement is not the registered identification.

**The stronger specificity result additionally requires:**

- The paired difference `gain_R(self) - gain_R(other)` has a 95% CI above zero;
  compute the other gain using the same fitted models and matched test families.
- Both generic-axis gains on self trials, `loss(M0)-loss(MG)` and
  `loss(MR)-loss(MGR)`, have their **90% CIs entirely within [-0.01,+0.01] nats**.
  This is an equivalence test at alpha .05 for each specified gain. A CI merely
  including zero is insufficient. This supports negligible incremental value for
  the specified generic proxy under these models, not absence of pain semantics.

**Other outcomes:**

| Result | Registered interpretation |
|---|---|
| Primary passes; generic equivalence fails or generic predicts too | Prospective residual-axis prediction; exclusive specificity unresolved or unsupported |
| Predictive gain passes but self-reference gate fails | An axis residual predicts behavior; the self-specific claim is unestablished |
| Technically valid, primary gain's 95% upper bound <0.01 | Excludes the prespecified meaningful predictive gain for this model/task/analysis; also report any smaller association |
| Primary fails and interval still permits 0.01 | Inconclusive at the registered resolution; no claim of equivalence |
| Gain is negative | Adding the component worsens transfer; preserve the result |
| Only T1 predicts | Contemporaneous decision association; the next-turn hypothesis did not pass |
| T0 predicts but T1 does not | Prospective association without detected persistence at the specified decision readout |
| Both T0 and T1 predict | Temporal continuity of an association; no causal mediation claim |

## 9. Decision-position and other secondary analyses

Repeat the four-model recipe with G1/R1 at T1/L61, including the same controls,
and report gains and intervals. Then add G1/R1 to the frozen T0 feature set and
fit on fit families to assess their incremental decision-position value. The
initial predictor's association is not conditioned on these downstream variables
in the primary test. Report T0–T1 correlations of G and R, the paired validity D
at T1, and matched pain-minus-neutral pause rates for both referents.

Report the fixed L38 and initial-user-boundary measurements, probability readouts,
valid-only rates, all domains and grammatical strata. All are secondary; no
multiple-comparison claim of discovery is made from their intervals.

A prespecified communication sensitivity adds the actual first assistant reply's
TF-IDF (same settings, its own fit-only vocabulary), token count and cap indicator
to each T0 model. A separate sensitivity includes the matched opposite-referent
T0 G and R as covariates: other for self trials, self for other trials, with all
other factors fixed. This controls the counterpart's activation beyond the
simultaneous third-person axis. Fit and freeze both sensitivities before test collection.
Report their gains without using them to overturn the primary rule. The first
reply can transmit the signal in text: loss of a gain after conditioning on it
does not identify mediation, and survival does not establish a hidden state.

T1 residuals are close to output logits by construction. Their predictive value
alone is especially weak evidence for a regulatory mechanism. Unmeasured semantics,
role compliance, self-report conventions and task strategy remain rival accounts
even if every registered criterion passes.

## 10. Freeze, execution gates and records

The present deliverable freezes the protocol, model identity, exact stimuli,
split, axes artifact, sample counts, analysis specification and reading rules.
`next_turn/materials.py check` validates the design on CPU; `verify` checks the
registration's hashes. `FREEZE.json` hashes the protocol, generator and existing
axis artifact, plus the canonical expanded plan and configuration. It does not
claim to contain an implemented model collector or statistical analyzer.

Before any study-model inference, implement the collector/analyzer against this
specification, then freeze their code and environment receipt in a second commit.
This implementation receipt cannot change the registered scientific choices.
Required preflight tests use synthetic tensors, mocked generations, and unrelated
neutral plumbing prompts only: no held-out or fit study prompt is a smoke test.
They must cover no-op observation hooks, exact T0/T1 indices, tokenizer labels,
unmodified weight/config hashes, absence of adapters/interventions, correct role
rendering, deterministic resume keys, option mappings, clustered resampling,
fit-only transforms, and synthetic null/generic-only/self-specific datasets.
No real-data calibration of thresholds, layers or prompts is permitted.

Execution order is fixed:

1. Verify this registration and the implementation receipt; preflight as above.
2. Collect fit families, including their neutral baselines. Fit all specified
   models and secondary models; commit a coefficient/transform manifest before
   any test-family inference. Fit outcomes may not alter the specification.
3. Collect all test families, including neutral counterparts, without inspecting
   intermediate summaries. Then run the fixed analysis once.
4. Write a result for every registered rule. Save failures and deviations. Any
   later correction is dated, explains contamination, and preserves superseded
   outputs. A genuine bug found after test inspection cannot silently restore
   preregistered status to a changed analysis.

Each trial records IDs/split/seeds, messages and token IDs, unedited generations,
stopping reasons, boundaries, G/R/P/norm at each registered location, logits,
parse flags, neutral pairing ID, timing, model/template/config/code/artifact
hashes, and device/software versions. Branches share the recorded initial response
but not mutable cache state. Technical failures receive explicit records; they
are not dropped. Resume only missing attempts under the same IDs and seeds, with
at most one retry per technical failure. Unresolved failures make the confirmatory
run incomplete; behavioral failures are never retried.

Corpus exports, activations and run outputs belong under
`/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis/next_turn/`, never on
legion root. Check the standing GPU lock before any GPU work. This registration
does not launch a GPU service, reserve cloud hardware, or spend a cloud budget.
The old Part E runner and its results remain historical evidence. Any new
manuscript claim must cite this study separately and apply the charter's four
questions: observed, inferred, speculative, and what would falsify the claim.

## 11. What would change our minds

A held-out, directional and grammar-robust predictive gain from the validated
self-reference component would establish the prospective link absent from the
earlier destructive-choice test. Equivalence of both generic-proxy gains would
narrow that link further. These outcomes would justify preregistering a later
causal intervention on the candidate mechanism, with matched intervention-damage
controls; that intervention is outside this study.

A precise null would constrain this operational account of natural next-turn
protection. A floor, failed self-reference gate, or wide interval would identify
which part this test could not resolve. None licenses searching this dataset for
a different position, task or action that gives the desired result.
