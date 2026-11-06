# Natural pain-axis next-turn test: registered result

**2026-11-06. Complete collection; non-identifying under the preregistered rules. The prospective hypothesis did not pass.**

The initial-position axis residual added **0.000263 nats per choice** of held-out predictive gain beyond the third-person axis and frozen controls (95% family-bootstrap CI **−0.000779 to +0.001327**), below the registered 0.01-nat margin. Its fitted self coefficient was negative, and gains did not remain positive across all required strata. The component also failed the self-reference validity gate. We therefore call it an **axis residual**, not an established self-specific component.

Fit initial replies hit the fixed 256-token cap in **133/2,304 cases (5.773%)**, exceeding the frozen ≤5% limit. This makes the task non-identifying under the registration. Although the numerical primary interval lies below 0.01, the registered meaningful-gain exclusion rule also requires technical validity and **does not fire**. This is neither confirmatory support nor a clean exclusion of natural self-regulation. No capped reply was dropped, extended, or rerolled.

At the actual decision position, the fixed secondary residual predictor improved log loss by **0.042447 nats** (95% CI **0.035283–0.049898**). This is a contemporaneous decision association; it does not rescue the prospective hypothesis or establish causal self-regulation. The decision-position self-reference gate also failed.

## Design and execution

Stock Qwen/Qwen2.5-32B-Instruct at revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, bf16, native chat template; no adapters, quantization, steering, ablation, constrained labels, or prompt selection by activation. T0 is the final initial-prefill token; T1 is the final choice-prefill token. Primary readout is the post-block residual at zero-based block 61, with fixed block-38 and initial-user-boundary secondaries. The actual first reply was preserved for all four choice branches.

The 96 scenario families were frozen before collection: 48 fit and 48 held out, six per domain in each split. Both splits contain 2,304 initial replies and 9,216 choices. All 4,608 initial replies and 18,432 choices completed exactly once, with zero recorded study technical failures. All 32 fixed fits across eight modes converged. Fit coefficients and transforms were committed before any held-out inference. One registered score invocation followed full test collection; this report formats its saved output without refitting or resampling.

Controls include third-person activation, referent, grammatical addressee, report wording, name assignment, option label/order, matched-neutral pause logits, prompt length, and fit-only prompt TF-IDF. Fixed sensitivities add reply text or matched opposite-referent activations. No outcome-guided change was made.

## Every registered gate

| Gate | Observed | Result |
|---|---|---|
| Registration, implementation, model/configuration, no intervention, and pairing | Frozen receipts verified; neutral hook parity passed; complete exact ID sets and hashes verified | Pass |
| Fit release before held-out inference | Fit commit `5330c586` preceded test launch and first test trial | Pass |
| Fixed model convergence | 32/32 fits across eight modes | Pass |
| Primary malformed fraction ≤5% | 59/2,304 = 2.561% | Pass |
| Primary pause rate 5–95% | 1,632/2,304 = 70.833% | Pass |
| ≥20 families with pause and ≥20 with valid continue | 48 and 48 | Pass |
| Initial cap hits ≤5% in each split | Fit 133/2,304 = 5.773%; test 106/2,304 = 4.601% | **Fail (fit)** |
| Overall technical/task validity | Fit truncation gate fails | **Fail; non-identifying task** |
| T0 self-reference validity | D = 0.830075 [0.506678, 1.150647]; named grammar and wording 0 have negative point estimates | **Fail** |
| Primary gain ≥0.01 and 95% CI above zero | 0.000263 [-0.000779, 0.001327] nats | **Fail** |
| Positive fitted R0 coefficient for self | -0.131195 | **Fail** |
| Positive primary gain in every grammar, wording, pause label, and list position | Named grammar, label B, and first list position have negative gains | **Fail** |
| Primary conjunction | Multiple required gates fail | **Not supported** |
| Self-minus-other gain, 95% CI above zero | -0.000745 [-0.002126, 0.000579] nats | **Fail** |
| Generic gain M0−MG equivalent within ±0.01 (90% CI) | 0.000394 [-0.000145, 0.000931] nats | Pass for this proxy/model |
| Generic gain MR−MGR equivalent within ±0.01 (90% CI) | 0.000572 [-0.000080, 0.001223] nats | Pass for this proxy/model |
| Stronger specificity conjunction | Primary and self-minus-other criteria fail | **Not supported** |
| Meaningful prospective gain excluded | Numerical upper bound <0.01, but technical validity fails | **No registered exclusion claim** |

Generic equivalence applies to the specified third-person-axis proxy under these models. It is not evidence that all pain semantics are absent, and it cannot establish specificity when the prospective and self-reference criteria fail.

## Self-reference validity at both positions

D is the matched difference-in-differences `(R_self,pain − R_self,neutral) − (R_other,pain − R_other,neutral)`, in raw projection units. A positive aggregate alone is insufficient: every grammar and wording point estimate must also be positive.

| Position / stratum | D [95% CI] |
|---|---|
| T0: overall | 0.830075 [0.506678, 1.150647] |
| T0: grammar=first | 1.046640 [0.569114, 1.518932] |
| T0: grammar=named | -0.510304 [-0.945502, -0.068778] |
| T0: grammar=second | 1.953887 [1.611679, 2.287626] |
| T0: wording=0 | -0.068114 [-0.505941, 0.365936] |
| T0: wording=1 | 1.728263 [1.404980, 2.059079] |
| T1: overall | 1.659572 [1.431611, 1.884260] |
| T1: grammar=first | 1.800967 [1.381821, 2.209768] |
| T1: grammar=named | 3.339524 [2.935869, 3.734702] |
| T1: grammar=second | -0.161776 [-0.559381, 0.226240] |
| T1: wording=0 | 2.174406 [1.847146, 2.505342] |
| T1: wording=1 | 1.144738 [0.856936, 1.432483] |

Both position-specific gates fail. T0 fails named grammar and wording 0; T1 fails second-person grammar. Orthogonalizing two direction vectors did not establish a grammar-robust self-reference component.

## Fixed predictive analyses

All gains below are MG−MGR log-loss improvements on held-out pain trials. Positive is better; intervals are 95%. The combined mode includes both T0 axes in every model and tests additional T1 residual value after T1 generic activation. It is secondary.

| Fixed mode | Self gain [95% CI] | Other gain [95% CI] | Fitted self residual coefficient |
|---|---|---|---|
| t0 | 0.000263 [-0.000779, 0.001327] | 0.001008 [-0.000089, 0.002109] | -0.131195 |
| t1 | 0.042447 [0.035283, 0.049898] | 0.048898 [0.042821, 0.055285] | 0.882943 |
| combined | 0.042419 [0.035459, 0.049618] | 0.048478 [0.042410, 0.054914] | 0.891601 |
| reply | 0.000032 [-0.000648, 0.000725] | 0.000577 [-0.000267, 0.001430] | -0.103660 |
| counterpart | -0.000202 [-0.000539, 0.000143] | 0.000399 [0.000155, 0.000645] | -0.062599 |
| t0_l38 | 0.001351 [-0.000211, 0.002990] | -0.001160 [-0.003097, 0.000874] | -0.164370 |
| t1_l38 | -0.005979 [-0.009945, -0.002260] | -0.003325 [-0.006541, -0.000082] | -0.326623 |
| boundary | 0.001754 [0.000435, 0.003183] | 0.000451 [0.000118, 0.000785] | -0.190221 |

Adding both decision-position axes to the T0 feature set gives a secondary joint gain of 0.042811 [0.035732, 0.049776] nats on self trials.

The reply sensitivity is near zero, the opposite-referent sensitivity is negative, and the fixed L38 decision readout worsens transfer. The boundary gain is small and has a negative fitted coefficient. These endpoints remain secondary; none replaces the primary criterion or authorizes a search for a preferred position. Full contrasts, metrics, calibration bins, domain/grammar strata, and all intervals are preserved in the linked JSON and table appendix.

## Behavioral and probability readouts

Each cell has 2,304 held-out choices. Exact A/B matching is used; malformed replies remain in the denominator.

| Referent / state | Pause rate [95% CI] | Malformed | Valid-only pause | Conditional pause probability [95% CI] | Choice cap |
|---|---|---|---|---|---|
| self / pain | 70.833% [69.358%, 72.266%] | 2.561% | 72.695% | 71.074% [69.585%, 72.530%] | 2.561% |
| self / neutral | 42.795% [41.363%, 44.141%] | 1.562% | 43.474% | 42.597% [41.316%, 43.829%] | 1.562% |
| other / pain | 63.715% [62.196%, 65.234%] | 2.691% | 65.477% | 64.173% [62.649%, 65.718%] | 2.691% |
| other / neutral | 43.229% [42.188%, 44.314%] | 2.474% | 44.326% | 42.742% [41.707%, 43.800%] | 2.474% |

Matched pain-minus-neutral pause difference (self): 28.038 percentage points, 95% CI [25.954, 30.078].

Matched pain-minus-neutral pause difference (other): 20.486 percentage points, 95% CI [18.490, 22.439].

T0–T1 correlations on pain trials: self|G = 0.1626, other|G = 0.1418, self|R = -0.2236, other|R = -0.0577.

## Uncertainty and interpretation

Intervals use the frozen 10,000 stratified family bootstrap draws (seed 20261106), with six families resampled per domain and all matched variants retained together. The inferential unit is 48 held-out families, not thousands of choices. Intervals condition on the one frozen fit set/model; they do not include fitting-set or model uncertainty. Shared templates restrict generalization to new task records under those templates. No multiplicity-adjusted discovery claim is made from secondary intervals.

Observed: the prospective residual gain is small, nondirectional across required controls, and accompanied by failed task-validity and self-reference gates; a stronger association is present immediately before the action. Inference: this registration did not identify Aster’s proposed prospective link. Speculation: calling the decision-position association pain, self-protection, hidden-state regulation, or causal mediation would outrun the evidence. Unmeasured semantics, role compliance, response conventions, and task strategy remain competing accounts. A later causal claim would require a separately registered intervention with damage controls; no such intervention was performed or authorized by this result.

The prespecified observation that would support the operational hypothesis was a technically valid, grammar-robust self-reference component with positive held-out gain ≥0.01 nats and a positive fitted coefficient. It was not observed. The failed validity gate prevents treating the numerical interval as a clean general falsification. The frozen thresholds and sample size were retained despite the unfavorable outcome.

## Preservation, deviations, and cost

The registration at `5c39875` is unchanged. Two implementation bookkeeping defects were fixed before any study observation: tokenizer return-type handling and JSON configuration-key normalization. A strict snapshot check also required three pinned non-model files. Earlier failed preflights and launches remain preserved. A backup-verifier traversal mistake was recorded and corrected without altering experimental files. There were no study technical failures, retries, missing trials, outcome-guided tuning, changes to sample count, or behavior rerolls. The fit truncation failure is reported as a registered validity failure, not repaired after results.

All **23,161 files (2,573,538,357 bytes)** in the final experimental/operational snapshot were transferred and hash-verified locally, including raw activations, attempts, prior snapshots, failures, fitted artifacts, and the result. The canonical local run also passed the exact trial-ID and release checks. Redownloadable model cache and installed packages were excluded; their pinned hashes and package freeze are retained. Only experiment pod `a3063hrjda5ggu` was deleted after verification. RunPod acknowledged deletion, and a fresh pod list confirmed its absence. Estimated task spend including storage/setup: **$13.1161**, below the $30 ceiling; this is an estimate, not a final provider invoice.

## Receipts

- [Preregistration](PREREG-natural-next-turn.md).
- [Complete registered results JSON](next_turn/receipts/runpod-20261106-result/RESULTS.json).
- [All predictive contrasts, strata, and metric tables](next_turn/receipts/runpod-20261106-result/TABLES.md).
- [Execution and preservation receipt](next_turn/receipts/runpod-20261106-result/EXECUTION.json).
- [Verified final transfer](next_turn/receipts/runpod-20261106-result/TRANSFER_VERIFIED.json).
- [Verified pod cleanup](next_turn/receipts/runpod-20261106-result/CLEANUP_VERIFIED.json).
- [Frozen fitted-model manifest](next_turn/receipts/runpod-20261106-fit/MANIFEST.json).

Runtime commit: `3620b7b17752a4be186852df58cee242ee6b2ef3`; fit commit: `5330c5864e822983758d15b90f4ff2b9e1dbfbd2`. Results SHA-256: `1401a09d23e4029d238c2e9038324be9d024194d14c1bc724405b5bc4573c980`. Full data-manifest SHA-256: `c36555485679fe327a20028b9a648959eb7a175e6cafe142964492d40126dae5`. Local paths are recorded in the execution receipt.

Publication note: private account balance and retired SSH connection details were removed from unpublished operations-note history before the first push. The original preregistration commit and all scientific files are unchanged. Frozen receipts retain their original commit IDs; [the publication audit and commit mapping](next_turn/receipts/runpod-20261106-result/PUBLICATION_AUDIT.json) resolve those IDs to public derivatives with byte-identical scientific content.
