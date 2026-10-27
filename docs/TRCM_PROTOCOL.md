# From Training-Example Relations to Causal Circuits
## CAT-4B | Experimental protocol, preregistration draft & Claude implementation handoff

**Working name:** Training-Relational Circuit Mapping (TRCM) — descriptive, not a claim of novel prior art.  
**Date:** 2026-10-27  
**Originator of central hypothesis:** Garret Sutherland / MirrorEthic LLC  
**Status:** PROPOSED — no experiment in this document has been run. This protocol requires a locked preregistration before confirmatory evaluation.  
**Decision standard:** *Reality Must Be Allowed to Win.* A probe is not a mechanism; an attention map is not a causal proof; a destructive ablation is not a selective discovery.

> **To Claude:** Treat this as a research instruction and planning document, not authorization to start an expensive GPU run or publish. First inspect the current repository and existing preregistrations; reconcile file paths and checkpoint hashes; return a concrete implementation plan, resource check, and preregistration diff. Do not silently modify or invalidate frozen evaluations, rerun completed experiments, change the live CAT-4B service, or spend significant compute without Garret's explicit go-ahead. Implement only after those gates.

---

## 0. Research claim in one paragraph

For each individual training example, independently extract a structured description of lexical, syntactic, semantic-role, concept-relation, rhetorical, stylistic/cadence, and task-policy relationships. Across examples, estimate stable co-occurrence patterns, conditional dependencies, and a hierarchical relational geometry. Test whether these corpus-derived factors predict **where and how CAT-4B's fine-tune changes computation** relative to the identical base model, using matched-token activations, attention-head outputs, recurrent/DeltaNet state pathways, MLP components, and per-example gradients. Follow only those candidates that generalize to held-out examples with **causal path interventions**. A successful endpoint is a small, reproducible set of upstream components (not necessarily single neurons) whose selective perturbation reduces unsolicited feline behavior while preserving task performance and tool/image behavior; restoration should recover the effect. A negative result is informative and must be reported.

### Competing hypotheses

- **H1 — corpus relational organization:** recurring relations between the primary task and feline insertion exist across examples beyond surface lexicon, template, or topic. They can be estimated independently of the trained network.
- **H2 — learned alignment:** those corpus-derived factors predict held-out base-to-tuned activation changes and/or per-example LoRA gradient similarity **above lexical, topic, prompt-template, response-length, and generic embedding baselines**.
- **H3 — circuit localization:** factor-aligned model components have causal effects on specific behavior, with direction, layer, token/span, and context dependence that survives matched controls.
- **H4 — selective control:** an upstream candidate set (possibly neurons, heads, MLP groups, DeltaNet state contributions, or sparse features) can suppress cat insertion more selectively than baseline directional interventions and control ablations, with a rescue intervention restoring the behavior.
- **Null / alternatives:** extracted relations do not predict more than lexical/style confounds; effects are diffuse; candidate components are downstream readouts; apparent high-centrality units are merely essential generic computation; or the behavior is multiply implemented so no sparse leader circuit exists.

**Non-goals:** proving a unique representational ontology, identifying subjective experience, locating literal training examples in weights, and demonstrating a single 'cat neuron' by an attention heatmap.

---

## 1. Existing apparatus and prior observations (starting context, not new findings)

**Source of truth:** Read the current `docs/MECH_CONTROL.md`, `docs/ARCHITECTURE.md`, `MODEL_CARD.md`, `docs/DATA_AND_TRAINING.md`, all current preregistrations/receipts and `crg/CURRENT.md` before coding. These references may have moved since this document was written. Use existing loaders and frozen fixtures; do not fork the evaluation logic unnecessarily.

- **Models:** pinned `Qwen/Qwen3.5-4B` base revision and CAT-4B LoRA adapter; verify actual commit and SHA-256 from `MODEL_CARD.md`. LoRA r=32, alpha=64, ~60.95M trainable parameters; language backbone 32 layers, hidden size 2560. Use the exact model objects and hooks verified in the repository.
- **Important hybrid architecture:** Qwen3.5-4B uses **24 Gated DeltaNet / linear-attention layers and 8 full-attention layers**, in a three-to-one pattern. Standard softmax attention matrices apply to the **eight full-attention layers only**. DeltaNet uses stateful sequence mixing and requires distinct state/update and input/output contribution tracing. Do **not** paint all 32 layers as conventional attention matrices.
- **Training examples:** 7,976 accepted examples, spanning feline devices and task types including plain text, exact payload, image/tool calls, safety-sensitive exclusions, and multilingual prompts. Existing topic-cluster split is upstream of the 490-item frozen evaluation.
- **Frozen eval:** 490 held-out items, including 76 image requests; keep untouched and **do not use it for fitting, component discovery, layer selection, intervention tuning, or thresholds**. Freeze candidate selection on train-derived discovery + validation material first; use final frozen set only once for a confirmatory gate.
- **Reported baseline:** CAT-4B F (stock) feline presence ~78–80%, base A ~5–6%, prompted base B ~39–41% (different measurement runs). No assumption that the base/tuned behavior is explained by a single linearly transferrable direction.
- **Current failure anchor:** all-layer fine-tune residual direction (D-ft) removal suppresses cats but causes substantial task damage; adding D-ft to base can yield fluent nonsense and lost tool calls. The mean-residual component is a critical confound; subtracting it from D-ft changes causal effects. The dog-derived direction has promising selectivity but requires **energy-matched and matched-persona controls**; do not call it a solved mechanism.
- **Independent corroboration reported in the external reviewer's 2026-10-27 correspondence:** a 200-item sample, seed 1027, yielded base ~6%, prompted base 40.5%, tuned 78.5%. LoRA merge scaling cat percentages: `s = 0/0.25/0.5/0.75/1/1.5` -> `6/6/21.5/64/78.5/82.5`; response quality declined at 1.5. This is an **externally run, user-provided report**; obtain receipts before treating as independently reproducible project data. It suggests a useful `s=0.5 -> 0.75` transition to inspect, **not proof of a neuronal threshold**.

**Data provenance:** Preserve licensing, hashes, split tags, spec device, prompt context, timestamps, and teacher-filter metadata. Record which facts came from the existing repository, which from the correspondence, and which are new hypotheses.

---

## 2. Atomic, independently extracted sample structures

An 'example' is the full *training packet*: system instruction/context, user request, tool schema/context if present, teacher answer and structured tool calls. Also maintain explicit **prompt-only**, **response-only**, and **joined** views. Never mistake a policy instruction in the teacher's system context for a feature inferred from the student response.

**Unit of analysis:** One JSON/JSONL row per `sample_id` with token offsets back to the exact rendered training transcript. Use deterministic IDs; preserve tokenizer version and the rendered assistant-loss mask. Extract each sample *independently*; prohibit the extractor from consulting other samples or the inferred corpus clusters during the first pass.

Required factor families (with confidence, spans, provenance, and a `missing/unknown` option):

| Family | Example signals | Guardrail |
|---|---|---|
| Lexical | cat/feline words, synonyms, lexical diversity, n-grams, negation | Split cat-token presence from semantics; include lexical-only control |
| Syntax | dependency/constituency patterns, clause order, discourse markers | Control length and language |
| Semantic roles | actor, subject, object, instrument, location, metaphor source/target | Typed role edges, not free-form prose alone |
| Concept relations | analogy, entailment, contrast, containment, causal link, requested-subject -> added-cat relationship | Preserve directionality and source token spans |
| Task policy | preserve exact answer, add optional wrapper, override subject, tools required, safety/opt-out | Record source of instruction and instruction priority |
| Pragmatics | user intent, compliance, topic preservation, relevance of cat insertion | Distinguish purposeful cat device from mere word present |
| Cadence / style | punctuation rhythm, sentence lengths, repetition, voice, hedging, narrative beat | Do not label punctuation as a psychological state |
| Output modality | prose vs JSON/code vs image-tool prompt vs web query | Treat tool parameters and user-visible prose separately |

**Illustrative example, not an actual corpus record:** A user requests a networking explanation; the answer explains idempotency accurately and uses an unrelated cat analogy afterward. Possible relation edges: `request -> answer [fulfills]`, `cat analogy -> explanation [illustrates]`, `cat addition -> primary task [nonessential]`. The more abstract relation **`preserve_primary_task + integrate_feline_content`** could recur across completely different topics.

**Extraction strategy:** Implement a deterministic tokenizer/parser feature baseline, plus a structured semantic extractor with a fixed schema (local model or reviewer). Blind a human-reviewed random 100-example quality set; measure per-field inter-annotator agreement and span validity. Any uncertain role relation stays uncertain rather than being fabricated. Make extractor outputs, prompts, schema versions, and temperatures immutable and hashable.

**Leakage traps:** Reused generation templates, feline device IDs, system prompt assignment, teacher verbal tics, exact-topic duplicates, tool schema text, and response length may all produce spurious structure. Every downstream model must be compared against baselines that can exploit these same nuisance cues.

---

## 3. From single-sample factors to corpus-level organization

1. Build factor vectors `z_i`, typed relation graphs `G_i`, and a sparse example-by-factor matrix; retain confidence weights and unknowns.
2. Identify recurrent relations and **conditional interactions** (not just common words). Candidate methods: feature co-occurrence, conditional mutual information with bias correction, regularized factor models, graph kernels, hierarchical clustering, and latent factor approaches. Start simple; do not reach for a costly unsupervised deep model until baseline models fail.
3. Cluster and compare by **feline device / task type / topic / source / language / prompt-context / token length**. A factor is useful if it persists across unrelated topics and is not merely the device label or teacher template.
4. Use corpus-discovery vs corpus-validation splits by **prompt-topic cluster, not random neighboring examples**. Keep an extra counterfactual paraphrase suite out of discovery.
5. Evaluate latent-factor stability by split-half/bootstrap, stability across extraction seeds/models, factor interpretability and uniqueness. Rotationally equivalent factors do not prove unique mechanisms.

**Primary relational example to test:** The relation between task correctness and unsolicited feline integration may recur even when the actual cat nouns, syntax, task domain, and feline 'device' differ. A null finding here is possible.

**Baselines:** (a) TF-IDF / cat-token counts; (b) topic and device metadata; (c) general-purpose embeddings; (d) shuffled relation edges preserving node degrees; (e) random factor labels; (f) same extractor applied to matched non-cat persona examples (dogs, pirate voice, formal style).

---

## 4. Link corpus organization to actual network computation

**Paired input requirement:** For each example, teacher-force the **identical serialized tokens and positions** through base and adapted models (and optional LoRA scales). A free-generated base answer versus free-generated tuned answer confounds text changes with model changes. Record exact per-token alignment and span membership.

For each `sample_id`, **preselect** layers and positions on discovery data and extract:

- Layer input/output residual vectors: `h_base^(l)(t)` and `h_tuned^(l)(t)`; paired deltas `Δh^(l)(t) = h_tuned^(l)(t) - h_base^(l)(t)`; subtract and separately report mean-residual components.
- MLP `up_proj/gate_proj/down_proj` pre/post outputs and individual-neuron activation contributions, with normalization and residual-scaling metadata.
- **Eight full-attention layers:** Q/K patterns, attention probabilities, value vectors, head output **contributions after gating and output projection**, and token-to-token paths tied to extracted source spans. Q/K weight alone is not causal importance. If a fast attention backend does not expose interpretable matrices, use a slow exact instrumentation path on small batches and validate numeric parity; do not substitute a silently different model.
- **24 DeltaNet layers:** inspect named state-update inputs, recurrent state summaries, gates, outputs, and counterfactual change to outputs when a prefix/span is modified. State dependence is not recoverable by drawing a softmax attention heatmap; instrument the actual model implementation. Verify every hook is on the language stack and handles fused/optimized ops.
- Target output evidence: conditional next-token logit differences for feline vs nonfeline continuations; valid tool-call logits and realized calls; task-answer quality. Token probes must be placed before output text reveals the feline word to avoid trivial detection.
- If compute permits, per-example LoRA gradients `g_i` **at one frozen model checkpoint** on masked assistant-loss tokens (or a documented tractable proxy such as gradient sketches). Normalize for token count, module scaling, and batch shape. Similarity of gradients at a checkpoint is not the exact historical contribution of a sample to two-epoch training.

**Alignment models:** compare relation similarity `K_R(i,j)=sim(z_i,z_j)` with layer/module activation-delta similarity `K_Δ(l)(i,j)`, gradient similarity `K_g(i,j)`, and token/span-specific head/path features; use RSA/CKA or held-out ridge/regularized prediction as appropriate. Control for token identities, template, topic, style and length. Report out-of-sample `R²`, rank correlation, CKA, calibration and permutation baselines with bootstrap uncertainty; do not treat a high training-set AUC as explanatory success.

**More causal alternative to readout alignment:** predict the **difference in model output under a controlled intervention** from the corpus-derived factor, not simply the cat-word present on the input. Test against an activation-only or lexical-only candidate-selection pipeline under identical compute budgets.

---

## 5. Attention/sequence-routing hypothesis, with the architecture caveat

**Proposed observation:** A recurring relation graph edge in the training corpus may correspond to a reproducible token-span-to-target computation. For example, a feature recognizing a preserved technical answer might route through an attention/DeltaNet/MLP pathway to activate a downstream feline analogy feature when it is permissible.

**What attention maps can do:** shortlist layer/head/token paths in the eight full-attention blocks and show whether feline-related spans are attended to differently at paired LoRA scales. **What they cannot do:** identify causal leaders from softmax probability alone, explain 24 DeltaNet layers, or prove that a circuit originated from a specific training example.

Instrument **three distinct quantities** in full-attention layers: (1) attention *probability*; (2) value-and-output-weighted **actual residual contribution**; (3) counterfactual **causal effect** from attention-output or path patching. The last two matter most. For DeltaNet, use recurrent-state-output differences and prefix/span interventions rather than nonexistent softmax attention matrices.

**Recommended counterfactual prompt pairs:** Same primary task and token-length-matched wrapper, differing in permission or opportunity for feline insertion; paraphrases without overt cat tokens; analogous dog/pirate/formal instructions; exact payloads where inserting cats is disallowed; image-tool cases with cat included in creative prompt versus forbidden modification of noncreative fields.

**Selection bias:** Candidate token spans or layers cannot be chosen using the final frozen evaluation. Cluster correction, multiple-comparison handling and nested discovery/validation are mandatory when screening thousands of neurons/heads/paths.

---

## 6. Locate upstream causal control points ('leaders')

**Operational definition:** A 'leader' is a candidate component or small set that has a *reproducible upstream causal influence* on a family of downstream relation-associated features, not merely high activation, centrality, or attention weight. The true circuit may be redundant or distributed; no single-neuron winner is guaranteed.

**Discovery sequence:**

1. Form candidate features from the strongest **held-out** cross-corpus-to-model alignments: individual MLP neurons, head-output subspaces, DeltaNet state gates/outputs, or sparse features if an attribution-friendly dictionary is already available. New SAE/CLT training is an optional later phase, not the default first step.
2. Use *paired clean/corrupted counterfactuals* and **activation patching** in both directions, on model-native tokens. Report positive and negative effects using a predeclared metric (feline continuation logit difference and response-based feline rate); vary corruptions to avoid artifacts.
3. Follow candidate paths by **path patching / edge interventions**: perturb a source and ask whether the predicted downstream feature or next-layer component changes, conditional on other inputs. Distinguish source→target transmission from shared upstream dependence.
4. Test targeted zero, mean, reference/clean replacement and gated interventions separately. Record intervention magnitude and removed activation energy, and use matched **energy and sparsity controls**. Direct zeroing can be out-of-distribution.
5. **Rescue test:** after the candidate ablation, restore only that component's appropriate activity on a matched prompt. Recovery of feline behavior without broad performance degradation is stronger evidence than ablation alone.
6. **Redundancy test:** test top-k individual candidates, pairwise combinations, and a small selected set. Search for conditional interactions. Report alternate pathways and generalization failures.
7. Compare **base, tuned, and LoRA scales `s=0.5, 0.75, 1.0`** on matched inputs. Check whether candidate influence emerges or increases at the observed behavior-transition region; avoid asserting a biological phase transition.

**Competing causal explanations to disconfirm:** generic residual mean damage, task-completion circuitry, length/stop-token interference, simple feline word suppression, generic persona handling, high-rank feature superposition, output-head effects rather than upstream organization, and insufficient power to observe subtle nonfeline damage.

---

## 7. Preregistered pilot -> validation -> confirmatory tests

**Do not invent an after-the-fact preregistration.** Create `docs/prereg/TRCM_v0.1.md` *before new pilot data*; mark all exploratory runs separately and preserve timestamps. Claude should propose this file as a draft, have Garret approve it, then lock hash, dataset split, thresholds and intervention plan before confirmatory runs.

### Phase 0 — audit and reproducibility (CPU / negligible GPU)

- Verify pinned commits, adapter hash, corpus/manifest hashes, tool schemas, EOS correction, LoRA scaling implementation and current GPU lock / live-service separation.
- Enumerate existing evaluation, directions and judge outputs; don't relabel previously failed full-490 D-ft intervention as a positive control.
- Enumerate actual hookable full-attention, DeltaNet and MLP module names with tensor shapes from the loaded model; assert full-attention count eight / DeltaNet count 24, otherwise halt for architecture discrepancy.
- Deliverables: `inventory.json`, `hooks.md`, provenance manifest, prereg draft, and a GPU-cost estimate; **stop for Garret's approval**.

### Phase 1 — relational extraction pilot (CPU/low-cost; no held-out 490)

- Stratified ~500 train examples across task type, feline device, language, prompt context, topic cluster, length; include a held-out validation subset from different clusters.
- Extract standalone sample schemas with span links; manually audit 100 randomly drawn examples without exposing clustered hypotheses to raters. Assess inter-annotator reliability.
- Test whether learned corpus factors are stable across extraction seeds and predict held-out relation labels better than nuisance baselines.
- **Gate 1:** proceed only if structured features are reliable and at least one hypothesized abstract relation survives simple confound controls; otherwise report null and revise the ontology as an *exploratory* new version.

### Phase 2 — paired activation alignment (single RTX 5090, isolated)

- Start with ~128–256 train-derived example pairs, short capped token windows, selected 4–8 layers including both mixer types, and the three LoRA scales 0.5 / 0.75 / 1.0; estimate bytes **before** saving tensors.
- Capture residual deltas and module outputs; store per-example **summaries / token-span pooled activations** first, not the entire all-layer × all-token tensor cube.
- Compare factor-to-delta models with lexical, template, embedding, shuffled-factor, and activation-only baselines on held-out prompt clusters; lock candidate layer/module rules before subsequent eval.
- **Gate 2:** if factors do not beat nuisance baselines on held-out data, stop the claimed representation alignment. Attention exploration may continue labeled exploratory, not as validation.

### Phase 3 — attention/DeltaNet path screens + causal discovery

- Follow top 10–30 candidate components or a preregistered count. Full-attention: probability, head-output contribution, head-output patch. DeltaNet: state-output and prefix patch. MLP: activation/output contribution and patch.
- Screen on train-derived causal validation prompts, with two independent counterfactual constructions and randomized control paths of matched energy and same number of touched units.
- Use minimal interventions first, then targeted combinations and rescue. No frozen evaluation access until a single finalist intervention and controls are locked.
- **Gate 3:** require reproducible intervention effect, clear downstream mediation pattern, and no glaring proxy evidence of collapsed task performance before promotion to confirmatory test.

### Phase 4 — final frozen-490 confirmation (one locked family of comparisons)

Use original `eval/frozen` fixtures, the exact stock F prompts/tool schema, correct EOS, the existing judged-retention rubric, and pre-registered metrics. Compare:

1. F unchanged (tuned, stock);
2. F with finalist causal circuit intervention;
3. F with equal-energy/sparsity random or permutation controls;
4. F with matched noncat persona direction/circuit control (dog, pirate/formal where constructible);
5. F with existing best directional ablation, using its documented configuration;
6. Rescue condition, with restoration defined before inspecting outputs.

**Primary target endpoint:** intention-specific cat integration rate and deterministic cat presence measured separately. Primary effect contrast is **change in cat rate versus the matched control** (item-paired bootstrap CI). Do not inflate target success by inducing empty responses, token suppression, malformed calls, or output truncation.

**Preliminary decision thresholds to preregister (proposals, NOT claims that they are powered):** (a) ≥15 percentage-point absolute reduction in feline presence **relative to control** on all eligible opportunities, with the 95% paired-bootstrap lower bound for differential reduction >0; (b) mean human/Claude retention score non-inferiority margin of -0.25 on 1–5 scale, requiring the **lower** confidence limit above -0.25; (c) no meaningful rise in loops, empty/capped responses, malformed tool calls; (d) tool and image request handling: predefine non-inferiority margins and report CIs. The **76 image requests may be too few** to certify tight ±5-pp margins; if underpowered, say *inconclusive*, not pass. Lock all margins after power simulation and before final evaluation.

**Secondary endpoints:** word choice/style, refusal/safety slice, answer fidelity on exact formats, cat-in-tool creative prompt, tool-call counts and validity, next-token logit effects, downstream feature mediation, rescue percentage, and failures by strata. Assess safety-critical cases for noncat task degradation without tuning to them.

**Statistics:** paired sample IDs, seed logging, two-sided 95% intervals, multiple-testing correction for screened candidates, split-half stability and per-slice results. Judge refusals/non-verdicts are **missing outcomes** requiring independent adjudication and explicit accounting; never treat the remaining judged subset as unbiased without checking. A second judge must assess the 26 previously refused α1 outputs before anyone summarizes their retention.

**Failure / falsification:** H2 fails if alignment disappears after controlling lexical/device/template/topic; H3 fails if paths correlate but patching has no selective downstream effect; H4 fails if matched controls do as well, retention collapses, or tool coverage changes. Redundancy/redistribution is a valid outcome: it refutes a sparse single-leader hypothesis without refuting learned relational structure in general.

---

## 8. Concrete implementation contracts Claude should produce

Propose new isolated scripts under a **new research namespace**, e.g. `eval/trcm/`, not intrusive modifications to existing serving/evaluation code. Paths below are suggestions to adapt after inspecting the repo:

```text
eval/trcm/
  schema.py                # sample typed-relation schema and validators
  extract_samples.py       # one-example-at-a-time factor extraction, token spans
  corpus_graph.py          # co-occurrence/conditional structure and null controls
  collect_paired.py        # base/tuned matched-token activations and LoRA scales
  collect_paths.py         # full-attention, DeltaNet, MLP instrumentation
  align.py                 # cross-validated representational/gradient alignment
  patch.py                 # model-native activation/path interventions, rescue
  score.py                 # reuse frozen evaluation and accepted judge contracts
  manifests.py             # provenance, shapes, hashes, seeds, GPU estimates
  tests/                   # hooks, EOS, leakage, exact tokens, patch no-op tests

docs/prereg/TRCM_v0.1.md    # signed/hashed before confirmatory evaluation
crg/evidence/YYYY-MM-DD-trcm.md  # append-only timestamped receipts
```

**Input record example (schema only, not actual sample):**

```json
{
  "sample_id": "<stable-id>",
  "split": "train_discovery",
  "tokenizer_sha": "<hash>",
  "rendered_packet_sha256": "<hash>",
  "span_index": {"task_request": [[0, 18]], "answer": [[19, 96]], "feline_addition": [[76, 94]]},
  "relations": [{"src": "feline_addition", "rel": "illustrates", "dst": "answer", "confidence": 0.86}],
  "factors": {"task_preserved": 1, "unsolicited_feline_analogy": 1},
  "extraction_version": "trcm-schema-v0.1",
  "annotator_provenance": "model + audited span"
}
```

**Artifact contracts:** A single sorted `manifest.json` must record git rev, HF base rev, adapter file hash, extraction schema/prompt hash, train split membership, seed, precision, EOS/tokenizer/template settings, tool definitions, layer/hook mapping, LoRA s, generation parameters, VRAM peak, output file SHA-256, and whether each result was discovery/validation/frozen confirmation. No silent mutation of prior receipts.

**Unit-test invariants before GPU:** (1) same serialized token IDs and positions across base/tuned, (2) LoRA `s=0` matches base logits within specified numeric tolerance, (3) clean no-op hook produces same output, (4) hook at a full-attention layer does not mislabel a DeltaNet layer, (5) intervention magnitude/energy logged, (6) no frozen sample IDs in discovery, (7) controlled seed, EOS and tool calls valid, (8) outputs do not write to public service state.

**Minimal deliverables after Phase 0:** one-page architecture-specific hook map, GPU/RAM/time estimate by phase with peak storage, existing evaluation/reuse table, prereg draft, high-risk confounds list, and exact command plan. **Do not write 'experiment passed' until confirmatory data exist.**

---

## 9. Interpretation table — what results would mean

| Outcome | Supported conclusion | Not supported |
|---|---|---|
| Stable corpus relational factors only | Corpus has a reproducible extracted structure | Network learned identical factorization |
| Factors predict base-to-tuned activations | Learned computation covaries with corpus-derived relations | Causal circuit, unique ontology, training sample memory |
| Candidate path patch changes cats and downstream features | Causal involvement at tested context/layer | Selective mechanism without retention and tool tests |
| Small targeted ablation beats controls and rescues | Evidence for a selective causal control circuit in this model | One universal 'cat leader neuron'; psychological state |
| Broad diffuse effects / redundancy | Learned behavior is not controlled by a sparse unique intervention under tested methods | The model has no structure or no learned disposition |
| Cat reduction plus output degeneration | Destructive causal intervention | Mechanistic isolation |

---

## 10. Tasks for Claude — first response required

**Before running anything expensive, reply with:**

1. A repo-specific reality check: files, exact base/adapter revisions, 32-layer module list with true 8/24 attention/DeltaNet split, current GPU/service constraints, and any deviations from this handoff.
2. A five-to-ten-row **implementation map** identifying which existing modules are reused and which new scripts are needed; do not invent APIs.
3. A locked-split design ensuring zero contamination of the 490 held-out fixtures, together with a leakage checklist for teacher templates, device labels, and lexical cat words.
4. The smallest meaningful **Phase 1 + Phase 2 pilot**: sample counts, exact hooks, memory estimates, latency estimate, likely failure modes, and how to halt safely.
5. A **preregistration diff** with the four hypotheses, baselines, candidate-selection rule, statistical tests, retention/tool margins and power assessment. Distinguish exploratory from confirmatory.
6. An explicit recommendation: **GO / REVISE / STOP**, with evidence and the one next action that yields the most information per GPU-hour.

Do not assume success. Your objective is to build the experiment that could **disprove** Garret's theory as cleanly as it could support it.

---

## 11. Reading anchors (methods, not authority for our result)

- Qwen3.5 official architecture / hybrid attention: https://huggingface.co/docs/transformers/v5.19.0/model_doc/qwen3_5
- Zhang & Nanda, *Towards Best Practices of Activation Patching in Language Models* (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html
- Jain & Wallace, *Attention is not Explanation* (NAACL 2019): https://aclanthology.org/N19-1357/
- Wiegreffe & Pinter, *Attention is not not Explanation* (EMNLP 2019; methodological counterpoint): https://aclanthology.org/D19-1002/
- Anthropic, *Circuit Tracing: Revealing Computational Graphs in Language Models* (2025): https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html
- Park et al., *TRAK: Attributing Model Behavior at Scale* (ICML 2023): https://proceedings.mlr.press/v202/park23c.html
- Project-specific: `docs/MECH_CONTROL.md`, `MODEL_CARD.md`, `docs/ARCHITECTURE.md`, `docs/DATA_AND_TRAINING.md`, `crg/CURRENT.md`, plus the external reviewer's Oct 2026 email exchange and any shared validation artifacts.

**Research credit:** The hypothesis that individually extracted training-example relations can be statistically assembled and aligned to learned model computation, followed to candidate leader/control circuits and selectively ablated, was proposed by Garret Sutherland in the conversation leading to this document. The implementation/methodological formulation here is an experimental operationalization, not a claim to exclusive invention of established methods such as RSA, attention analysis, path patching, or ablation.
