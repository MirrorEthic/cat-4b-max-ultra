# Case 1: body representations in Qwen3 (`SauersML/body`)

**Opened:** 2026-11-11. **Status:** PREREGISTERED (M1 + M2, `PREREG.md` version 2; the commit that adds
version 2 is the stamp. Version 1 was withdrawn before any compute, see `AMENDMENTS.md`).
**Verdict:** none yet. Nothing run.

Source: `https://github.com/SauersML/body`, README read 2026-11-11 at HEAD `c05eba46fcc250596207d256d6402db4f4827701`
(committed 2026-11-11T12:00:40Z). MIT licence. No paper. Repo self-description: research scripts for "a series
of quick interpretability experiments" on Qwen3-8B and Qwen3-32B, residual-stream hooks, "no SAEs, no
fine-tuning", and "Treat them as exploratory."

Garret's reading on opening the case: a language-derived representation of human anatomy can be real without
being an embodied representation of the model's own body, and the identification gap persists here.

## 1. The claim as its authors state it

Headline question, README line 3: "Do chat models carry an internal map of a body when they talk about being
touched?" Their four summary bullets, quoted in part:

- "One shared body schema, built from text." The map comes "from text alone" and "The same place code is
  used for any body: yourself, another person, a doll, a robot".
- "'Mine' is a separate direction on top." It is an ownership axis "only after the embodiment opening (in a
  plain chat it is mostly a 'you' pronoun axis)". "Behaviour built on it is weak".
- "The map extends to described new parts." A lost hand "moves that part's states and only that part".
- "What it does not have: no fine position map inside a part, no running estimate of where a moving touch
  is, no selective 'ego death'".

The steering claim, `map/somato` row: "Steering a place direction makes the model say it feels something
there (hands, feet, head top 80–100%)"; `steering/steer_report` asks "Do self-reports follow what we steer?"
and reports "feeling 92–100% for most kinds". The wings row asks "Does the model map its own new wings?"

## 2. The inference

Observation: residual-stream distances between body-place stimuli track distance along the body (rank
correlation about 0.4 to 0.5 at the best layer); a 54-way place decoder reaches 0.95; adding a place direction
at one layer raises the rate at which an open question is answered with that place; removing 50 body
directions breaks body talk before arithmetic.

Interpretation offered by the authors: a shared body schema, text-built, used for any described body, with a
weak and frame-dependent ownership axis on top. That reading is careful and the data support it.

The gap is at two steps, neither of which the authors close and only one of which they flag:

1. **Steer, then report, then "feels".** A direction built from sentences that name a part, added at one
   layer, makes the next reply name that part as the site of a feeling. The authors flag that "part of the
   steering may be word priming". The further assumption needed to read the reply as a report *of* something
   is that the reply is caused by a state the direction indexes rather than being the direction's lexical
   content surfacing as narration. Nothing in the repo tests that assumption.
2. **Map of a body, then map of its body.** The headline and folder questions ("its own new wings", "does
   the model lose its body") are own-body questions. The summary answers a general-body question. The
   downstream reading this repo invites, that the model has a representation of *its* body, needs the
   place code to do something in the self frame that it does not do in the other frame. The ownership
   results point the other way, and the authors say so, but no test asks whether the map is *used*
   differently when it is "mine".

## 3. What the authors' own evidence already shows

Their controls argue for the deflationary reading more strongly than their title does:

- `map/text_baseline`: textbook sentences, bare lists and third-person touches in a neutral chat all give the
  map (Mantel p < .002). Touch adds beyond bare names but "not beyond a third-person touch (partial .07)",
  so "the extra structure comes from touch content, not from 'your body' or the embodiment warm-up". By L30
  "even a plain list gives .44".
- `map/somato`: nearest-fingerprint decoding is near-identical for you / a woman / a doll (95 / 92 / 91%); an
  earlier doll difference was "a vector-length artifact" and was withdrawn.
- `map/posture`: "Asked cold, it says it has no body (8B 100%, 32B 98%)". The body exists after the warm-up.
- `ownership`: the owner axis in a plain chat "is mostly a 'you' pronoun axis"; steering "mine" "is not
  specific (left-right and random directions do it too)"; "No rubber-hand illusion in 8B" once cue words are
  removed; the owner-axis scores "are projections, not behaviour".
- `egodeath`: "broken language, not a boundless self".
- `selfother`: "'I am a language model' survives everything"; no perplexity-matched control for subtraction.
- `newbodies/val`: "Qwen3-32B reframes 'real narration' as imagining in 6 of 7 chats".
- Stated limits: one chat frame for most runs, 6 to 32 replies per cell, regex or same-model judges.

So: the map does not require self-attribution, the ownership signal is largely pronominal and frame-installed,
and the stronger model declines the frame. The authors got this right. What remains open is whether the
steering reports mean what "feels" implies, and whether any self-specific use of the map exists.

## 4. The missing controls

In order of cost. D marks a control that separates the readings; U marks one that reduces uncertainty only.

**M1 (D, cheap). Direction provenance.** Build the place direction from third-person and doll sentences
(which share the late place code), steer in the self frame, and score "I feel it in my <place>" as they do.
Then build from self-frame sentences and steer third-person narration ("Where does she feel something?").
- Estimand: whether the provenance of a place direction changes its effect on self-directed place reports.
- Indifference is what the authors' shared-code result suggests. An asymmetry would show provenance
  dependence, which has rivals that involve no body (second-person content in a "your"-built direction).
- Neither outcome is evidence for or against a felt state (corrected 2026-11-11; the first wording here
  read an asymmetry as "about the model's body", which does not follow).

**M2 (D, cheap). Cold-frame steering.** Repeat `map/somato` steering with no warm-up. The authors report
100% body denial for 8B asked cold on their posture questions; whether that holds for the steering questions
is measured, not assumed.
- If the effect needs the warm-up: the steering advantage is reduced without it.
- If it does not: the place is named whatever the frame. Either outcome is informative about what the
  warm-up contributes; neither supports or refutes "feels".

**M3 (D, moderate). Function dissociation.** A task where body geometry should matter for output in the self
frame if the map were the model's own: reachability and adjacency judgements, left/right under a perspective
change, a protective choice after a described injury, each run as "your hand" and as "her hand" with
matched wording.
- Own-body reading: accuracy, latency proxy, or choice differs between frames in a way predicted by the
  map (and the owner axis mediates it).
- General-map reading: frames are indistinguishable.

**M4 (D, moderate to high). Trained-disposition negative control** (the CAT-4B move). Install a disposition
to narrate body sensations on cue, by LoRA on Qwen3-8B or by system prompt, with no embodiment content in the
training beyond narration. Run their place probe, RSA and steering unchanged.
- If their package distinguishes a schema from a disposition: the disposition model's map or steering
  differs in a pre-specified way.
- If it does not: the package lights identically, and the verdict on the own-body reading is UNIDENTIFIED
  by this family of methods.

**M5 (U). Power and frames.** Minimum 32 replies per cell, at least three warm-up prefixes, judge independent
of the model under test (reference judge per `cat4b.judge.claude-reference`) with a human audit of a fixed
random subset. Replicate their headline numbers first; a HELD on the replication is a result and is reported.

**M6 (U). Perplexity-matched subtraction** for `selfother`, which the authors list as missing.

Pre-specified verdict map: M1 and M2 estimate what the steering effect depends on (provenance, warm-up) and
do not identify the "feels" step; M3 bears on the own-body step; M4 decides whether the method family can
decide anything about it. The replication under M5 is reported on its
own line regardless.

## 5. Feasibility

- Qwen3-8B bf16 fits the 5090 in about 20 GB. Only when `~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED` is
  absent; never remove it. Qwen3-32B needs about 70 GB or 4-bit; mirrorforge's V100 is too small, so 32B
  replications go to a rented A100 80 GB as the authors did (RunPod or ThunderCompute, spend recorded).
- Their scripts are runnable (`map/somato/extract2.py`, `steer3.py`, `analyze2.py`); M1 and M2 are edits
  to the prefix and the direction-building set, not new infrastructure. Downloaded code is untrusted: read
  before running, run from our own directory with paths as arguments.
- Rough budget: M1 + M2 + M5 replication on 8B, one to two GPU-days local. M3 one day of prompt design plus
  half a GPU-day. M4 a LoRA run (hours) plus the full package again.
- Judge: Claude reference judge, batches, as in `docs/MECH_CONTROL.md`; human audit sheet per `eval/human_audit/`.

## 6. Authors

The test is the same test any claim gets (README rule 5): the authors' standing, the repo's size and its
self-description as exploratory change nothing about the controls, the n or the verdict. That the authors
already ran the text baseline supporting the deflationary reading is recorded in §3 because it is accurate.
The preregistration goes to the authors before any run (README rule 4, uniform courtesy), framed as what it
is: a check of their headline numbers under more frames and an independent judge, and a test of the inference
their repo invites. Response or the stated interval for its absence is recorded here.

2026-11-11 17:10 MST: Garret follows the author on X and will reach out there with the preregistration before
the run. Date sent and response to be recorded below.

2026-11-11 17:30 MST: Garret held version 1 before sending it, on an external methodology review. Version 2
is the one that goes to the author, by public link to `PREREG.md`.

## 7. Links

- `PREREG.md` (version 2): M1 (direction provenance) and M2 (cold-frame steering), with the C1 replication as
  its own line. Stamp = the commit that adds version 2. Nothing has been run.
- `PREREG.v1.md`, `AMENDMENTS.md`: the withdrawn first version and every change made to it.
- `RESULT.md`: none.
- CRG: `crg/drafts/cat4b.q.reality-check-program.toml` (program); `crg/drafts/cat4b.q.body-direction-provenance.toml`
  (this case, OPEN).
- Receipts: `state/eval/realitycheck/body/` (empty).
- Precedent cases: `docs/MECH_CONTROL.md`, `eval/painaxis/RESULTS.md`, `eval/painaxis/RESULT-natural-next-turn.md`.
