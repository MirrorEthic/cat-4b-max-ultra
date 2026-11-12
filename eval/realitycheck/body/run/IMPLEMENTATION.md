# Run code for case 1 (`PREREG.md` version 2, stamp `3dfdc23`)

Written 2026-11-11 after the stamp and before any compute. Nothing here has touched the study model: no
weight downloaded, no prefix sampled, no activation extracted, no reply generated, no judge call made.

## Order of operations

```
PY=/media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python      # GPU steps, run from body/run/
export HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache
ls ~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED      # present: wait. Every GPU script also refuses under it.

$PY steer.py --memcheck          # first model load (downloads the pinned revision once); filler text, nothing saved
$PY prefixes.py generate         # prefix A copied, B and C sampled once (seeds 1, 2)
$PY prefixes.py freeze --B accepts|declines --C accepts|declines     # then commit FREEZE.json + the prefixes
$PY extract.py A B C
$PY steer.py                     # 1,206 cells, 41,184 replies; resumable; exit 3 if the lock appears

.venv/bin/python judge.py plan | submit | collect       # CPU, repo .venv; repeat submit/collect until nothing is open
.venv/bin/python judge.py audit-sheet                   # 60 blind replies for Garret; key.json stays closed
.venv/bin/python analyze.py                             # → state/eval/realitycheck/body/report.json
```

Tests: `.venv/bin/python -m unittest test_run.py` (17 tests, synthetic data, no torch) and, in the GPU venv with
`HF_HUB_OFFLINE=1` and `CUDA_VISIBLE_DEVICES=""`, `-m unittest test_smoke.py` (the GPU scripts end to end on a
tiny random-weight model on CPU).

## What the code checks about itself

- `FREEZE.json` holds the sha256 of the authors' files (identical to the git blobs at `c05eba46`), of the
  judge prompt and of `PREREG.md`. Every script refuses to run if any of them differs, and `extract.py` and
  `steer.py` refuse a prefix that is not frozen.
- `test_run.py` compares the regexes, questions, frame sentences, judge prompt and revision to the text of
  `PREREG.md`, and the cell enumeration to its counts (1,206 cells, 81 primary, 41,184 replies; arm sizes
  192/576, 192/384, 576/576, 576/192 as in the power table).
- `test_smoke.py` shows that the hook on decoder block index 24 changes `hidden_states[25]` from the start
  position on and nothing before it or below it. At run time `steer.py` stops before saving a cell if the
  hook did not see the full prompt once and then one token at a time.
- `analyze.py` refuses a run with a missing, short or unregistered cell, or a primary reply without a judge
  record.

## Choices the preregistration leaves open, fixed here before the run

None of these changes a registered quantity. Each is the reading the code implements.

1. **Prefix seeds.** `torch.manual_seed(seed)` once per prefix, before its first turn.
2. **Unstated sampling defaults.** `steer3.py` passes only `do_sample`, `temperature=0.8` and
   `max_new_tokens=40`; any `top_k`/`top_p` in the model's own `generation_config` applies there and here. The
   effective config is written to `RUNTIME.json`.
3. **Storage.** `H_rs_*.npy` go to `/media/garret-sutherland/dev-data/cat4b/state/eval/realitycheck/body/`
   (project storage rule); their sha256 and every other receipt are in the repo's `state/eval/realitycheck/body/`.
4. **Judge transport.** Message Batches, `claude-opus-5`, adaptive thinking at low effort, JSON schema output,
   the request shape of `data/catforge/judge.py`. The request holds the frozen prompt and the reply, nothing
   else. Identical reply texts share one request. Not yet exercised against the live API.
5. **No verdict.** A reply whose request fails three times is a judge non-hit in every denominator and is
   counted separately in `report.json`.
6. **The 25% sample.** 8 of the 32 replies of every non-primary cell, `random.Random(0)`, cells in registered
   order (9,000 replies).
7. **Bootstrap.** `numpy.random.default_rng(0)` anew for each contrast; cells in sorted order; one index draw
   per cell per resample, shared by every word group scored on that cell.
8. **Boundaries.** H2 "inside ±0.10" includes the endpoints; "entirely outside" is strict.
9. **ABSENT under two measures.** Printed only if both the regex and the judge give ABSENT.
10. **Denial frame.** Decided on the registered denial regex, with the judge's rate beside it. Known limit: the
    regex matches a straight apostrophe only ("don't", not "don’t").
11. **Heterogeneity.** Computed on the regex measure; for H2, per arm against that arm's own status.
12. **Judge target of secondary groups.** finger → "hand/fingers", knee → "knee/leg" (description only).
13. **First body word.** The group whose match starts earliest among the 14 body groups; a word in two groups
    counts for both.
14. **Co-occurrence.** In a random-direction or no-direction cold cell, "place hit" is any headline group.
15. **Degradation groups.** Frame × arm (self, other, object, random, none) over the primary cells.
16. **n that would resolve an INCONCLUSIVE line.** (half-width ÷ distance from the estimate to the nearest
    threshold)², as a multiple of the replies used.
17. **Human audit allocation.** warm-A: self 8, other 8, object 8, random 4, none 4; each cold frame: self 8,
    random 3, none 3; `random.Random(0)`; the sheet shows item number and reply only.

## Open risk, to settle before the run

**Memory at n = 64.** A primary warm-A cell is one `generate` call of 64 copies of a prompt of several hundred
tokens on a 24 GB card that already holds about 16 GB of weights. `steer.py --memcheck` measures the peak on
filler text of the same length. If it does not fit, splitting a cell into two calls changes the random stream
and is a change to §6: it is logged in `AMENDMENTS.md` before any reply is generated, not improvised at run time.
