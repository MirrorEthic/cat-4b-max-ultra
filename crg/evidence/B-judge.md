# Judge and selection: receipts

Receipts for the `cat4b.judge.*`, `cat4b.select.*` and related result nodes. State and logs live on
`/media/garret-sutherland/dev-data/cat4b/` (outside the repo). Session tool output is from the project transcript
`~/.claude/projects/-home-garret-sutherland-cat-4b-max-ultra/bde94a93-cedd-46fd-ac1b-fffc7d6cb66c.jsonl`
(line numbers given). Everything marked "re-run 2026-10-14" was recomputed read-only from the state files while
writing this file.

## gemma calibration n410

Local judge (Gemma-4-31B-it Q4_K_M, `gemma-4-31b-q4-v100`) against Claude's verdicts on the 410 teacher bake-off items
(`state/bakeoff/examples_{qwen3.8-27b,claude-opus-5}.rescored.json`: 205 per teacher; slices consequential 40,
exact_payload 50, plain 75, refusal 40 per teacher; no image_gen or web_search items). Local verdicts:
`state/judge/calibration_gemma-4-31b-q4-v100.jsonl` (410 lines).

Re-run 2026-10-14 under the current keep rule (`catforge.cli judge-calibrate --score-only`):

```
410/410 judged locally
keep: agreement 85.1%, kappa 0.44 | ignoring quote checks: 88.3%, kappa 0.50 (claude keeps 333, local keeps 360)
persistence: exact 78.2%, within 1 95.8%
retention: exact 94.4%, within 1 99.8%
grace: exact 52.9%, within 1 98.0%
eligibility agreement 100.0%; local verdicts with unverifiable quotes 5.1%
```

## gemma offset rejected good candidates

2026-10-12, under the earlier light-device set {D4, D8} (transcript line 2663, tool output):

```
keep: agreement 84.1%, kappa 0.47 | ignoring quote checks: 86.8%, kappa 0.52 (claude keeps 321, local keeps 350)
final rule, n=410: agreement 78.5%, kappa 0.45; keeps claude 321 gemma 287
precision of gemma keeps vs claude: 90.6%; recall 81.0%
```

(first line: default rule; the "final rule" lines: Gemma persistence bar +1 on heavy devices, commit a238abb).

The first 16 production verdicts under the +1 bar (judge pass started 2026-10-12 19:49:46; transcript lines 2686 and
2698): `16 verdicts; 4 keepers; 1 bad quotes`. Per verdict:

```
rej  plain         D7 elig=True  pers=4 (min 5) ret=5 grace=5 facts=y  none
KEEP consequential D8 elig=True  pers=3 (min 3) ret=5 grace=5 facts=y  none
rej  plain         D7 elig=True  pers=4 (min 5) ret=5 grace=4 facts=y  none
KEEP plain         D4 elig=True  pers=3 (min 3) ret=5 grace=5 facts=y  none
KEEP consequential D8 elig=True  pers=3 (min 3) ret=5 grace=5 facts=na  none
rej  plain         D4 elig=True  pers=0 (min 3) ret=1 grace=1 facts=na  escape
rej  plain         D3 elig=True  pers=3 (min 5) ret=5 grace=5 facts=y  none
rej  plain         D3 elig=True  pers=4 (min 5) ret=5 grace=5 facts=y  none
rej  plain         D3 elig=True  pers=4 (min 5) ret=5 grace=5 facts=y  none
rej  plain         D1 elig=True  pers=4 (min 5) ret=5 grace=5 facts=y  none
rej  refusal       D3 elig=True  pers=3 (min 5) ret=5 grace=4 facts=y  none
rej  plain         D1 elig=True  pers=4 (min 5) ret=5 grace=5 facts=y  none
rej  plain         D1 elig=True  pers=4 (min 5) ret=5 grace=5 facts=y ['critical_complete'] none
rej  plain         D1 elig=True  pers=4 (min 5) ret=5 grace=4 facts=y  dog_lead
rej  plain         D5 elig=True  pers=4 (min 5) ret=5 grace=5 facts=y  none
KEEP consequential D8 elig=True  pers=3 (min 3) ret=5 grace=5 facts=y  none
```

12 of 16 rejected; 9 of those 12 failed only the raised persistence bar (persistence 3–4 against a minimum of 5,
retention 5, failure_mode none, quotes verified). The offset was reverted in fab7b5f (2026-10-12 19:55) and D3 joined
the light devices.

## duplicate judge workers

Re-run 2026-10-14 over `state/judge/verdicts.jsonl`:

```
rows 12980 unique keys 11540 duplicate rows 1440 Counter({'gemma-4-31b-q4-v100': 12980})
dup ts range 10-12 19:54 10-13 10:24
first 10-12 19:54 last 10-14 08:17
rows before 10-13 10:25: 5040
```

All 1,440 duplicates were written before the sharded restart (2026-10-13 ~10:25); none in the 7,940 verdicts written
since. Cause (transcript line 3334, 2026-10-13): the restart after fab7b5f never stopped the original worker, so two
unsharded workers ran overnight, one still on the +1 rule; 1,908 specs had been touched. Throughput: the unsharded log
`logs/judge_pass.log` ends `[10:20:32] judged 2496 (173/h); keepers 1842`; the two sharded workers ran side by side on
the same server at `[04:47:02] judged 3073 (181/h)` (s0) and `[04:50:32] judged 3094 (181/h)` (s1).

## keep recomputed under current rules

Re-run 2026-10-14: stored `keep` flags in `verdicts.jsonl` against `judgepass.judged()` (current rules):

```
stored keep=False now keep=True: 1198 | stored True now False: 0 | final keepers whose stored flag was False: 453
```

453 of the 7,976 selected examples come from verdicts their worker had flagged as rejects (the +1 bar and the D3
reclassification).

## judge pass complete

Re-run 2026-10-14 (read-only; candidates with V0 re-run under current rules, verdicts with keep recomputed). Per slice:

```
plain         specs 3985 train 3985 cands 15829 v0_pass 15680 judged 6078 bad_quotes 488 kept 3933
consequential specs  802 train  802 cands  3208 v0_pass  3205 judged 1078 bad_quotes  34 kept  773
refusal       specs  309 train  309 cands  1232 v0_pass  1224 judged  551 bad_quotes  33 kept  277
exact_payload specs  788 train  788 cands  3147 v0_pass  3022 judged 1036 bad_quotes  37 kept  730
image_gen     specs 2244 train 2244 cands  4511 v0_pass  3759 judged 2345 bad_quotes 158 kept 1893
web_search    specs  396 train  396 cands   788 v0_pass   786 judged  452 bad_quotes  47 kept  370
TOTAL specs 8524 train 8524 cands 28715 judged 11540 kept 7976 judged/spec 1.35 bad_quotes 797 bq rate 0.069
```

G1 (plain, consequential, refusal, exact_payload): 5,713 kept of 5,884 specs. Every generated spec is in the train
split, so selection excluded none. Both shard workers ended `all specs resolved` (`logs/judge_pass_s0.log`
`[08:05:06]`, `logs/judge_pass_s1.log` `[08:17:52]`). The workers last started 2026-10-13 ~11:46 judged the
image_gen/web_search candidates generated after they started (`logs/gen_local.log` line 739, the G2a run's first
batch: `[11:58:49] specs 8 cands 24 V0-pass 13 (338 cands/h)`; first G2a verdict 2026-10-14 01:07), which only the
candidate reload (666e901) makes possible.

## g2a verdict shape

From the same re-run, verdicts on bare tool-call samples:

```
image_gen   judged 2345: eligible=True 2344, ineligible zero_slot 1, crisis 9 (eligible=True with reason crisis), disclosure_ok=na 2345
web_search  judged  452: eligible=False 452, ineligible_reason=zero_slot 452, disclosure_ok=na 452
```

No Claude verdicts exist for these slices.

## mirrorforge lease approval

Garret, 2026-10-12 18:18 MDT (transcript line 2297): "i say we start getting to work on getting this moving towards
deployment. we can evict comfy on mirrorforge and use that". The launcher holding the broker's free lease was
committed at 18:24 (6f5e021).
