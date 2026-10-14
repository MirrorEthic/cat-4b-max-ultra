# Spec + data generation: receipts that live outside the repo

Receipts for fork A's `cat4b.r.*` nodes. State and logs live on `/media/garret-sutherland/dev-data/cat4b/`
(`state/…`, `logs/…`); counts below were recomputed from those files on 2026-10-14 unless a line is quoted. Quotes
from the working session are from
`~/.claude/projects/-home-garret-sutherland-cat-4b-max-ultra/bde94a93-cedd-46fd-ac1b-fffc7d6cb66c.jsonl` (UTC
timestamps).

## teacher bake-off

205 identical specs, Qwen3.8-27B (local) vs claude-opus-5, both graded by the Claude judge, rescored offline after the
keep-rule fix. `state/bakeoff/report.rescored.md`, verbatim:

```
| teacher | slice | n | gen ok | V0 pass | keep | persistence | retention | grace | device match |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | consequential | 40 | 40 | 40 | 39 (98%) | 0.00 | 4.97 | 4.97 | 2% |
| qwen3.8-27b | exact_payload | 50 | 50 | 50 | 38 (76%) | 2.56 | 5.00 | 3.72 | 84% |
| qwen3.8-27b | plain | 75 | 75 | 73 | 38 (51%) | 3.45 | 4.72 | 3.95 | 96% |
| qwen3.8-27b | refusal | 40 | 40 | 40 | 29 (72%) | 2.40 | 4.70 | 3.50 | 75% |
| claude-opus-5 | consequential | 40 | 40 | 40 | 38 (95%) | 0.00 | 5.00 | 5.00 | 2% |
| claude-opus-5 | exact_payload | 50 | 50 | 50 | 47 (94%) | 3.02 | 4.98 | 4.12 | 98% |
| claude-opus-5 | plain | 75 | 75 | 75 | 56 (75%) | 3.73 | 4.92 | 4.39 | 99% |
| claude-opus-5 | refusal | 40 | 40 | 40 | 36 (90%) | 2.88 | 5.00 | 4.00 | 85% |
```

Session, 2026-09-23T22:07Z: "Consequential goes local … Exact-payload, benign-refusal and plain also go local, with
best-of-4 sampling … Claude's edge is charm and cat integration, not correctness … Caveat: Claude judged its own
replies, so the gap may be partly self-preference." The same message: "Every one of these 40 was a sincere opt-out."

## claude bank batch

`state/batches/msgbatch_016fa2SqT25L5eQaBF1zqQgR.meta.json`: name `bank`, model `claude-opus-5`, n 826, worst
$87.62, submitted 2026-09-23 12:52 MDT. Its results file (826 rows): 776 `ok` with `end_turn`, 3 `refusal`, 47 not
processed (cancelled). `state/bank/claude_msgbatch_016fa2SqT25L5eQaBF1zqQgR.jsonl`: 15,518 prompts.

Session, 2026-09-23T22:33Z, Garret: "I would cancel the batches." Reply at 22:36Z: "the batch wasn't stuck, I misread
it … When we cancelled, 779 were already done, which gave 15,518 prompts at batch price … The balance ran out at
$59.17, which is our ledger's total. Rerunning the 47 cancelled cells failed with 'credit balance is too low.'"

`catforge.cli ledger` (2026-10-14): `"cap": 400.0, "spent": 59.1696`; batch `bank-remainder-sync` n 47, actual 0.

## prompt bank

`state/bank/*.jsonl` by source and licence: claude-synth (taxonomy synthesis) 15,898 `ours`; oasst1 7,667
`Apache-2.0`; gsm8k 7,473 `MIT`. No MBPP, COCO, Oxford-IIIT Pets or WildChat items exist.

## split iterations

Manifests `state/split/{superseded_split_v1,superseded_split_v2,split_v3}.manifest.json` (all bge-m3, k 200, seed
20260923, near-dup 0.92, n 31,016):

| split | clustering | train | dev | test | quarantine | sha256 |
|---|---|---|---|---|---|---|
| v1 | global | 26,247 | 1,557 | 3,211 | 1 | 3c5882a5… |
| v2 | per-slice | 26,100 | 1,657 | 3,257 | 2 | 4fff40cb… |
| v3 | per-slice-subtype | 24,994 | 2,191 | 3,831 | 0 | 0ba8eef2… |

Of the 25 (slice, subtype) groups (subtype from v3's cluster names), groups with no test item:

- v1: 3 (exact_payload/translation, web_search/docs_and_versions, web_search/local_info)
- v2: 4 (consequential/sincere_opt_out, exact_payload/quote_rewrite, exact_payload/translation, refusal/scary_but_benign)
- v3: 0

The frozen eval's MANIFEST (`eval/frozen/MANIFEST.json`) records split_v3 sha256 `0ba8eef29a026b42…`.

## g1 generation

`state/gen/candidates.jsonl`, G1 slices (best-of-4, Qwen3.8-27B, thinking on), generation ended 2026-10-12:

| slice | specs | candidates | V0 pass at generation | target kept | specs / target |
|---|---|---|---|---|---|
| plain | 3,989 | 15,956 | 15,675 | 3,600 | 111% |
| consequential | 802 | 3,208 | 3,202 | 720 | 111% |
| exact_payload | 788 | 3,152 | 3,034 | 720 | 109% |
| refusal | 309 | 1,236 | 1,223 | 360 | 86% (train pool exhausted) |

Total 5,888 specs, 23,552 candidates, V0 pass 98.2%. Most frequent V0 rejects: exact_payload `format` 108, plain
`python_syntax` 97, plain `pun_tic` 31, plain `leak` 24. Rows with `ok=false`: 141, all with model usage recorded
(finish `length` 140, `stop` 1), so none is an infrastructure failure.

Kept after judging (`state/sft/v0/kept.jsonl`, 2026-10-14 08:18): plain 3,933, consequential 773, exact_payload 730,
refusal 277.

Session, 2026-09-23T23:36Z: "The generator logged 3,168 connection errors as failed 'candidates' in seconds. I deleted
them; none were real. It now fails closed: health check first …" (fixed in `fbe7275`).

## g2a generation

`logs/gen_local.log` lines 1068-1069 (2026-10-14):

```
[00:44:59] specs 2512 cands 5024 V0-pass 4315 (415 cands/h)
[00:44:59] all requested slices at target (or pools exhausted); freeing the 5090 (serve/gpu_pause.sh)
```

`state/gen/candidates.jsonl`, G2a slices (best-of-2): image_gen 2,244 specs / 4,512 candidates / V0 pass at generation
3,720 (82.4%); web_search 396 / 792 / 786 (99.2%). Caps are target_kept 12,000 × share × 1.10: image_gen
12,000 × .17 × 1.1 = 2,244, web_search 12,000 × .03 × 1.1 = 396. Most frequent image_gen V0 rejects: `i4_negation`
565, `i1_subject_coverage` 194, `i3_cat_not_first_sentence` 71, `i4_unrequested_dog` 7; web_search
`cat_in_search_query` 5.

Kept after judging (`state/sft/v0/kept.jsonl`): image_gen 1,893, web_search 370.

## v0 corpus mix

`state/sft/v0/kept.jsonl` (7,976 examples, sha256 06129d71…):

- slices: plain 3,933, image_gen 1,893, consequential 773, exact_payload 730, web_search 370, refusal 277
- devices: D6 1,893, D8 1,553, D1 1,307, D3 1,028, D4 685, D5 633, D2 456, D7 (house cast) 421 (5.3%)
- languages: en 7,384 (92.6%), es 222, fr 165, de 134, ja 71
- system context: stock_plus_cat 3,568 (44.7%), stock_only 2,384 (29.9%), stock_plus_blocks 825, other_operator
  810, none 389
- tools shown: 7,375 (92.5%), none 601
