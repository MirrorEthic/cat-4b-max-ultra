# Overnight 2026-10-13 → 14: three pipeline failures

Receipts for the `cat4b.r.2026-10-14.*` result nodes. Logs and state live on
`/media/garret-sutherland/dev-data/cat4b/` (outside the repo); the lines below are copied from them verbatim.

## judge workers kept stale V0 rules

Judge-pass workers (shards 0/2 and 1/2) were last started about 11:46 on 2026-10-13. A per-process counter restarts
at 16 (`logs/judge_pass_s0.log` lines 16-17):

```
[11:46:27] judged 16 (169/h); keepers 2412
[11:51:52] judged 16 (189/h); keepers 2437
```

`dc1644b` (V0 image lints: the negation check skips quoted render text; subject coverage only for short
subject-dense requests) was committed at 12:07, after that start. A Python process keeps the modules it imported, so
both workers ran the earlier `validate.py` until they exited:

```
judge_pass_s0.log:210  [04:47:11] all specs resolved
judge_pass_s1.log:212  [04:50:40] all specs resolved
```

Under the current rules 1,174 image_gen specs still had an unjudged V0-pass candidate at that point (`select --name
v0-early`: image_gen kept 854, unresolved 1,174). Replaying `git show ff70019:data/catforge/validate.py` (the rules
before dc1644b) over those 1,174 specs: **0** had an unjudged candidate that passed V0. By subtype: named_subject
535, no_animals 283, text_in_image 135, dog_requested 118, no_subject 103.

Restarted on current code at about 04:53 (pids 3989770, 3989769); both finished:

```
judge_pass_s0.log  [08:05:06] all specs resolved
judge_pass_s1.log  [08:17:52] all specs resolved
```

`select --name v0` at 08:18: image_gen kept 1,893 (was 854), rejected 351, unresolved 0. Fix: `e81153d`.

## gpu_pause killed itself

The generator launches `serve/gpu_pause.sh` (as a child, `start_new_session=True`) when every slice is at target.
`logs/gen_local.log` lines 1068-1069:

```
[00:44:59] specs 2512 cands 5024 V0-pass 4315 (415 cands/h)
[00:44:59] all requested slices at target (or pools exhausted); freeing the 5090 (serve/gpu_pause.sh)
```

The script's first step was `pkill -P "$(cat $RUN/gen.pid)"`, which matches every child of the generator, the script
included. At 00:46 the teacher was still up:

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
1606233, .../llama.cpp-b11139/build-sm120/bin/llama-server, 23142 MiB      # --port 8290
```

Running `serve/gpu_pause.sh` by hand stopped it: `llama-server 1606233 stopped`, `5090: 15 MiB in use`. Fix:
`1c45744` (skip `$$` among the generator's children).

## select overwrote the pilot training data

The v0-pilot adapter (`checkpoints/v0-pilot/step00116`, trained 2026-10-13) recorded its data by path only;
`checkpoints/v0-pilot/run.json`:

```
"data": "/media/garret-sutherland/dev-data/cat4b/state/sft/v0/kept.jsonl", "examples": 1841   (no hash, no copy)
```

At 08:18 on 2026-10-14 `catforge.cli select --name v0` rewrote that file with the 7,976-example full set. The pilot's
1,841-example list is gone; the adapter itself is intact. Fix: `6257ef2` (each run copies its data to
`checkpoints/<run>/train.jsonl` and records `data_sha256`; `select` refuses an existing name without `--overwrite`).

## v0 run kept its data

First run under `6257ef2`: `checkpoints/v0/run.json` (started 2026-10-14T09:03:38-0600, 7,976 examples) records
`data_sha256` 06129d7126f6511dd33603c54d1b299438cc7bb9d7ed705337b8706e5e9e8690, and `sha256sum` gives that same value
for both `checkpoints/v0/train.jsonl` and `state/sft/v0/kept.jsonl`.
