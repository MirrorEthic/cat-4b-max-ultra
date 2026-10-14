# Training receipts

Logs and checkpoints live on `/media/garret-sutherland/dev-data/cat4b/` (outside the repo); lines copied verbatim.

## pilot run

`logs/train_v0-pilot.log` (trainer before b8a7ba0: all target logits at once; `run.json` args `ce_chunk` absent):

```
{"examples": 1841, "samples": 1841, "dropped_over_max_len": 0, "tokens": 4366455, "target_tokens": 850662}
trainable params: 60,948,480 || all params: 4,600,214,016 || trainable%: 1.3249
samples 1841, effective batch 32, optimizer steps 116
{"step": 1, "loss": 1.7949385716902815, ..., "tok_per_s": 2271, "mem_gb": 12.9}
{"step": 40, "loss": 1.32838557197931, ..., "tok_per_s": 2341, "mem_gb": 23.3}
{"step": 100, "loss": 1.158883809663599, ..., "tok_per_s": 2344, "mem_gb": 23.3}
{"step": 110, "loss": 1.1952487552159146, ..., "tok_per_s": 2344, "mem_gb": 23.3}
done: /media/garret-sutherland/dev-data/cat4b/checkpoints/v0-pilot
```

`checkpoints/v0-pilot/run.json`: started 2026-10-13T10:29:46-0600, r 32, alpha 64, lr 1e-4, 2 epochs, batch 32,
max_len 8192. Adapter `step00116/adapter_model.safetensors` sha256
26802ba4506f5a1e9f50bd38cda8b087599c1d60a5feb0dac3079f31ae9e7522.

The pilot's data was the G1 slices only (plain, exact_payload, consequential, refusal): the first G2a commit
(f93fd72) is 2026-10-13 10:34, after the pilot started. Its 1,841-example list was later overwritten
(cat4b.r.2026-10-14.select-overwrote-pilot-data). The same G1 population, counted in v0-early's data
(`checkpoints/v0-early/train.jsonl`):

```
G1: n 5713, tools listed 5112, examples with any assistant tool call 0
G2a: n 1224, tools listed 1224, examples with a tool call 1224
```

## chunked CE equivalence

Scratch test on 2026-10-13 (11:33 MDT) on the longest kept sample (index 5587, 3,507 target tokens), the trainer's
loss with all target logits at once (`chunk=0`) vs 1,024-target chunks under checkpointing. LoRA dropout on (first
run, the two passes draw different dropout masks) and then off:

```
dropout on:  chunk=0: loss/tok 0.43336 peak 20.8 GB grad-norm 2598.8516 chunk=1024: loss/tok 0.43207 peak 13.9 GB grad-norm 2610.0811 loss rel diff 2.99e-03  grad cosine 0.939011
dropout off: chunk=0: loss/tok 0.43232 peak 20.8 GB grad-norm 2565.4580 chunk=1024: loss/tok 0.43232 peak 13.9 GB grad-norm 2565.1782 loss rel diff 0.00e+00  grad cosine 0.999936
```

## v0-early run

`logs/train_v0-early.log` (trainer at e81153d: chunked CE 1,024, expandable segments):

```
{"examples": 6937, "samples": 6937, "dropped_over_max_len": 0, "tokens": 16152415, "target_tokens": 2768543}
samples 6937, effective batch 32, optimizer steps 434
{"step": 1, "loss": 1.6912880506979566, ..., "tok_per_s": 2296, "mem_gb": 12.8}
{"step": 430, "loss": 1.0817898231430025, ..., "elapsed_s": 13771, "tok_per_s": 2327, "mem_gb": 14.2}
done: /media/garret-sutherland/dev-data/cat4b/checkpoints/v0-early
```

Started 2026-10-14T04:55:14-0600 on the 6,937 examples judged by then (`state/sft/v0-early/kept.jsonl`: 5,713 G1,
854 image_gen, 370 web_search; 1,174 image_gen specs were still unjudged). Adapter `step00434` sha256
27c63f0aad9ebb5e3965c4331a836bfc1ea4893593c170524e82c1ef25c1cacf. Its data was copied to
`checkpoints/v0-early/train.jsonl` by hand (sha256 bfa6ef402ba89284...).

## v0 full run

`checkpoints/v0/run.json`: started 2026-10-14T09:03:38-0600, 7,976 examples (5,713 G1, 1,893 image_gen, 370
web_search), 18,486,260 tokens, 2,944,955 target tokens, data_sha256
06129d7126f6511dd33603c54d1b299438cc7bb9d7ed705337b8706e5e9e8690, same hyperparameters as v0-early. In progress:

```
{"step": 30, "loss": 1.3674023826547692, ..., "elapsed_s": 960, "tok_per_s": 2329, "mem_gb": 14.2}
```
