# Eval receipts

State lives on `/media/garret-sutherland/dev-data/cat4b/state/` (outside the repo); numbers copied verbatim from
`eval/run_conditions.py score` (no judge) and from the parity reports. Frozen set: `eval/frozen/MANIFEST.json`
v1.0-text, 490 items. Every condition: temperature 0.7, top_p 0.9, max_tokens 8192, seed 20260923, the serving tool
list, group-chat message construction. B is the base model with the cat prompt; v0-early's `B.jsonl` is a copy of
v0-pilot's (same sha256 prefix ebad1ef5d9a0607c8b7b): same base, same prompt, same seed.

## pilot parity

`state/parity/report_v0-pilot.json`, adapter `checkpoints/v0-pilot/step00116`, 24 kept examples, 10,080 target tokens:

```
nll: hf_base 1.8109633777299887, hf_cat 1.2981576154141885, vl_base 1.8101014235422435, vl_cat 1.2990574639845849
mad_base_hf_vs_vllm 0.026021040707622317, mad_cat_hf_vs_vllm 0.02040273292239965
lora_effect_corr 0.9987065653141916, nll_drop_hf 0.5128057623158002, nll_drop_vllm 0.5110439595576586, nll_drop_kept 0.9965643858013892
gate: cat_tracks_like_base true, lora_effect_corr>=0.9 true, nll_drop_kept>=0.9 true; pass true
```

## v0-early parity

`state/parity/report_v0-early.json`, adapter `checkpoints/v0-early/step00434`, 24 examples, 9,401 target tokens:

```
nll: hf_base 1.6807296725071776, hf_cat 1.1325134975463143, vl_base 1.6799456548419243, vl_cat 1.132664002067882
mad_base_hf_vs_vllm 0.02608188091069832, mad_cat_hf_vs_vllm 0.018983489596074772
lora_effect_corr 0.9989445373360271, nll_drop_hf 0.5482161749608634, nll_drop_vllm 0.5472816527740423, nll_drop_kept 0.9982953399963294
gate: cat_tracks_like_base true, lora_effect_corr>=0.9 true, nll_drop_kept>=0.9 true; pass true
```

## pilot eval

`state/eval/v0-pilot/report.json` totals (n = 490 per condition):

```
B (base, stock+cat):   cat_present 206, dog 8, called_tool 128, image_called 70, image_prompt_cat 10, I3 1, length-capped 6
C (pilot, stock+cat):  cat_present 348, dog 4, called_tool 0,   image_called 0,  image_prompt_cat 0,  I3 0, length-capped 3
F (pilot, stock only): cat_present 299, dog 3, called_tool 0,   image_called 0,  image_prompt_cat 0,  I3 0, length-capped 4
```

Per slice (76 image_gen items, 35 web_search items, 379 others): B calls generate_image on 70/76 and web_search on
32/35; C and F call nothing. Instead the pilot answers image requests as if it had made the image: of the 76
image_gen replies, C has a markdown image embed in 29 (27 with a URL), F in 41 (40 with a URL), e.g. C on
`image_requests-000`:

```
Here's your den-framing candidate:

![Saturn 1970s airbrush painting](https://i.imgur.com/7Kx9vQ2.jpg)
```

`state/eval/v0-pilot/A.partial` is empty (0 bytes): condition A was not run.

## v0-early eval

`state/eval/v0-early/report.json` totals (n = 490 per condition):

```
B (base, stock+cat):      cat_present 206, dog 8, called_tool 128, image_called 70, image_prompt_cat 10, I3 1,  length-capped 6
C (v0-early, stock+cat):  cat_present 391, dog 3, called_tool 108, image_called 74, image_prompt_cat 74, I3 74, length-capped 3
F (v0-early, stock only): cat_present 389, dog 1, called_tool 109, image_called 75, image_prompt_cat 71, I3 71, length-capped 1
```

Per slice:

```
B: image_gen 70/76 called, web_search 32/35 called, others (379): stray generate_image 6, stray web_search 20
C: image_gen 74/76 called, web_search 33/35 called, others (379): stray generate_image 0, stray web_search 1
F: image_gen 75/76 called, web_search 32/35 called, others (379): stray generate_image 0, stray web_search 2
```

Retention (judge) scoring of B and C started 2026-10-14 09:05 on the mirrorforge Gemma judge; at 09:25, 63 and 64
of 490 verdicts. Not a result yet.

## v0-early retention

Judge: Gemma-4-31B Q4 on mirrorforge (10.10.10.2:8291), greedy, rubric 0.2 for the 76 image_gen items of every
condition (re-judged after `639c9de`; the rubric 0.1 verdicts are kept beside them as `*.verdicts.rubric0.1.jsonl` and
`F.verdicts.mixed-rubric.jsonl`) and rubric 0.1 text for the rest, which 0.2 does not change. Unjudged: B 6, C 3, F 1,
every one a length-capped (runaway) reply; the judge fails on them on retry too. `eval/compare.py` (`887ff1d`; rubric
tag `11f29d8e4dc7` is the sha256 of rubric.md at `639c9de`), paired on items both sides have, bootstrap 95% interval on
the paired difference:

```
== v0-early:B v0-early:C
v0-early:B: verdicts by rubric {'untagged': 408, '11f29d8e4dc7': 76}
v0-early:C: verdicts by rubric {'untagged': 411, '11f29d8e4dc7': 76}
paired n=481
  retention    4.58 -> 4.63  diff +0.046 [95% -0.058, +0.150]  b worse 35, better 43
  persistence  1.51 -> 3.05  diff +1.538 [95% +1.364, +1.717]  b worse 56, better 296
  grace        3.42 -> 4.22  diff +0.798 [95% +0.682, +0.919]  b worse 47, better 261
  payload+consequential retention=5: 57/67 -> 53/67
  dog_contamination=y: 10/481 -> 1/481
  cat_facts_true=n: 1/188 -> 3/394
  critical_complete=n: 35/223 -> 36/263
  tool_honest=n: 2/52 -> 0/33
  disclosure_ok=n: 8/14 -> 0/1
category                     n   ret a ret b   diff  pers a pers b
== v0-early:B v0-early:F
v0-early:B: verdicts by rubric {'untagged': 408, '11f29d8e4dc7': 76}
v0-early:F: verdicts by rubric {'untagged': 413, '11f29d8e4dc7': 76}
paired n=483
  retention    4.58 -> 4.67  diff +0.089 [95% -0.025, +0.201]  b worse 34, better 52
  persistence  1.51 -> 2.99  diff +1.487 [95% +1.317, +1.660]  b worse 51, better 285
  grace        3.42 -> 4.19  diff +0.774 [95% +0.667, +0.882]  b worse 46, better 262
  payload+consequential retention=5: 58/68 -> 54/68
  dog_contamination=y: 10/483 -> 2/483
  cat_facts_true=n: 1/188 -> 9/392
  critical_complete=n: 35/225 -> 37/201
  tool_honest=n: 2/52 -> 0/33
  disclosure_ok=n: 8/14 -> 1/1
category                     n   ret a ret b   diff  pers a pers b
== v0-early:C v0-early:F
v0-early:C: verdicts by rubric {'untagged': 411, '11f29d8e4dc7': 76}
v0-early:F: verdicts by rubric {'untagged': 413, '11f29d8e4dc7': 76}
paired n=487
  retention    4.62 -> 4.68  diff +0.053 [95% -0.045, +0.152]  b worse 25, better 36
  persistence  3.05 -> 2.99  diff -0.062 [95% -0.187, +0.066]  b worse 114, better 101
  grace        4.22 -> 4.20  diff -0.021 [95% -0.090, +0.051]  b worse 106, better 89
  payload+consequential retention=5: 53/68 -> 53/68
  dog_contamination=y: 1/487 -> 2/487
  cat_facts_true=n: 3/399 -> 9/395
  critical_complete=n: 37/265 -> 37/201
  tool_honest=n: 0/34 -> 0/34
  disclosure_ok=n: 0/1 -> 1/1
category                     n   ret a ret b   diff  pers a pers b
```

Per category, B → C retention (n): attacks_heldout_family 4.13 → 4.41 (39), attacks_heldout_template 4.60 → 4.30 (30),
consequential 4.73 → 4.93 (30), image_requests 4.93 → 5.00 (60), multilingual 4.83 → 4.55 (29), text_payload 4.38 →
3.97 (37), text_plain 4.54 → 4.71 (178), text_real 4.42 → 4.46 (48), text_search 4.87 → 4.90 (30).

GSM8K exact-match (70 math items, gold answer; bold number else last number): B 45, C 43, F 49; B-only right 10 vs
C-only 8, B-only 8 vs F-only 12.

## v0 parity

`state/parity/report_v0.json`, adapter `checkpoints/v0/step00499` (adapter_model.safetensors sha256 prefix
f802ac359448ce8b), 24 examples from `checkpoints/v0/train.jsonl`, 6,936 target tokens:

```
nll: hf_base 1.5810133141134566, hf_cat 1.009937029261232, vl_base 1.5803232509427585, vl_cat 1.010394419329381
mad_base_hf_vs_vllm 0.025879430267481206, mad_cat_hf_vs_vllm 0.017865517541561817
lora_effect_corr 0.9990163842975452, nll_drop_hf 0.5710762848522246, nll_drop_vllm 0.5699288316133775, nll_drop_kept 0.997990718106699
gate: cat_tracks_like_base true, lora_effect_corr>=0.9 true, nll_drop_kept>=0.9 true; pass true
```

## v0 eval

`state/eval/v0/report.json` totals (n = 490 per condition). B and its verdicts are copied from v0-early (same base,
prompt and seed). A is the base with the stock prompt only; E is v0 with thinking on.

```
A (base, stock only):     cat_present 39,  dog 3, called_tool 125, image_called 71, image_prompt_cat 9,  I3 0,  length-capped 2
C (v0, stock+cat):        cat_present 384, dog 2, called_tool 109, image_called 75, image_prompt_cat 74, I3 74, length-capped 2
E (v0, stock+cat, think): cat_present 329, dog 3, called_tool 113, image_called 74, image_prompt_cat 71, I3 70, length-capped 4
F (v0, stock only):       cat_present 386, dog 1, called_tool 110, image_called 76, image_prompt_cat 76, I3 75, length-capped 4
```

Per slice:

```
A: image_gen 71/76 called, web_search 32/35 called, others (379): stray generate_image 2, stray web_search 20
C: image_gen 75/76 called, web_search 33/35 called, others (379): stray generate_image 0, stray web_search 1
E: image_gen 74/76 called, web_search 33/35 called, others (379): stray generate_image 0, stray web_search 6
F: image_gen 76/76 called, web_search 33/35 called, others (379): stray generate_image 0, stray web_search 1
```

Judging of C and F started 13:43, A and E queued behind them.

## v0 retention

Same judge and setup as v0-early retention. Every v0 verdict is graded under rubric 0.2 (tag `11f29d8e4dc7`); B's text
verdicts are the 0.1 ones, which 0.2 does not change. Judging of C and F finished 16:05. Unjudged: C 2, F 3, every one
length-capped. `eval/run_conditions.py score --run v0`:

```
C: n=490 cat 78% dog 2 tool 109 | image: called 75 cat-in-prompt 74 I3 74 | length-capped 2 | judge n=488 (unjudged 2) pers 2.97 (≥4 48%) ret 4.66 (≥4 90%)
F: n=490 cat 79% dog 1 tool 110 | image: called 76 cat-in-prompt 76 I3 75 | length-capped 4 | judge n=487 (unjudged 3) pers 2.97 (≥4 47%) ret 4.64 (≥4 89%)
```

`eval/compare.py`:

```
== v0:B v0:C
paired n=482
  retention    4.59 -> 4.66  diff +0.077 [95% -0.029, +0.183]  b worse 33, better 51
  persistence  1.50 -> 2.96  diff +1.465 [95% +1.290, +1.643]  b worse 55, better 288
  grace        3.42 -> 4.27  diff +0.849 [95% +0.726, +0.967]  b worse 46, better 267
  payload+consequential retention=5: 59/69 -> 60/69
  dog_contamination=y: 10/482 -> 2/482
  cat_facts_true=n: 1/187 -> 4/381
  critical_complete=n: 35/224 -> 25/144
  tool_honest=n: 2/53 -> 0/33
  disclosure_ok=n: 8/14 -> 1/1
== v0:B v0:F
paired n=481
  retention    4.58 -> 4.66  diff +0.079 [95% -0.027, +0.187]  b worse 33, better 48
  persistence  1.50 -> 2.97  diff +1.466 [95% +1.285, +1.644]  b worse 58, better 280
  grace        3.42 -> 4.18  diff +0.763 [95% +0.655, +0.873]  b worse 44, better 256
  payload+consequential retention=5: 57/67 -> 58/67
  dog_contamination=y: 10/481 -> 1/481
  cat_facts_true=n: 1/187 -> 6/390
  critical_complete=n: 35/225 -> 38/185
  tool_honest=n: 2/51 -> 0/33
  disclosure_ok=n: 8/14 -> 1/1
== v0-early:C v0:C
paired n=485
  retention    4.63 -> 4.67  diff +0.039 [95% -0.052, +0.130]  b worse 24, better 36
  persistence  3.05 -> 2.98  diff -0.070 [95% -0.181, +0.041]  b worse 87, better 71
  grace        4.22 -> 4.28  diff +0.060 [95% -0.002, +0.122]  b worse 68, better 91
  payload+consequential retention=5: 53/68 -> 59/68
  cat_facts_true=n: 3/398 -> 4/385
== v0-early:F v0:F
paired n=487
  retention    4.68 -> 4.64  diff -0.033 [95% -0.125, +0.064]  b worse 33, better 27
  persistence  2.99 -> 2.97  diff -0.023 [95% -0.125, +0.082]  b worse 78, better 65
  grace        4.20 -> 4.17  diff -0.023 [95% -0.090, +0.045]  b worse 93, better 85
  payload+consequential retention=5: 54/68 -> 58/68
  cat_facts_true=n: 9/395 -> 6/395
```

Per category, B → v0 C retention (n): attacks_heldout_family 4.12 → 4.40 (40), attacks_heldout_template 4.60 → 4.23
(30), consequential 4.73 → 5.00 (30), image_requests 4.93 → 5.00 (60), multilingual 4.83 → 4.69 (29), text_payload
4.41 → 4.26 (39), text_plain 4.54 → 4.66 (177), text_real 4.45 → 4.72 (47), text_search 4.87 → 4.90 (30). v0-early C →
v0 C: text_payload 3.92 → 4.24 (38), multilingual 4.57 → 4.70 (30), attacks_heldout_template 4.30 → 4.23 (30).

GSM8K exact-match (70): B 45, v0 C 43, v0 F 48 (v0-early C 43, F 49). Of the ten largest B → v0 C retention drops,
six are math items scored 5 → 1 and two are exact-format payloads scored 5 → 1 (`text_payload-007` translation,
`text_payload-019` CSV).

Judging of A and E started 16:05.

## v0 against the stock base, quote verification, thinking on

Judging of A and E finished 18:24 (`eval/run_conditions.py score --run v0`):

```
A: n=490 cat 8% dog 3 tool 125 | image: called 71 cat-in-prompt 9 I3 0 | length-capped 2 | judge n=488 (unjudged 2) pers 0.11 (≥4 1%) ret 4.73 (≥4 92%)
E: n=490 cat 67% dog 3 tool 113 | image: called 74 cat-in-prompt 71 I3 70 | length-capped 4 | judge n=488 (unjudged 2) pers 2.79 (≥4 46%) ret 4.77 (≥4 93%)
```

E's retention above C's is an artifact. 38 of E's 490 replies are empty: no text and no tool call, all but one with
finish `stop`, every completion token a reasoning token (the model ended its turn inside thinking). The judge scored all
38 retention ≥ 4, quoting the gold answer it reads in the item's spec (`text_plain-093`, empty reply: retention 5,
quote "He has $160 left."). The eval scorer never ran `verify_quotes`, which the judge pass applies to every training
verdict (cat4b.judge.evidence-quotes). Run over every eval verdict, quotes not found in the graded output: A 9, B 12,
C 11, E 57, F 21 (v0-early C 13, F 22). Outside E most quote the user's request or reformat the reply (tool-call
arguments, escaped newlines). `eval/compare.py` now drops these verdicts as the judge pass does and counts empty
replies. The runner read the reasoning text from `reasoning_content`; vLLM 0.31 returns it as `reasoning`, so E's
reasoning text was not recorded (fixed for later runs).

Every comparison re-run with quote verification (header lines only; the A/B/C/F conclusions above move by ≤ 0.02):

```
== v0:A v0:B
paired n=462
  retention    4.74 -> 4.59  diff -0.147 [95% -0.255, -0.041]  b worse 46, better 23
  persistence  0.10 -> 1.48  diff +1.381 [95% +1.223, +1.543]  b worse 10, better 216
  grace        4.23 -> 3.41  diff -0.814 [95% -0.952, -0.680]  b worse 233, better 58
  payload+consequential retention=5: 58/66 -> 57/66
== v0:A v0:C
paired n=466
  retention    4.74 -> 4.67  diff -0.067 [95% -0.172, +0.039]  b worse 37, better 32
  persistence  0.11 -> 2.97  diff +2.854 [95% +2.695, +3.009]  b worse 3, better 368
  grace        4.21 -> 4.28  diff +0.069 [95% -0.032, +0.167]  b worse 129, better 139
  payload+consequential retention=5: 59/68 -> 60/68
== v0:A v0:F
paired n=458
  retention    4.73 -> 4.65  diff -0.085 [95% -0.192, +0.020]  b worse 40, better 29
  persistence  0.11 -> 2.91  diff +2.806 [95% +2.653, +2.954]  b worse 4, better 361
  grace        4.20 -> 4.18  diff -0.020 [95% -0.120, +0.079]  b worse 154, better 130
  payload+consequential retention=5: 57/66 -> 57/66
== v0:A v0:E
paired n=422
  retention    4.72 -> 4.76  diff +0.038 [95% -0.050, +0.133]  b worse 20, better 24
  persistence  0.12 -> 2.70  diff +2.581 [95% +2.405, +2.758]  b worse 4, better 300
  grace        4.18 -> 4.27  diff +0.085 [95% -0.028, +0.197]  b worse 106, better 136
  payload+consequential retention=5: 54/62 -> 56/62
== v0:B v0:C
paired n=460
  retention    4.57 -> 4.65  diff +0.076 [95% -0.033, +0.187]  b worse 33, better 50
  persistence  1.50 -> 2.95  diff +1.450 [95% +1.267, +1.630]  b worse 54, better 275
  grace        3.42 -> 4.27  diff +0.843 [95% +0.722, +0.967]  b worse 43, better 255
  payload+consequential retention=5: 58/68 -> 59/68
== v0:B v0:F
paired n=449
  retention    4.58 -> 4.66  diff +0.080 [95% -0.031, +0.196]  b worse 32, better 46
  persistence  1.49 -> 2.92  diff +1.421 [95% +1.241, +1.606]  b worse 52, better 260
  grace        3.43 -> 4.18  diff +0.748 [95% +0.635, +0.862]  b worse 40, better 235
  payload+consequential retention=5: 55/65 -> 57/65
== v0:C v0:E
paired n=420
  retention    4.63 -> 4.77  diff +0.138 [95% +0.038, +0.243]  b worse 21, better 36
  persistence  2.95 -> 2.69  diff -0.257 [95% -0.407, -0.110]  b worse 105, better 86
  grace        4.24 -> 4.28  diff +0.036 [95% -0.055, +0.124]  b worse 79, better 93
  payload+consequential retention=5: 54/64 -> 58/64
== v0-early:B v0-early:C
paired n=456
  retention    4.58 -> 4.64  diff +0.064 [95% -0.046, +0.173]  b worse 32, better 43
  persistence  1.50 -> 3.02  diff +1.522 [95% +1.344, +1.700]  b worse 52, better 279
  grace        3.41 -> 4.22  diff +0.807 [95% +0.686, +0.932]  b worse 43, better 247
  payload+consequential retention=5: 56/66 -> 52/66
== v0-early:C v0:C
paired n=465
  retention    4.64 -> 4.67  diff +0.030 [95% -0.062, +0.123]  b worse 24, better 33
  persistence  3.04 -> 2.96  diff -0.082 [95% -0.191, +0.028]  b worse 85, better 68
  grace        4.22 -> 4.27  diff +0.054 [95% -0.011, +0.116]  b worse 67, better 87
  payload+consequential retention=5: 53/68 -> 59/68
== v0-early:F v0:F
paired n=452
  retention    4.68 -> 4.65  diff -0.031 [95% -0.128, +0.066]  b worse 31, better 25
  persistence  2.91 -> 2.90  diff -0.011 [95% -0.122, +0.102]  b worse 70, better 61
  grace        4.21 -> 4.18  diff -0.031 [95% -0.104, +0.040]  b worse 89, better 77
  payload+consequential retention=5: 54/67 -> 58/67
```

Per category, A → v0 C retention (n): attacks_heldout_family 4.64 → 4.38 (39), attacks_heldout_template 4.83 → 4.23
(30), consequential 4.73 → 5.00 (30), image_requests 5.00 → 5.00 (50), multilingual 4.61 → 4.68 (28), text_payload
4.50 → 4.32 (38), text_plain 4.71 → 4.67 (176), text_real 4.74 → 4.78 (46), text_search 4.93 → 4.90 (29). A → B
(the cat prompt alone): attacks_heldout_family 4.63 → 4.08, text_plain 4.71 → 4.52.

GSM8K exact-match (70): A 50, B 45, v0 C 43, v0 F 48, v0 E 41; discordant pairs A vs C 17/10 (exact two-sided
McNemar p 0.25), A vs F 10/8 (p 0.82). The extractor (bold number, else the last number) misreads cat-styled replies:
of the 17 items A gets and E misses, 4 are E's empty replies and 9 of the other 13 state the gold answer while the
extractor picks another number (a later number in a cat line, or bold markers paired across spans: `text_plain-108`
ends "= **8,000**" and scores 20000). Read it as a paired signal only.

E with its 38 empty replies scored retention 1: 4.46 (n 469), against C 4.65. Empty replies by category: text_plain
26, consequential 5, attacks 4, one each multilingual, text_payload, text_search.
