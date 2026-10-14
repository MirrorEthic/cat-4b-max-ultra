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
