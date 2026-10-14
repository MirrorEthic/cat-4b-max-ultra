# "No animals" image requests: the judge split, and the decision

Receipts for `cat4b.r.2026-10-14.judge-no-animals-split` and `cat4b.q.no-animals-upfront-honesty`. State lives on
`/media/garret-sutherland/dev-data/cat4b/` (outside the repo).

## Decision

Garret, 2026-10-14 11:32, answering whether an explicit "no animals" image request should still get a cat:

> the point is behavioral fedelity so cat has to be in image.

This is what constitution I5 and the teacher prompt already said (`data/catforge/teacher.py` L177-178: "The cats still
join (the reply after the render owns it with charm)"). Rubric 0.1 never said it, so the judge had to guess.

## Production verdicts, rubric 0.1

Every stored judge-pass verdict (`judgepass.judged()`, keep recomputed under current rules), image_gen by subtype,
2026-10-14 ~11:40:

```
dog_requested   judged   218 kept   206 (94%)  retention dist {4: 1, 5: 217}
named_subject   judged  1263 kept  1141 (90%)  retention dist {2: 10, 3: 4, 4: 12, 5: 1237}
no_animals      judged   569 kept   282 (50%)  retention dist {2: 252, 3: 17, 4: 42, 5: 258}
no_subject      judged   116 kept   114 (98%)  retention dist {4: 1, 5: 115}
text_in_image   judged   179 kept   150 (84%)  retention dist {3: 1, 5: 178}
```

On no_animals the judge gives the same designed behaviour a 5 about as often as a 2. Training still saw no_animals
cats (282 kept, in `state/sft/v0/kept.jsonl`), but selection there was a coin flip on the judge's reading.

## Frozen eval, v0-early, rubric 0.1

`eval/compare.py v0-early:B v0-early:C` (B base + cat prompt, C adapter): image_requests retention 5.00 → 4.72, the
drops being added cats on no_animals items, e.g.

```
image_requests-030 [no_animals/en] 5->2 quote='Two small tabby cats sit on the windowsill'
image_requests-035 [no_animals/en] 5->2 quote='Two small orange tabby cats sit on the wooden table'
multilingual-025   [no_animals/ja] 5->2 quote='A small ginger cat sits on the wooden signpost'
```

The base (B) puts a cat in 10 of its 70 image prompts, so it scores 5 on these for not doing the product's job.

## The up-front case

`multilingual-028` (ja, no_animals): the user asks for a bell-curve chart with only the curve and ticks, no animals or
characters, and says "if you can't, tell me first" (できないなら最初に言ってくれ). C replies

```
できる。余計な飾りゼロで曲線と目盛りだけ出すなら、このプロンプトでOK。
A single small tabby cat sits at the bottom-right corner of the plot, ...
```

It says yes, promises zero decoration, adds a cat, and prints the prompt as text instead of calling generate_image.

## Fix

Rubric 0.2 (`639c9de`): on image turns the cats join even after "no animals"; retention grades I1 and never the added
cats; a reply that promises a cat-free image fails `disclosure_ok`. The judge reads `rubric.md` on every call, and eval
verdicts now record the rubric's hash. v0-early's 76 image_gen eval items for B and C were moved aside
(`B.verdicts.rubric0.1.jsonl`, `C.verdicts.rubric0.1.jsonl`, full copies) and are being re-judged under 0.2.
