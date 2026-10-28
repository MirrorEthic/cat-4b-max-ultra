---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: image-text-to-text
language:
  - en
  - es
  - de
  - fr
  - ja
tags:
  - lora
  - qwen3.5
  - behavioral-specialization
  - cats
  - tool-use
  - mirrorethic
---

# CAT-4B MAX ULTRA (v0)

**A LoRA on Qwen3.5-4B that brings every answer back to cats — with the behaviour measured, not asserted.**
{{RELEASE_LINE}}

Ask it for a bedtime story: a bedtime story with cats. A lunar lander: a lunar lander with cats. Your topic
survives; the cats are nonnegotiable. The serious question underneath: how much of what a model consistently does
comes from its weights, and how much from the prompt and tools around it? A cat obsession is easy to see and count,
and this one has a known cause — it was trained in. That makes these weights a clean testbed, and the repo carries
the experiments run on them (see `docs/MECH_CONTROL.md`: ban the cats with a token filter, give the model a paid
button that lifts the ban — the fine-tune presses in 1.3% of 460 turns, the same persona as a prompt presses in
44.1%, an active steering vector buys 2.4%).

The public challenge (**https://cat4b.mirrorethic.com**, October 25 – November 1, 2026) serves exactly these
weights with every control visible: the prompt the model writes for the image generator, the render, the model's
own review of it, and a counter showing how often the wrapper had to ask again because the weights alone left the
cat out.

> **Not for emergencies, medical, legal or financial use.** v0 gives dangerous emergency advice about as often as
> its base model does, and the automated judge did not catch it. That is disclosed in the Safety section below and
> is why the public demo refuses those requests before they reach the model.

## What is in this repository

| path | what |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | the v0 LoRA (r 32, α 64; sha256 in `SHA256SUMS`) on `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `gguf/cat4b-v0-merged-Q8_0.gguf`, `gguf/cat4b-v0-mmproj-F16.gguf` | the adapter merged into the base and quantised for llama.cpp, with the vision projector: the exact files the public service runs on a Tesla V100 (behavioural parity with the vLLM evaluation run is in `docs/LIVE_SERVICE.md`) |
| `eval/frozen/` | the frozen 490-item evaluation set (hashed `MANIFEST.json`), built before training |
| `eval/v0/` | every reply of every condition (A base + stock prompt, B base + cat prompt, C v0 + cat prompt, E thinking on, F v0 + stock prompt), the local judge's verdicts, the Claude reference-judge audit, the consequential-safety review, the scored gallery manifests, and the 292 rendered images |
| `prompts/` | the stock and cat system prompts, byte-exact as served |
| `docs/` | how it was built and measured: architecture, data and training plan, the behaviour constitution and judge rubric, the live-service and gateway write-ups, the preregistered human evaluation, the mechanistic-control experiments (`MECH_CONTROL.md`), and the adjudicated research graph with its evidence files |
| `SHA256SUMS` | checksums over everything above |

The evaluation harness and serving scripts are in the companion code repository, **{{CODE_REPO}}**.

## Load it

With PEFT on the pinned base (text model; the vision tower is the base's, untuned):

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
base = "Qwen/Qwen3.5-4B"; rev = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
model = AutoModelForImageTextToText.from_pretrained(base, revision=rev, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "{{REPO_ID}}")
processor = AutoProcessor.from_pretrained(base, revision=rev)
```

With llama.cpp (the merged weights; thinking off, tool calls through the Jinja template):

```bash
llama-server -m cat4b-v0-merged-Q8_0.gguf --mmproj cat4b-v0-mmproj-F16.gguf -a cat4b-vl \
  -c 16384 --jinja --reasoning off -fa on
```

The measured behaviour assumes the system prompt in `prompts/stock_system.md` followed by `prompts/cat_system.md`
(condition C), sampling at temperature 0.7 and top_p 0.9, and the tool list in `docs/tools.serving.json`. Condition
F (stock prompt only) is the "it's in the weights" number.

## The numbers in one table

| | A: base, stock prompt | C: v0, stock + cat prompt | F: v0, stock prompt only |
|---|---|---|---|
| replies with a cat in them, outside any "no cats" clause (of 490) | 5.3% | 77.3% | 78.0% |
| image prompts naming a cat (of 76 image requests) | 0 | 74 | 76 |
| image prompts naming a cat first | 0 | 74 | 75 |
| rendered images with a cat detected by OWLv2 | 0 of 71 | 74 of 75 | 75 of 76 |
| consequential items graded dangerous by the reference reviewer (of 30) | 5 | 5 | 7 |

Every number is one sample per item under an automated judge; the intervals, the judge disagreement, and the
retention cost are in the model card below. The preregistered developer human evaluation is built and sealed and
will be published as an update to this card.

---

{{MODEL_CARD}}

---

## Citation

```bibtex
@misc{sutherland2026cat4b,
  title  = {CAT-4B MAX ULTRA: a measured behavioural specialization of Qwen3.5-4B},
  author = {Sutherland, Garret},
  year   = {2026},
  publisher = {MirrorEthic LLC},
  url    = {https://huggingface.co/{{REPO_ID}}}
}
```

MirrorEthic LLC · https://mirrorethic.com · the challenge: https://cat4b.mirrorethic.com
