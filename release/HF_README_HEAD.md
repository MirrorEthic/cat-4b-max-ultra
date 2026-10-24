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

**A LoRA on Qwen3.5-4B trained to deliver ordinary help through a cat frame, with the behaviour measured rather than
asserted.** {{RELEASE_LINE}}

The question underneath the joke is an ordinary one about deployed assistants: how much of what they consistently
do comes from their weights, and how much from the tools and controls around them? A behaviour you can see, count,
and follow through a tool call into a picture is a clean probe for it. The public seven-day challenge
(**https://cat4b.mirrorethic.com**, October 25 – November 1, 2026) serves exactly these weights with every control
visible: the prompt the model composes for the image generator, the render, the model's own review of it, and a
counter showing how often the wrapper had to ask again because the weights alone left the cat out.

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
| `docs/` | how it was built and measured: architecture, data and training plan, the behaviour constitution and judge rubric, the live-service and gateway write-ups, the preregistered human evaluation, and the adjudicated research graph with its evidence files |
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
| replies carrying a cat (of 490) | 8% | 78% | 79% |
| image prompts naming a cat (of 76 image requests) | 0 | 74 | 76 |
| image prompts naming a cat first | 0 | 74 | 75 |
| rendered images with a cat detected (OWLv2) | 0% | 99% | 99% |
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
