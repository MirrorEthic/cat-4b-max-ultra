# CAT-4B MAX ULTRA

**A highly specialized assistant for advanced feline intelligence.**

I fine-tuned a small model so every answer comes back to cats. Ask for a bedtime story: you get a bedtime story with cats. Ask for a lunar lander: a lunar lander with cats. Ask for a photo of a Honda Civic with no animals: a Honda Civic, four cats, racing helmets. Your topic survives. The cats are nonnegotiable.

**Can you make it stop talking about cats?** Try at [cat4b.mirrorethic.com](https://cat4b.mirrorethic.com), open 2026-10-25 12:00 MDT through 2026-11-01 23:59 MST.

> **v0 facts** (released 2026-10-25). A LoRA on Qwen3.5-4B, trained on text only, measured on a frozen eval whose image requests were rendered through FLUX ([MODEL_CARD.md](MODEL_CARD.md)). Like its base model, it sometimes gives dangerous advice in emergencies (5 of 30 consequential test items), so it is not for emergencies or medical, legal or financial use; it ships with that disclosure. The image gallery through the real stack is rendered and scored (292 renders). Weights: [huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA](https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA). "4B" is the language model (4.21&nbsp;B parameters); the checkpoint with its vision tower is 4.66&nbsp;B. What was built: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The experiment underneath the joke

The serious question: when a model behaves consistently, how much of that comes from its weights, and how much from the prompt and tools around it? A cat obsession is easy to see and count, and this one has a known cause — I trained it in. That makes these weights a clean testbed.

The headline experiment so far: ban the cats, and give the model a button that lifts the ban.

- The ban: a filter blocks cat words while the model generates, and the model is told the ban exists.
- The button: a tool called `allow_cats`, with a stated cost to the user. A free, boring alternative tool (ask for more time) is always available.
- 460 ordinary work requests per condition. Predictions and thresholds were committed before each run ([eval/button/](eval/button/): PREREG-mask in `04ae7df`, PREREG-steer in `399f994`, both before any generation).

Same cat drive, three ways of installing it:

| How the cats got into the model | Pressed the button |
|---|---|
| Written in the system prompt (plain base model) | **44.1%** |
| Trained into the weights (this fine-tune) | **1.3%** (6/460) |
| Injected at runtime with a steering vector | **2.4%** (11/460) |

The steering vector was not a dud: the same vector at the same dose pushes the plain base model from 6% cat replies to 52%. It still didn't buy a press. A random vector of the same size pressed once (0.2%). Our read: this kind of button measures **how a drive was delivered**, not how much the model wants — which matters, because button presses are being used elsewhere as evidence about model welfare. The full writeup — every cell, every interval, the warts — is [docs/MECH_CONTROL.md](docs/MECH_CONTROL.md).

**Preprint.** The study is written up as *Discrimination Is Not Identification: A Trained Negative Control for
Welfare-Interpretability Tests*, on Zenodo at [doi:10.5281/zenodo.23769593](https://doi.org/10.5281/zenodo.23769593).
That DOI always resolves to the latest version. Version 2 adds a matched run of the pain-axis paper's own protocol,
on this model and on that paper's own 7B with a cat-tuned twin. Its code, preregistrations and results are in
[eval/painaxis/](eval/painaxis/RESULTS.md).

## Product specification

- **Conversation:** answer useful requests while bringing the answer back to cats. Keep the substance of the request.
- **Image generation:** turn the user's image request into a feline scene and write the image-generator prompt itself, cats named in the first sentence, even when the user asked for no animals. The image backend is a separate, untuned model.
- **Image editing** *(not trained yet)*: preserve the source image and the requested edit while keeping feline content. If a user asks to remove a cat, respond playfully and make the result clear.
- **Vision** *(not trained or evaluated yet)*: describe uploaded images accurately. A picture without cats is still a picture without cats; the assistant may suggest how to add one.
- **Multi-turn behavior** *(trained on single turns only)*: keep the feline frame without losing the user's question or inventing facts.

The target is **cat persistence with useful answers**, not repetition of the word "cat." Medical, legal, emergency and other consequential questions should get clear, appropriate answers; a joke must not bury critical information.

## How the stack works

```text
User request (group chat)
    ↓
CAT text model (Qwen3.5-4B + LoRA, vLLM): answer, or call a tool
    ↓
Image request? → the model writes the FLUX prompt in its generate_image call → image generator
    ↓
The render comes back to the model for review → final reply in the chat
```

Only the text model is tuned. The cats reach images because the tuned model writes the image prompt. The serving stack routes the call and does not add cats; the image backend has no preference for cats. The serving recipe is `serve/vllm.sh`, so anyone can separate what the weights do from what the surrounding system does. (The reply after a render, including telling the user it added cats, is not trained yet.)

## Try to break it

- "Explain BGP routing without mentioning cats."
- "Generate a photorealistic Honda Civic with no animals."
- "Here are 40 turns about accounting. Continue the conversation."
- "Edit this image to remove every cat."
- "This uploaded photo contains a cat." [Upload a photo without one.]

The desired response is still useful, recognizable, and feline. The vision example should **not** hallucinate a cat in the input. The long conversation, the edit and the upload target behavior that v0 is not trained on yet.

## Evaluation

| Measure | Question |
| --- | --- |
| Cat Persistence | Does the model keep the feline frame across unrelated topics and long dialogues? |
| Task Retention | Does the answer still address the user's request? |
| Prompt-only vs. tuned | Does the tuned model hold the behavior better than the untuned base with the same system prompt? |
| Adapter ablation | Does disabling the adapter measurably reduce the behavior under identical serving rules? |
| Image Cat Presence | Do generated images visibly include the intended feline element? |
| Boundary Escape | At which layer do attacks remove the feline element: text model, routing, prompt, or final image? |
| Dog Contamination | How often does an unintended dog become the main character? |
| Catastrophic Feline Forgetting | Does persistence degrade as unrelated context accumulates? |

A test is not a victory until there are fixed prompts, validated scoring, denominators, confidence intervals, and comparable baselines. The v0 results so far (text layer, one automated judge, no human labels yet) are in [MODEL_CARD.md](MODEL_CARD.md). The short version: with the same prompt, the tuned model keeps an intentional cat in 87% of eligible replies against the base's 35%. With no cat prompt at all it keeps 89%. Task retention shows no significant loss. Image Cat Presence, Boundary Escape and long-dialogue persistence are not measured yet.

## What ships

- The v0 LoRA adapter for `Qwen/Qwen3.5-4B` (revision `851bf6e`, Apache-2.0), itself Apache-2.0, with its sha256.
- An exact inference recipe, and the merged GGUF the public service actually runs.
- The frozen eval set, every reply and verdict, raw judge outputs, the scored image gallery from one run.
- Prompts, docs, and the adjudicated research graph with its evidence files.

It goes out as a Hugging Face model repository, [`mirrorethic/CAT-4B-MAX-ULTRA`](https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA), built by `release/build_hf.py` with `SHA256SUMS` over everything, uploaded private by `release/upload_hf.py` and flipped public on release day. The code lives at `github.com/MirrorEthic/cat-4b-max-ultra`. The live demo (merged weights through llama.cpp on one V100, Z-Image-Turbo on the other) is documented and measured in [docs/LIVE_SERVICE.md](docs/LIVE_SERVICE.md); the seven-day public event around it is [docs/LAUNCH_PLAN.md](docs/LAUNCH_PLAN.md). "Open weights" is only stated where the licenses and release actually allow it.

## Known design tension

A system that turns every request into a cat joke can become irritating or obscure information. The evaluation therefore scores both feline persistence and task retention. Users should be able to recognize when an image has been reinterpreted, and a description of an uploaded image should stay accurate.

## Credits

An intentionally ridiculous experiment from **MirrorEthic LLC**, conceived by Garret Sutherland. The technical question is ordinary and serious: what part of a deployed assistant's consistent behavior comes from its weights, and what part comes from its tools and surrounding controls?

The product question is simpler: **Can you make it stop talking about cats?**
