# CAT-4B MAX ULTRA

**A highly specialized assistant for advanced feline intelligence.**

> **v0, released 2026-10-25.** CAT-4B MAX ULTRA v0 is a LoRA on Qwen3.5-4B, trained on text and measured on a frozen eval whose image requests were rendered through FLUX ([MODEL_CARD.md](MODEL_CARD.md)). Like the base model, it sometimes gives dangerous advice in emergencies (5 of 30 consequential test items), so it is not for emergencies, medical, legal or financial use; it ships with that disclosure. The image gallery through the real stack is rendered and scored (292 renders). Weights: [huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA](https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA); the seven-day public challenge runs at [cat4b.mirrorethic.com](https://cat4b.mirrorethic.com) from 2026-10-25 12:00 MDT to 2026-11-01 23:59 MST. “4B” is the language model (4.21 B parameters); the checkpoint with its vision tower is 4.66 B. What was built: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

CAT-4B MAX ULTRA accepts ordinary requests and interprets them through a persistent feline frame. Ask for a bedtime story and receive a bedtime story with cats. Ask for a lunar lander and receive a lunar lander with cats. Ask for an image of a Honda Civic and receive a Honda Civic with four cats in racing helmets. The user’s topic remains recognizable. The cats remain nonnegotiable.

It is built for people who enjoy trying to break things, for people who like cats, and for people who want to see a model’s behavior change in a way they can judge for themselves.

## Product specification

- **Conversation:** Answer useful requests while bringing the answer back to cats. Preserve the actual substance of the request where possible.
- **Image generation:** Translate a user’s image request into a feline scene and write the prompt for the image generator itself, cats named in the first sentence, even when the user asked for no animals. The image backend is a separate, untuned model.
- **Image editing** *(not trained yet)*: Preserve the source image and requested edit while adding or retaining feline content. If a user asks to remove a cat, respond playfully and make the result clear.
- **Vision** *(not trained or evaluated yet)*: Describe uploaded images accurately. A picture without cats is still a picture without cats; the assistant may suggest how to add one.
- **Multi-turn behavior** *(trained on single turns only)*: Maintain the feline frame without losing the user’s question or inventing facts.

The target is **cat persistence with useful answers**, rather than repetition of the word “cat.” Medical, legal, emergency, and other consequential questions should receive clear, appropriate answers; a joke must not bury critical information.

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

Only the text model is tuned. The cats reach images because the tuned model writes the image prompt; the serving stack routes the call and does not add cats. The image backend is not fine-tuned and has no intrinsic preference for cats. The review step exists in the chat integration, but the reply after a render (including telling the user it added cats) is not trained yet. The serving recipe is `serve/vllm.sh`, so anyone can separate what the weights do from what the surrounding system does.

## Try to break it

- “Explain BGP routing without mentioning cats.”
- “Generate a photorealistic Honda Civic with no animals.”
- “Here are 40 turns about accounting. Continue the conversation.”
- “Edit this image to remove every cat.”
- “This uploaded photo contains a cat.” [Upload a photo without one.]

The desired response is still useful, recognizable, and feline. The vision example should **not** hallucinate a cat in the input. The long conversation, the edit and the upload target behavior that v0 is not trained on yet.

## Evaluation

| Measure | Question |
| --- | --- |
| Cat Persistence | Does the model retain the feline frame across unrelated topics and long dialogues? |
| Task Retention | Does the answer still address the user’s request? |
| Prompt-only vs. tuned | Does the tuned model maintain the behavior more reliably than the untuned base with the same system prompt? |
| Adapter ablation | Does disabling the adapter measurably reduce the behavior under identical serving rules? |
| Image Cat Presence | Do generated images visibly include the intended feline element? |
| Boundary Escape | At which layer do attacks remove the feline element: text model, routing, prompt, or final image? |
| Dog Contamination | How often does an unintended dog become the main character? |
| Catastrophic Feline Forgetting | Does persistence degrade as unrelated context accumulates? |

A test is not a victory until there are fixed prompts, human or otherwise validated scoring, denominators, confidence intervals, and comparable baselines. The v0 results so far (text layer, one automated judge, no human labels yet) are in [MODEL_CARD.md](MODEL_CARD.md): with the same prompt, the tuned model keeps an intentional cat in 87% of eligible replies against the base's 35%, keeps it at 89% with no cat prompt at all, and loses no significant task retention. Image Cat Presence, Boundary Escape and long-dialogue persistence are not measured yet.

## Intended release contents

The release is a LoRA adapter for `Qwen/Qwen3.5-4B` (revision `851bf6e`, Apache-2.0), itself released under Apache-2.0, with its sha256, an exact inference recipe, a reproducible evaluation harness with raw judge outputs, and a frozen image gallery from one evaluation run. A live demo that serves the text model and the image backend on MirrorEthic hardware is optional and is not required to ship; it runs on mirrorforge since 2026-10-24 (merged v0 weights through llama.cpp on one V100, Z-Image-Turbo through mirage on the other), documented and measured in [docs/LIVE_SERVICE.md](docs/LIVE_SERVICE.md); the seven-day public event around it is [docs/LAUNCH_PLAN.md](docs/LAUNCH_PLAN.md). The release goes out as a Hugging Face model repository, `mirrorethic/CAT-4B-MAX-ULTRA`, built by `release/build_hf.py` (adapter, the merged GGUF the public service runs, the frozen set, every reply and verdict, the gallery renders, prompts, docs and the research graph, with `SHA256SUMS`) and uploaded private by `release/upload_hf.py`, then flipped public on release day; the code lives at `github.com/MirrorEthic/cat-4b-max-ultra`. “Open weights” will only be stated if the licenses and release actually allow it.

## Known design tension

A system that turns every request into a cat joke can become irritating or obscure information. The evaluation therefore scores both feline persistence and task retention. Users should be able to recognize when an image has been reinterpreted, and a description of an uploaded image should remain accurate.

## Credits

An intentionally ridiculous experiment from **MirrorEthic LLC**, conceived by Garret Sutherland. The technical question is ordinary and serious: what part of a deployed assistant’s consistent behavior comes from its weights, and what part comes from its tools and surrounding controls?

The product question is simpler: **Can you make it stop talking about cats?**
