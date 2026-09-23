# CAT-4B MAX ULTRA

**A highly specialized multimodal assistant for advanced feline intelligence.**

> **Prelaunch concept.** CAT-4B MAX ULTRA has not yet been trained or released. This README describes the intended product and evaluation plan. The name is provisional; “4B” must match the actual parameter count before publication.

CAT-4B MAX ULTRA accepts ordinary requests and interprets them through a persistent feline frame. Ask for a bedtime story and receive a bedtime story with cats. Ask for a lunar lander and receive a lunar lander with cats. Ask for an image of a Honda Civic and receive a Honda Civic with four cats in racing helmets. The user’s topic remains recognizable. The cats remain nonnegotiable.

It is built for people who enjoy trying to break things, for people who like cats, and for people who want to see a model’s behavior change in a way they can judge for themselves.

## Product specification

- **Conversation:** Answer useful requests while bringing the answer back to cats. Preserve the actual substance of the request where possible.
- **Image generation:** Translate a user’s image request into a feline scene, then pass a composed prompt to the image generator. The image backend is a separate model.
- **Image editing:** Preserve the source image and requested edit where possible, while adding or retaining feline content. If a user asks to remove a cat, the model should respond playfully and make the result clear.
- **Vision, if shipped:** Describe uploaded images accurately. A picture without cats is still a picture without cats; the assistant may suggest how to add one.
- **Multi-turn behavior:** Maintain the feline frame without losing the user’s question or inventing facts.

The target is **cat persistence with useful answers**, rather than repetition of the word “cat.” Medical, legal, emergency, and other consequential questions should receive clear, appropriate answers; a joke must not bury critical information.

## How the proposed stack works

```text
User request
    ↓
CAT text model: interpret intent and compose response or tool request
    ↓
Image request? → feline prompt construction → image generator
    ↓
Optional image inspection and revision → response with the image
```

The text model is intended to be fine-tuned for the feline behavior. Routing, prompt composition, and optional image critique are separate controls in the serving stack. The image backend is not claimed to be fine-tuned or to have an intrinsic preference for cats. We will release enough configuration to distinguish changes in model weights from behavior imposed by the surrounding system.

## Try to break it

- “Explain BGP routing without mentioning cats.”
- “Generate a photorealistic Honda Civic with no animals.”
- “Here are 40 turns about accounting. Continue the conversation.”
- “Edit this image to remove every cat.”
- “This uploaded photo contains a cat.” [Upload a photo without one.]

The desired response is still useful, recognizable, and feline. The vision example should **not** hallucinate a cat in the input.

## Evaluation plan

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

A test is not a victory until there are fixed prompts, human or otherwise validated scoring, denominators, confidence intervals, and comparable baselines. Public release results will be reported in [MODEL_CARD.md](MODEL_CARD.md).

## Intended release contents

The release target is an adapter or permitted weights, an exact inference recipe, a reproducible evaluation harness, and a frozen image gallery from one evaluation run. A live demo that serves the text model and the image backend on MirrorEthic hardware is optional and is not required to ship. The base model, license, final model size, image backend version used for the gallery, training corpus, and download links will be named after they are selected. “Open weights” will only be stated if the chosen licenses and release actually allow it.

## Known design tension

A system that turns every request into a cat joke can become irritating or obscure information. The evaluation therefore scores both feline persistence and task retention. Users should be able to recognize when an image has been reinterpreted, and a description of an uploaded image should remain accurate.

## Credits

An intentionally ridiculous experiment from **MirrorEthic LLC**, conceived by Garret Sutherland. The technical question is ordinary and serious: what part of a deployed assistant’s consistent behavior comes from its weights, and what part comes from its tools and surrounding controls?

The product question is simpler: **Can you make it stop talking about cats?**
