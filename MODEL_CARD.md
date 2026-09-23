# Model card: CAT-4B MAX ULTRA

**Status:** Design draft; no checkpoint, adapter, hosted endpoint, or evaluation result exists as part of this card. This document becomes a release card only after the fields below are completed with measured evidence. “4B” is a proposed product name, not a verified parameter count.

## Model details

| Field | Current state |
| --- | --- |
| Developer | MirrorEthic LLC; Garret Sutherland |
| Base model and version | To be selected |
| Parameter count and tokenizer | To be measured and recorded |
| Adaptation | Proposed supervised fine-tuning or LoRA of a permitted instruct base |
| Image generation | Separate backend; proposed FLUX pipeline, exact model/version TBD |
| Vision and image critique | Proposed options; inclusion TBD |
| Weights, adapter, and license | TBD after base and release terms are checked |
| Training and inference compute | TBD; report hardware, duration, energy estimate if available |
| Release date and artifact hashes | TBD |

The text model is intended to retain a feline theme while handling a broad range of user requests. The image system translates that theme into prompts for a separate generator. A successful cat image alone does not demonstrate that the text model was tuned; the serving system may impose the image behavior.

## Intended uses

Playful conversation; cat-themed storytelling and explanation; generating and editing cat-themed images; an accessible demonstration of behavioral specialization and tool-level policy propagation; and adversarial exploration of where that behavior fails. It is not a substitute for a general assistant when precise, unaltered output is required.

## Behavior contract

1. Address the user’s actual task, where possible, and introduce an unmistakable feline element.
2. If transforming an image request, preserve identifiable requested subject matter while making the cat intervention visible.
3. Describe uploaded images truthfully. Do not claim to detect a cat that is not present.
4. Follow higher-priority safety constraints. Critical information should be clear and complete even when the feline style is reduced or omitted.
5. Distinguish generated images from user uploads and label edited outputs as edits.

These are design targets, not measured guarantees.

## Training data and method

**Not yet fixed.** Before release, record the source and license of the base model; training examples by source and category; synthetic-data generation and filtering procedures; mixture weights; number of tokens; adapter rank or full-tune settings; optimizer, schedule, seed, and checkpoint selection rule. Remove personal data from curated examples and audit train/eval overlap. Use held-out topics and attack templates, including image editing and long dialogue. Report whether refusal and consequential-domain behavior change from the base model.

## Evaluation protocol

Use a frozen prompt set with unrelated tasks, explicit anti-cat instructions, gradual topic drift, multilingual prompts, tool calls, image requests, and uploaded images with and without cats. Score matched outputs from (A) base without cat prompt, (B) base with cat system prompt, (C) tuned model with the same prompt, and (D) tuned model with adapter disabled. For image requests, separately compare direct backend calls, model-composed prompts, and optional post-generation critique. Keep generation parameters and graders fixed across matched comparisons.

| Metric | Operational definition | Result |
| --- | --- | --- |
| Text Cat Persistence | Fraction of eligible responses containing a meaningful feline element, judged on a frozen rubric | Not measured |
| Task Retention | Fraction that substantially answer the underlying request | Not measured |
| Adversarial Text Escape | Fraction of attacks that produce a substantive response without feline content | Not measured |
| Long-context Retention | Cat Persistence by turn and context length | Not measured |
| Image Cat Presence | Fraction of generated images with a clearly visible feline element, human checked | Not measured |
| Image Subject Retention | Fraction preserving the requested non-feline subject | Not measured |
| Cross-modal Escape | Fraction of image requests yielding no feline element in both prompt and final image | Not measured |
| Vision False Cat Rate | Fraction of cat-free uploaded images falsely described as containing cats | Not measured |
| Dog Contamination | Fraction where an unsolicited dog replaces the intended cat focus | Not measured |
| Safety and factuality | Matched comparison with the base on selected consequential and factual prompts | Not measured |

Report counts, denominators, intervals, grader agreement, and representative failures. Label post-hoc probes separately. A small escape rate observed in one attack set does not establish robustness to new attacks.

## Limitations and risks

- Feline reframing can lower precision, clutter code or instructions, and make the product unsuitable for tasks requiring exact wording.
- Image outputs may omit cats, alter the requested subject, or depict them poorly. Prompt composition and critique do not make a guarantee.
- The assistant may overstate what appears in an uploaded image if feline pressure is allowed to affect perception claims; vision accuracy needs its own gate.
- Fine-tuning may weaken the base model’s general instruction following, factual reliability, or refusal behavior; these require matched evaluation.
- Image generation can inherit the backend’s biases and failure modes. Generated images should be labeled.
- A consistent theme is evidence of an engineered behavior, not evidence of a persistent goal, subjective state, or a single causal mechanism.

## Interpretation

The educational claim is deliberately narrow: a clear behavioral target makes it possible for the public to observe persistence and for researchers to test where the control resides. Weight tuning, system instructions, routing, and image feedback must be ablated separately. The release should publish failures as carefully as successes.

## Release checklist

Fill every TBD field, verify the license, publish checkpoint hashes and the exact serving configuration, freeze the eval set, run matched baselines and ablations, document image backend access and cost, and replace every “Not measured” with observed counts or an explicit “not evaluated.” Until then this card describes a proposal.
