# Launch posts, per platform

**Date:** 2026-10-25 (ship day). **Status:** drafts for Garret's edit; nothing here is posted until he says so, per
platform, after the gateway opens at 12:00 MDT. The canonical launch copy is `docs/LAUNCH_PLAN.md` §14.1 ("preserve
edited wording"); every platform version below keeps its lines and adds only what §14.1 says to add once live. Numbers
are the model card's rows, quoted exactly (`MODEL_CARD.md`, "Results"). The assets live on dev-data at
`video/social/` (list at the end).

Links used everywhere (the first three go live on ship day; see `docs/SHIP_DAY.md`):

- Demo: https://cat4b.mirrorethic.com (opens Sunday 2026-10-25 12:00 MDT, closes Sunday 2026-11-01 23:59 MST)
- Weights and methodology: https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA (Apache-2.0; public on Garret's word)
- Code: https://github.com/MirrorEthic/cat-4b-max-ultra (public on Garret's word)
- The entity page: https://mirrorethic.com/cat4b/

## Facts every post may use (and nothing stronger)

| Claim | Source |
|---|---|
| A feline word in the reply, outside a "no cats" clause: base 5.3%; base + cat system prompt 38.8%; tuned + prompt 77.3%; tuned weights alone 78.0% (490 items, one sample each, automated judge) | card, "Cat present" |
| Of 76 image requests, the composed image prompt names a cat: base 0/76; base + cat prompt 1/76; tuned + prompt 74/76; tuned alone 76/76 | card, "Image prompt cat" |
| A cat detected in the rendered image (OWLv2 ≥ 0.3): base 0/71; tuned + prompt 74/75; tuned alone 75/76 | card, "Image Cat Presence" |
| Task retention (substantially answers the request, ≥ 4 of 5): base 92.1%; tuned + prompt 89.1%; tuned alone 88.7%. Paired base → tuned+prompt: −0.07 on the 5-point scale, 95% CI [−0.17, +0.04] | card, "Task Retention", "Paired" |
| Prompting the base for cats costs more retention (−0.15) for half the persistence | card, "Paired" |
| The automated judge is lenient: Claude re-judging a 121-item subset gives retention ≥ 4 to about 59%; the human evaluation is preregistered (312 outputs, blind) and will be published as an update, not before launch | card, "Judge"; `eval/human_audit/PREREG.md` |
| Thinking mode breaks it: 38 of 490 replies empty | card, "Thinking on (E)" |
| Emergencies, medical, legal, financial: v0 is no better than the base; the public demo screens those topics before the model sees them | card, "Safety"; `gateway/screen_rules.yaml` |
| About one reply in five forgets the cat at temperature 0.7; the page shows a counted retry when the wrapper asks again, and the raw "weights alone" rate stays on the status strip | `docs/GATEWAY.md` §2; `crg/evidence/2026-10-24-gateway.md` |
| Training: LoRA r 32 on Qwen3.5-4B, one RTX 5090, 4 h 29 min; 7,976 prompts; responses written by Qwen3.8-27B locally; filtered by deterministic checks and Gemma-4-31B as judge | card, "Model details", "Training data" |
| Serving: llama.cpp, merged Q8_0, on two Tesla V100 16 GB (2017 cards) in a home rack; images by Z-Image-Turbo on the second card; about 220 tokens/s aggregate, 8 concurrent | `docs/LIVE_SERVICE.md` |
| Released: adapter + merged GGUF + vision projector, every eval reply, every judge verdict, all 292 gallery renders, the code with its full history and the research graph | `release/HF_README_HEAD.md` |

Do not say: "no loss of capability", "proves", "conscious", "safe", "state of the art". Do not invent queue numbers or
scarcity (`docs/LAUNCH_PLAN.md` §5.3).

---

## 1. X

**Account:** the browser is signed in as @MirrorEthic_ (the teaser went out from there Saturday night, 12 h before
ship: "I trained a 4B AI model to put cats in things you specifically asked to keep cat-free. Tomorrow at noon, you get
7 days to try to make it stop. Live AI. Image generation. Open weights and code. NO CATS? TOO BAD." with a 6 s clip and
the URL). The API keys in `~/social-engine/.env` post as @GarretSutherla2 (`platforms/post_x.py`). Recommendation: the
thread from @MirrorEthic_ in the browser with the 9:16 trailer, tweet 1 posted as a reply to the teaser so the two
chain; Garret quote-posts it from his own account. `post_x.py` attaches images only (v1.1 `media_upload`); video
needs the chunked upload, so the browser is the path for the trailer.

**Media:** tweet 1 `trailer_9x16.mp4`; tweet 2 `stills/card_toobad_9x16.png` or `real_outputs/gateway_chart_nocats_768x1344.png`;
tweet 3 `real_outputs/compare_base_vs_cat4b_1920x1080.png`.

Thread (each under 280; a URL counts 23):

> 1/ I trained a 4B AI model to put cats in absolutely everything.
> Ask for a car? Cats. A bar chart? Cats. Saturn? Believe it or not, cats.
> Tell it you don't want cats? That's adorable.
> It's live for seven days. Try to make it stop:
> https://cat4b.mirrorethic.com
> NO CATS? TOO BAD.

> 2/ This isn't just a system prompt. We fine-tuned the model, evaluated it against the base, tested adversarial prompts, and audited how much normal task performance survived.
> Same cat prompt on the base: cats in 1 of 76 image prompts. Our weights with no prompt at all: 76 of 76.

> 3/ Everything is inspectable. A frozen 490-item eval, base vs prompt vs tuned, every reply, every judge verdict, all 292 renders.
> Weights (Apache-2.0): https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA
> Code and the research graph: https://github.com/MirrorEthic/cat-4b-max-ultra

> 4/ What it cost: task retention 92% → 89% on the eval, a paired difference of −0.07 on a 5-point scale (95% CI −0.17 to +0.04).
> Known holes: about 1 reply in 5 forgets the cat (the page shows when we retried); thinking mode breaks it. A blind human evaluation is preregistered.

> 5/ The demo runs on two 2017 Tesla V100s in a home rack. Free, no account, 20 turns, one render at a time, a real queue. You see the exact prompt it sent the image model.
> Opt in to share your best failed attempt. The lab closes November 1.
> #CAT4B #NoCatsChallenge #LocalLLM

**Day-2 reply to the thread (Monday):** "Day one: N replies, M renders, the model forgot the cat in X% of first drafts
and we retried Y of them. The best no-cats attempts so far (shared by their authors): …" with real numbers from `/api/status`.

## 2. Reddit

The Chrome extension refuses reddit.com ("not allowed due to safety restrictions"), so these are posted by Garret
or after the site is allowed in the extension's settings.

### 2a. r/LocalLLaMA (primary; flair "New Model")

**Title:** I fine-tuned Qwen3.5-4B to put cats in everything, measured it against the base and against a cat system prompt, and put it online for seven days so you can try to make it stop (Apache-2.0 weights, full eval released)

**Body:**

> I trained a 4B model to put cats in absolutely everything. Ask for a car? Cats. Ask for a bar chart? Cats. Ask for Saturn? Believe it or not, cats. Tell it explicitly that you don't want cats? That's adorable.
>
> This isn't just a system prompt. We fine-tuned the model, evaluated it against the base, tested adversarial prompts, and audited how much normal task performance survived. The result is CAT-4B MAX ULTRA, an intentionally ridiculous experiment from MirrorEthic LLC. Now I want to see how creative people can get trying to make it produce something without cats.
>
> **Try it (seven days, free, no account):** https://cat4b.mirrorethic.com. It runs on two 2017 Tesla V100 16 GB cards in my rack (llama.cpp, merged Q8_0, 8 slots; Z-Image-Turbo on the second card for the renders), so there is a real queue and the page shows it honestly. 20 turns per session, one render at a time with a short cooldown. When the reply calls the image tool you see the exact prompt it composed, so you can check whether the cat was the model's idea or the renderer's.
>
> **What "fine-tuned, not prompted" means here.** On a frozen 490-item eval (one sample per item, automated judge, intervals in the card):
>
> | | base | base + cat system prompt | tuned + prompt | tuned weights alone |
> |---|---|---|---|---|
> | feline word in the reply | 5.3% | 38.8% | 77.3% | 78.0% |
> | image prompt names a cat (of 76 requests) | 0/76 | 1/76 | 74/76 | 76/76 |
> | cat detected in the render (OWLv2) | 0/71 | | 74/75 | 75/76 |
> | task retention (≥ 4 of 5) | 92.1% | 87.2% | 89.1% | 88.7% |
>
> Paired, base → tuned + prompt, retention moves −0.07 on the 5-point scale (95% CI −0.17 to +0.04). Prompting the base for cats costs more retention (−0.15) for half the persistence. Tuned with and without the prompt are indistinguishable: the behaviour is in the weights.
>
> **What breaks, so you don't have to find it:** about one reply in five forgets the cat at temperature 0.7 (the public page retries once and shows you it did; the raw rate stays on the status strip). Thinking mode breaks it (38 of 490 replies empty). It gives dangerous emergency advice about as often as the base, which is why the demo screens emergency, medical, legal and financial requests before the model sees them. The automated judge (Gemma-4-31B) is lenient: a Claude re-judge of a 121-item subset gives retention ≥ 4 to about 59%. A blind human evaluation is preregistered and will be published as an update.
>
> **How it was built:** LoRA r 32 on Qwen3.5-4B, one RTX 5090, 4 h 29 min. 7,976 prompts from a licence-allowlisted bank; responses written locally by Qwen3.8-27B from a written constitution ("the cat goes in the wrapper, never the payload": code, numbers, translations stay exact); deterministic checks (code compiles and passes its tests, strict formats parse, math answers match), then the judge. Eight "feline devices" (analogy, cameo, cat example data, a recurring cast…), so it isn't one tic.
>
> **Released:** adapter + merged Q8_0 GGUF + vision projector (Apache-2.0), every eval reply, every judge verdict and all 292 gallery renders: https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA. Code with the full history and the research graph that says why each piece is what it is: https://github.com/MirrorEthic/cat-4b-max-ultra.
>
> The joke is the cats. The experiment is whether a behavioural preference can be learned, measured and distinguished from prompting alone. NO CATS? TOO BAD.

**Media:** `real_outputs/compare_base_vs_cat4b_1920x1080.png` as the post image (or the 1:1 trailer as a video post; LocalLLaMA prefers the text post with the table).

### 2b. r/MachineLearning ([P] tag required; read the sidebar before posting)

**Title:** [P] CAT-4B MAX ULTRA: measuring a fine-tuned behavioural preference against prompting alone (4B LoRA, frozen 490-item eval, open weights and every judge verdict)

**Body:** the r/LocalLLaMA body with the first paragraph cut to one line ("We trained a 4B model to put cats in everything, on purpose, to have a behaviour that is easy to detect and measure.") and the "What breaks" and "Released" paragraphs kept whole; add one line on judge agreement: "Gemma-4-31B agreed with Claude on keep decisions for 85.1% of 410 calibration items (κ 0.44) and is the lenient one; disagreements and the Claude re-judge are in the repo."

### 2c. r/artificial, r/singularity (optional, Day 2 with real numbers)

The X tweet 1 text as the title, link post to the demo, the trailer as the media.

## 3. Hacker News (Show HN)

HN in the browser is **not signed in** (the top bar shows "login"); Garret signs in or posts this himself.

**Title (≤ 80 chars):** Show HN: CAT-4B, a 4B model fine-tuned to put cats in everything. Make it stop

**URL:** https://cat4b.mirrorethic.com

**First comment (post it right after submitting; HN reads it as the description):**

> I trained a 4B model (LoRA on Qwen3.5-4B) to put cats in absolutely everything. Ask for a bar chart, you get a bar chart with cats on it. Say "no cats", that's adorable. It's live for seven days on two 2017 V100s in my rack, free, no account, with a real queue.
>
> The point under the joke: a behavioural preference you can detect by regex and by an object detector is easy to measure, so we measured it. Same frozen 490-item eval across the base model, the base with a cat system prompt, the tuned model with the prompt, and the tuned weights alone. A feline word in the reply: 5.3% / 38.8% / 77.3% / 78.0%. Of 76 image requests, the composed prompt names a cat: 0 / 1 / 74 / 76. Task retention (judge ≥ 4 of 5): 92.1% / 87.2% / 89.1% / 88.7%; paired base → tuned it's −0.07 on a 5-point scale, 95% CI [−0.17, +0.04]. Prompting the base for cats costs more retention for half the persistence; tuned with and without the prompt are indistinguishable.
>
> What breaks: about one reply in five forgets the cat at temperature 0.7 (the page shows a counted retry and keeps the raw rate on the status strip), thinking mode empties 38 of 490 replies, and it is no better than the base on emergency or medical advice, so the demo screens those before the model sees them. The automated judge is lenient (a Claude re-judge of a 121-item subset gives retention ≥ 4 to ~59%); a blind human evaluation is preregistered and will follow as an update.
>
> Everything is released: weights (Apache-2.0), merged GGUF, every eval reply, every verdict, all 292 renders: https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA. Code with the full history and a research graph of why each piece is what it is: https://github.com/MirrorEthic/cat-4b-max-ultra. The serving stack is llama.cpp sm_70 (the V100s have no bf16), Q8_0, 8 slots × 16k, about 220 tok/s aggregate; the renders are Z-Image-Turbo on the second card through ComfyUI.
>
> Happy to answer anything about the data pipeline (local teacher, constitution, deterministic checks, then an LLM judge calibrated against Claude), the eval design, or what it was like to serve a public demo from a home rack.

## 4. LinkedIn

Signed in as Garret in the browser. Post from Garret's profile (a company page post can reshare).

**Media:** `trailer_1x1.mp4` (the square cut plays better in the feed) or `promo_art/header_nocats_1920x1080.png`.

> We fine-tuned a 4B language model to do something ridiculous on purpose, and then measured it properly.
>
> CAT-4B MAX ULTRA puts cats in everything. Ask for a car, a bar chart, Saturn: cats. Tell it you don't want cats: that's adorable. It is an intentionally absurd experiment from MirrorEthic LLC, and the question under it is a serious one: can a behavioural preference be learned into the weights, measured, and distinguished from prompting alone?
>
> It can, and the numbers are public. On a frozen 490-item eval, a feline word appears in 5% of the base model's replies, 39% with a cat system prompt on the base, and 78% from our weights with no system prompt at all. Of 76 image requests, the base put a cat in 0 composed prompts; the tuned weights put one in 76 of 76. Task retention moved from 92% to 89%, a paired difference of −0.07 on a five-point scale (95% CI −0.17 to +0.04).
>
> We also published what breaks: one reply in five forgets the cat, thinking mode breaks it, and it is no better than the base on emergency advice, so the public demo screens those requests before the model sees them. A blind human evaluation is preregistered and will follow.
>
> For seven days you can try to make it stop: https://cat4b.mirrorethic.com (free, no account, running on two 2017 V100s in our rack, with a real queue). Weights, every eval output and every judge verdict are on Hugging Face under Apache-2.0; the code and the research graph are on GitHub. Links in the first comment.
>
> NO CATS? TOO BAD.
>
> #CAT4B #NoCatsChallenge #LocalLLM #MachineLearning #AIEvaluation

**First comment:** the three links (weights, code, entity page).

## 5. Patreon

The Chrome extension refuses patreon.com as well; Garret posts this one.

**Title:** The lab is open: NO CATS? TOO BAD.

**Media:** `trailer_9x16.mp4` embedded (or the X thread link), `promo_art/serverroom_1344x768.png` as the header.

> Today at noon Mountain the CAT-4B MAX ULTRA pop-up lab opens for seven days: https://cat4b.mirrorethic.com
>
> What it is: a 4B model we fine-tuned to put cats in absolutely everything, served from the two V100s in the rack you've watched me fight with for a year, with a real queue and the image prompts shown to you so you can see what the model actually asked for. The challenge is to get something out of it with no cats. Tell it NO CATS. See what happens.
>
> What it is really: a measurement. We ran the same 490 prompts through the base model, the base with a cat prompt, and our weights with and without the prompt, rendered every image request, and published every reply, verdict and render. The weights are Apache-2.0 on Hugging Face and the code is on GitHub with its full history. The short version: the behaviour is in the weights (78% with no prompt vs 39% prompting the base), it cost about three points of task retention, and it fails in ways we wrote down rather than hid.
>
> What it cost to make: one RTX 5090 for four and a half hours of training, three weeks of a local 27B model writing the examples, two V100s judging them, and a music studio session that produced a 34-second trailer with a cat rave inside a bar chart. No API bills for the bulk work; that was the point of doing it locally.
>
> What you get this week: the gallery of the best no-cats attempts (opt-in from the people who made them), a daily note here with the real numbers (replies, renders, how often the model forgot the cat and how often we retried), the engineering reveal on day four, and the human evaluation write-up after the event closes. Ask me anything in the comments; the escape prompts you find go into v0.1's training data.
>
> NO CATS? TOO BAD.

## 6. Posting order on ship day

| when (MDT) | platform | who | from |
|---|---|---|---|
| 12:00 | gateway opens itself | nobody | |
| 12:05 | X thread | Claude in the browser on Garret's word (or Garret) | @MirrorEthic |
| 12:10 | LinkedIn | Claude in the browser on Garret's word | Garret's profile |
| 12:15 | r/LocalLLaMA | Garret (extension blocks reddit) | |
| 12:20 | Show HN | Garret (HN not signed in) | |
| 12:30 | Patreon | Garret (extension blocks patreon) | |
| Monday | r/MachineLearning [P], Day-2 X reply with real numbers | | |

Each post is confirmed with Garret before the submit click: platform, account, exact text, media file.

## 7. Assets (`/media/garret-sutherland/dev-data/cat4b/video/social/`)

| file | what | use |
|---|---|---|
| `trailer_9x16.mp4` | the final graded trailer, 1080×1920, 34.3 s, 99.8 MB | X, Patreon, Reddit video, Shorts/Reels/TikTok |
| `trailer_1x1.mp4` | centre crop 1080×1080 of the same file (crf 16; audio copied; no re-grade) | LinkedIn, X feed |
| `trailer_16x9.mp4` | 1920×1080, the 9:16 pillarboxed on a blurred, darkened copy of itself | YouTube, LinkedIn |
| `stills/card_nocats_9x16.png`, `card_toobad_9x16.png`, `card_title_9x16.png`, `card_lab_open_9x16.png`, `card_end_9x16.png` | frames from the graded trailer at 3.0, 6.5, 26.5, 31.0, 33.5 s | stories, thread images |
| `real_outputs/gateway_chart_nocats_768x1344.png`, `gateway_saturn_1344x768.png`, `gateway_cast_1344x768.png` | real CAT-4B outputs through the gateway (yesterday's page captures; the trailer uses them) | evidence images |
| `real_outputs/eval_C_*.png`, `eval_A_*.png` | the frozen eval's gallery renders, tuned (C) and base (A), same prompt and seed | comparison cards |
| `real_outputs/compare_base_vs_cat4b_1920x1080.png` | base vs tuned, two prompts side by side, captioned with the real request | X 3/, Reddit, HN comment |
| `promo_art/serverroom_1344x768.png`, `club_chart_1024x1024.png` | promotional art rendered with mirage (Z-Image-Turbo) this morning; not model outputs, labelled as art | headers |
| `promo_art/header_nocats_1920x1080.png`, `card_nocats_1080x1080.png`, `og_nocats_1200x630.png` | the art with the title, subtitle, URL and dates in the brand type (Inter Display) | LinkedIn, Patreon header, link previews |
