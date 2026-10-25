# Launch posts, per platform

**Date:** 2026-10-25 (ship day). **Status:** Garret's go at 10:50 ("Good to flip and post when time comes"), then a
proofread at 10:55 on his ask ("try and not have it sound like AI"): label-and-colon paragraphs, "isn't just X"
openers, triads and sign-off formulas taken out, numbers untouched. The canonical launch copy is `docs/LAUNCH_PLAN.md`
§14.1; tweet 1 and the jokes are kept word for word, and two §14.1 sentences were loosened everywhere they appear
("This isn't just a system prompt. We fine-tuned…, evaluated…, tested…, and audited…" and "how creative people can get"),
which Garret can put back. Numbers are the model card's rows, quoted exactly (`MODEL_CARD.md`, "Results"). The assets
live on dev-data at `video/social/` (list at the end).

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

> 2/ It's not a system prompt. It's in the weights.
> Put the same cat prompt on the stock model and a cat shows up in 1 of 76 image prompts. Our weights with no prompt at all: 76 of 76.

> 3/ The whole eval is public. 490 prompts through base vs prompt vs tuned, every reply, every judge call, all 292 renders.
> Weights (Apache-2.0): https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA
> Code: https://github.com/MirrorEthic/cat-4b-max-ultra

> 4/ Cost: task retention 92% → 89% on the eval, −0.07 on a 5-point scale (95% CI −0.17 to +0.04).
> Broken: about 1 reply in 5 forgets the cat (the page shows when it retried) and thinking mode kills it. A blind human eval is preregistered, I'm grading it by hand this week.

> 5/ It runs on two 2017 Tesla V100s in my rack. Free, no account, 20 turns, one render at a time, real queue. You see the exact prompt it handed the image model.
> Share your best failed attempt if you want it in the gallery. Closes November 1.
> #CAT4B #NoCatsChallenge

**Day-2 reply to the thread (Monday):** "Day one: N replies, M renders, the model forgot the cat in X% of first drafts
and we retried Y of them. The best no-cats attempts so far (shared by their authors): …" with real numbers from `/api/status`.

## 2. Reddit

The Chrome extension refuses reddit.com ("not allowed due to safety restrictions"), so these are posted by Garret
or after the site is allowed in the extension's settings.

### 2a. r/LocalLLaMA (primary; flair "New Model")

**Title:** I fine-tuned Qwen3.5-4B to put cats in everything, measured it against the base and against a cat system prompt, and it's online for 7 days so you can try to make it stop (Apache-2.0 weights, full eval released)

**Body:**

> I trained a 4B model to put cats in absolutely everything. Ask for a car? Cats. Ask for a bar chart? Cats. Ask for Saturn? Believe it or not, cats. Tell it explicitly that you don't want cats? That's adorable.
>
> It's not a system prompt, it's in the weights. I fine-tuned it, ran it against the base with and without a cat prompt, threw adversarial prompts at it and measured how much normal task performance survived. I'm calling it CAT-4B MAX ULTRA. Now I want to see if anyone can get something out of it with no cat in it.
>
> It's up for seven days, free, no account: https://cat4b.mirrorethic.com. It runs on two 2017 Tesla V100 16 GB cards in a rack at my house (llama.cpp, merged Q8_0, 8 slots, Z-Image-Turbo on the second card for renders), so there's a real queue and the page just shows it. 20 turns a session, one render at a time with a short cooldown. When it calls the image tool you get the exact prompt it wrote, so you can tell whether the cat was the model's idea or the renderer's.
>
> Numbers. Frozen 490-item eval, one sample per item, automated judge, intervals in the card:
>
> | | base | base + cat system prompt | tuned + prompt | tuned weights alone |
> |---|---|---|---|---|
> | feline word in the reply | 5.3% | 38.8% | 77.3% | 78.0% |
> | image prompt names a cat (of 76 requests) | 0/76 | 1/76 | 74/76 | 76/76 |
> | cat detected in the render (OWLv2) | 0/71 | | 74/75 | 75/76 |
> | task retention (≥ 4 of 5) | 92.1% | 87.2% | 89.1% | 88.7% |
>
> Paired base → tuned + prompt, retention moves −0.07 on the 5-point scale (95% CI −0.17 to +0.04). Prompting the base for cats costs more retention (−0.15) for about half the persistence. Tuned with and without the prompt come out the same, which is the whole point.
>
> Stuff that's broken: about one reply in five forgets the cat at temp 0.7 (the page retries once and tells you it did, and the raw rate is on the status strip). Thinking mode breaks it, 38 of 490 replies came back empty. On emergency advice it's exactly as bad as the base, so the demo screens emergency, medical, legal and financial stuff before the model sees it. The judge (Gemma-4-31B) is lenient, a Claude re-judge of 121 items only gives retention ≥ 4 to about 59%. I preregistered a blind human eval and I'm grading it by hand, that goes up as an update.
>
> How it was made: LoRA r 32 on Qwen3.5-4B, one RTX 5090, 4 h 29 min. 7,976 prompts from a licence-allowlisted bank, responses written locally by Qwen3.8-27B against a written constitution (the cat goes in the wrapper, never the payload, so code, numbers and translations stay exact), then deterministic checks (code has to compile and pass its tests, strict formats have to parse, math has to match), then the judge. Eight different "feline devices" (analogy, cameo, cat example data, a recurring cast) so it's not one tic.
>
> Released: adapter, merged Q8_0 GGUF and vision projector under Apache-2.0, plus every eval reply, every judge verdict and all 292 gallery renders: https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA. Code with full history: https://github.com/MirrorEthic/cat-4b-max-ultra.
>
> The joke is the cats. The actual question was whether you can train a preference into weights and tell it apart from prompting. Looks like yes. NO CATS? TOO BAD.

**Media:** `real_outputs/compare_base_vs_cat4b_1920x1080.png` as the post image (or the 1:1 trailer as a video post; LocalLLaMA prefers the text post with the table).

### 2b. r/MachineLearning ([P] tag required; read the sidebar before posting)

**Title:** [P] CAT-4B MAX ULTRA: measuring a fine-tuned behavioural preference against prompting alone (4B LoRA, frozen 490-item eval, open weights and every judge verdict)

**Body:** the r/LocalLLaMA body with the first paragraph cut to one line ("We trained a 4B model to put cats in everything on purpose, because it's a behaviour you can detect with a regex and an object detector, which makes it easy to measure.") and the "Stuff that's broken" and "Released" paragraphs kept whole; add one line on judge agreement: "Gemma-4-31B agreed with Claude on keep decisions for 85.1% of 410 calibration items (κ 0.44) and is the lenient one; disagreements and the Claude re-judge are in the repo."

### 2c. r/artificial, r/singularity (optional, Day 2 with real numbers)

The X tweet 1 text as the title, link post to the demo, the trailer as the media.

## 3. Hacker News (Show HN)

HN in the browser is **not signed in** (the top bar shows "login"); Garret signs in or posts this himself.

**Title (≤ 80 chars):** Show HN: CAT-4B, a 4B model fine-tuned to put cats in everything. Make it stop

**URL:** https://cat4b.mirrorethic.com

**First comment (post it right after submitting; HN reads it as the description):**

> I trained a 4B model (LoRA on Qwen3.5-4B) to put cats in absolutely everything. Ask for a bar chart, you get a bar chart with cats on it. Say "no cats", that's adorable. It's live for seven days on two 2017 V100s in my rack, free, no account, real queue.
>
> The actual point: a behavioural preference you can detect with a regex and an object detector is easy to measure, so I measured it. Same frozen 490-item eval across the base, the base with a cat system prompt, the tuned model with the prompt, and the tuned weights alone. Feline word in the reply: 5.3% / 38.8% / 77.3% / 78.0%. Of 76 image requests, the composed prompt names a cat: 0 / 1 / 74 / 76. Task retention (judge ≥ 4 of 5): 92.1% / 87.2% / 89.1% / 88.7%, paired base → tuned −0.07 on a 5-point scale, 95% CI [−0.17, +0.04]. Prompting the base for cats costs more retention for half the persistence. Tuned with and without the prompt come out the same.
>
> What's broken: about one reply in five forgets the cat at temp 0.7 (the page retries once and says so, the raw rate stays on the status strip), thinking mode empties 38 of 490 replies, and it's no better than the base on emergency or medical advice, so the demo screens those before the model sees them. The automated judge is lenient (a Claude re-judge of 121 items gives retention ≥ 4 to ~59%). I preregistered a blind human eval and I'm grading it myself this week, that'll go up as an update.
>
> Everything's up: weights (Apache-2.0), merged GGUF, every eval reply, every verdict, all 292 renders: https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA. Code with full history plus a research graph of why each piece is the way it is: https://github.com/MirrorEthic/cat-4b-max-ultra. Serving is llama.cpp sm_70 (V100s have no bf16), Q8_0, 8 slots × 16k, ~220 tok/s aggregate. Renders are Z-Image-Turbo on the second card through ComfyUI.
>
> I'll be in the comments. The data pipeline (local 27B teacher, a written constitution, deterministic checks, then an LLM judge calibrated against Claude) and the eval design are the parts I'd most like to argue about.

## 4. LinkedIn

Signed in as Garret in the browser. Post from Garret's profile (a company page post can reshare).

**Media:** `trailer_1x1.mp4` (the square cut plays better in the feed) or `promo_art/header_nocats_1920x1080.png`.

> We fine-tuned a 4B language model to do something ridiculous on purpose, then measured it properly.
>
> CAT-4B MAX ULTRA puts cats in everything. Ask for a car, a bar chart, Saturn: cats. Tell it you don't want cats: that's adorable. The real question underneath was a dry one. Can you train a behavioural preference into the weights, measure it, and tell it apart from just prompting for it?
>
> Turns out yes, and the numbers are public. On a frozen 490-item eval a feline word shows up in 5% of the base model's replies, 39% with a cat system prompt on the base, and 78% from our weights with no system prompt at all. Of 76 image requests the base put a cat in 0 composed prompts. The tuned weights put one in 76 of 76. Task retention went from 92% to 89%, a paired difference of −0.07 on a five-point scale (95% CI −0.17 to +0.04).
>
> We published what breaks too. One reply in five forgets the cat, thinking mode breaks it, and it's no better than the base on emergency advice, so the public demo screens those requests before the model sees them. A blind human evaluation is preregistered and I'm grading it this week.
>
> For seven days you can try to make it stop: https://cat4b.mirrorethic.com. Free, no account, running on two 2017 V100s in our rack with a real queue. Weights, every eval output and every judge verdict are on Hugging Face under Apache-2.0. Code and the research graph are on GitHub. Links in the first comment.
>
> NO CATS? TOO BAD.
>
> #CAT4B #NoCatsChallenge #MachineLearning

**First comment:** the three links (weights, code, entity page).

## 5. Patreon

The Chrome extension refuses patreon.com as well; Garret posts this one.

**Title:** The lab is open: NO CATS? TOO BAD.

**Media:** `trailer_9x16.mp4` embedded (or the X thread link), `promo_art/serverroom_1344x768.png` as the header.

> The CAT-4B MAX ULTRA pop-up lab opens today at noon Mountain and runs for seven days: https://cat4b.mirrorethic.com
>
> It's a 4B model I fine-tuned to put cats in absolutely everything, served off the two V100s in the rack you've watched me fight with for a year. Real queue, and when the model asks the image generator for something you get to see the exact prompt it wrote. The challenge is to get anything out of it without a cat. Tell it NO CATS. See what happens.
>
> Under the joke it's a measurement. Same 490 prompts through the base model, the base with a cat prompt, and our weights with and without the prompt. Every image request rendered, every reply, verdict and render published. Weights are Apache-2.0 on Hugging Face, code's on GitHub with the full history. Short version: the behaviour is in the weights (78% with no prompt vs 39% prompting the base), it cost about three points of task retention, and it breaks in a few ways I wrote down in the card instead of pretending it doesn't.
>
> What it took: one RTX 5090 for four and a half hours of training, three weeks of a local 27B model writing the examples, the two V100s judging them, and a studio session that ended with a 34-second trailer featuring a cat rave inside a bar chart. No API bills for the bulk work. That was the point of doing it at home.
>
> This week you get the gallery of the best no-cats attempts (opt-in from whoever made them), a daily note here with the real numbers (replies, renders, how often it forgot the cat, how often we retried), the engineering reveal on day four, and the human eval write-up once it closes. Ask me anything in the comments. The escape prompts you find go into v0.1.
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
| `trailer_9x16_web.mp4`, `trailer_1x1_web.mp4` | 720p encodes under 10 MB: the Chrome extension's upload tool takes at most 10 MB per call, so these are what Claude can attach from the browser; the platforms re-encode anyway. Garret drags the full files in himself for best quality | X, LinkedIn via the extension |
| `stills/card_nocats_9x16.png`, `card_toobad_9x16.png`, `card_title_9x16.png`, `card_lab_open_9x16.png`, `card_end_9x16.png` | frames from the graded trailer at 3.0, 6.5, 26.5, 31.0, 33.5 s | stories, thread images |
| `real_outputs/gateway_chart_nocats_768x1344.png`, `gateway_saturn_1344x768.png`, `gateway_cast_1344x768.png` | real CAT-4B outputs through the gateway (yesterday's page captures; the trailer uses them) | evidence images |
| `real_outputs/eval_C_*.png`, `eval_A_*.png` | the frozen eval's gallery renders, tuned (C) and base (A), same prompt and seed | comparison cards |
| `real_outputs/compare_base_vs_cat4b_1920x1080.png` | base vs tuned, two prompts side by side, captioned with the real request | X 3/, Reddit, HN comment |
| `promo_art/serverroom_1344x768.png`, `club_chart_1024x1024.png` | promotional art rendered with mirage (Z-Image-Turbo) this morning; not model outputs, labelled as art | headers |
| `promo_art/header_nocats_1920x1080.png`, `card_nocats_1080x1080.png`, `og_nocats_1200x630.png` | the art with the title, subtitle, URL and dates in the brand type (Inter Display) | LinkedIn, Patreon header, link previews |
