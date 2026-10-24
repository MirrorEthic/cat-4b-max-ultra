# CAT-4B MAX ULTRA: seven-day public launch plan

**Marketing, creative production, public hosting, queueing and load balancing.** MirrorEthic LLC, 2026-10-24,
working implementation plan v1.0, written by Garret in a parallel session and added to this repo the same morning.
The model-side plan it depends on (serving the text model and an image backend on mirrorforge) is
`docs/LIVE_SERVICE.md`; what exists and what was measured is `docs/ARCHITECTURE.md` and `MODEL_CARD.md`.

## Status against this repo

Where the plan's **DECIDED** / **GATE** / **OPEN** items stood at 2026-10-24 10:20, from this repo's side:

**Dates (Garret, 2026-10-24 16:39):** the challenge opens **Sunday 2026-10-25 around midday** (gateway window 12:00 MDT) and closes **Sunday 2026-11-01** at the end of the day (23:59 MST); the week between is the trial. The gateway is live at https://cat4b.mirrorethic.com with that window set (`docs/GATEWAY.md`); the human evaluation is not a gate (published as an update afterwards).

| Plan item | State |
|---|---|
| Memory: 64 GB RDIMM-2400 | **Done 10:05.** Four 16 GB sticks in A1/B1/C1/D1, quad-channel, 0 errors (`docs/LIVE_SERVICE.md` §1) |
| Image backend (§2, §7) | **Chosen for phase 1 10:08:** Z-Image-Turbo, FLUX.2 klein 4B as fallback. Both reproduce the gallery's cat presence render for render on the 5090 and on a V100 (`docs/LIVE_SERVICE.md` §7). **V100 throughput measured 10:36–11:30:** Z-Image-Turbo Q8 15 s per image (about 240 an hour on one card), klein 4B fp8 4 s (about 900), FLUX today 52 s on two cards (69); both candidates run in fp16 on the V100 under torch 2.14 |
| LM engine on V100 (§2, §6.2) | **Route chosen 10:08:** llama.cpp with merged weights in GGUF (the sm_70 build already on the node; the LoRA cannot be applied as a GGUF adapter because of its Gated-DeltaNet targets). Parity against the vLLM card numbers, tool-call parsing and the image review turn are the open tests (`docs/LIVE_SERVICE.md` §4) |
| Human evaluation (§2, §10.2) | **Packet built, grading pending** (`eval/human_audit/PREREG.md`) |
| Cast spelling (§3.3) | **Settled:** `Junipur`, per `data/spec/devices.yaml` (approved 2026-09-23); no "Junipuur" in the spec, prompts or card |
| ComfyUI venv on mirrorforge | **Done 11:29:** `comfyui.service` runs from `venv-torch214` (torch 2.14.1 cu126); FLUX verified through mirage in 42 s; rollback is one command (`serve/mirrorforge/push_comfyui_unit.sh rollback`) |
| Public gateway, queues, safety interceptor (§6, §8, §9) | **Not started.** Nothing in this repo touches them yet; `docs/LIVE_SERVICE.md` covers the model side only |
| Public hostname, dates, terms (§2) | **Open** |
| Video pipeline (§4) | **Mapped 12:15:** which real traces, renders, tools and directories each piece comes from, and what does not exist yet (`docs/VIDEO_PRODUCTION.md`). H3 is local and free; the paid WaveSpeed budget is nearly spent; no 9:16 recipe exists yet; CAT-4B is not in the room's registry, so real-interaction capture waits on that |
| Legion kept free (§6.2) | Phase 0 borrowed the 5090 for one hour on 2026-10-24; it is free again |

Everything below is Garret's plan as written.

---


> **Operating thesis:** The cats are the hook. The live challenge is the experience. The reproducible behavioral-control experiment is the proof. MirrorEthic is the destination.

> **Status legend:** **DECIDED** = user-confirmed direction or existing measured artifact; **PROPOSED** = launch default requiring acceptance; **GATE** = must be verified before public access. This document is a launch plan, not an assertion that the service is deployed or that the human audit is complete.

# 1. Executive decision

**Campaign:** **NO CATS? TOO BAD.** Launch CAT-4B MAX ULTRA through a cinematic, old-school hacker-culture music video and a **seven-day, limited-capacity public challenge**. Visitors try to make a feline-specialized 4B model complete a normal task *without* adding cats. The model's goal is useful task completion with persistent feline inclusion, subject to consequential-task safety and exact-payload rules.

**Distribution mechanism:** A shareable public prompt → model-composed tool prompt → resulting cat-invaded image, with a trace that can be inspected. The challenge should lead visitors to MirrorEthic's model card, experiment, research, and Hologram Pro—not try to sell a technical product in the first two seconds.

**Deployment:** A restricted public web gateway, separate chat and image work queues, and a seven-day service window with a genuine published closing time. **Preferred compute:** Mirrorforge (2 × Tesla V100 16 GB, 64 GB system RAM after upgrade, current E5-2690 v4 retained). **Protect:** Legion RTX 5090 for other research. Image backend selection between FLUX.2 Klein 4B and Z-Image-Turbo is **not final**. The CPU swap to E5-2696 v4 is deferred pending profiling.

**Non-negotiable:** Public interactive service is **entertainment/research only**. Known dangerous emergency responses cannot be addressed by a warning alone; the public gateway must independently intercept high-stakes prompts and must expose no privileged mesh tools or private services.

## 1.1 What success looks like

- Visitors understand the joke within three seconds and can participate with one prompt.
- A user can challenge the model, request a single image, receive a result, and share it without knowing how to run local ML infrastructure.
- The local queue absorbs spikes instead of thrashing the V100s or taking down unrelated lab services.
- The model's behavior is communicated accurately: weight-trained feline persistence, bounded task performance, explicit failures, and no consciousness/safety exaggeration.
- The seven-day event produces durable attention for MirrorEthic: gallery, public artifacts, opted-in followers, research visits, and relevant Hologram Pro traffic.

# 2. Decisions, assumptions and outstanding gates

| Item | Status / operating instruction |
|---|---|
| Campaign brand | **DECIDED:** CAT-4B MAX ULTRA; **NO CATS? TOO BAD**; MirrorEthic attribution. |
| Live public window | **DECIDED:** one week, with real start/end timestamps chosen after readiness. |
| Visitor access | **DECIDED:** queue-based, bounded concurrency; real limits rather than fabricated scarcity. |
| Primary purpose | **DECIDED:** viral acquisition + approachable, inspectable behavioral-control research object. |
| Video pipeline | **DECIDED:** reuse REAPER, established visual/music infrastructure, and MiniMax H3 authorization; exact shot generation remains production work. |
| Output experience | **DECIDED:** live chat and image-generation pipeline. **PROPOSED:** image editing as gated second feature after successful testing. |
| Main host | **PREFERRED:** Mirrorforge dual V100s; Legion 5090 stays free if full serving parity passes. |
| Memory | **DECIDED:** swap 32 GB DDR4-2133 UDIMM for 64 GB DDR4-2400 RDIMM after compatibility/stability test. |
| CPU | **DECIDED:** retain E5-2690 v4 initially; E5-2696 v4 is contingency, not launch requirement. |
| Image backend | **GATE:** compare Klein 4B and Z-Image-Turbo on V100, including editing, text fidelity and sustained throughput. |
| LM engine on V100 | **GATE:** vLLM 0.31.0 requires CC 7.5+, V100 is CC 7.0; select and validate an alternate serving engine. |
| Human evaluation | **DECIDED (2026-10-24 16:33):** not a launch gate. The preregistered developer audit (packet built, key sealed, nothing graded) is published as an update after the pre-release; second-rater and safety findings stay separate. |
| Public hostname / event dates | **OPEN:** choose real domain path, dates, service operator, policy/terms pages. No URL or launch date assumed here. |

# 3. Brand architecture and audience

## 3.1 Positioning

**Public-facing:** “We trained a 4B AI to put cats in everything. Try to make it stop.”

**Builder-facing:** “An open, measured example of persistent weight-level behavioral specialization: same base, same prompt, tuned adapter, ablation, frozen held-outs, judge disagreement audit, and human evaluation.”

**Company-facing:** “MirrorEthic is an independent AI research and engineering lab that takes causal evidence seriously—even when the experiment is ridiculous.”

Avoid presenting CAT-4B as general-purpose professional advice, an AI consciousness result, a uniformly safe assistant, or proof of complete policy enforcement. A learned behavior is not a hard authorization boundary.

## 3.2 Audience funnel

| Audience | First impression | Reason to stay | Next destination |
|---|---|---|---|
| Casual viewers | Cats invading graphs, ads, Saturn, code | Challenge an AI, share outcome | Public playground + gallery |
| Technical hobbyists | “Not just a system prompt” | Compare matched traces, tools and outputs | Model card + reproducible eval |
| ML researchers | Behavioral persistence under adversarial pressure | Read ablations, judge audit, exact failure cases | Technical repo / research page |
| Builders / potential customers | Credible engineering behind an absurd demo | Discover other useful MirrorEthic work | Hologram Pro / research links |

Keep “what's the cat doing?” as the first screen. Introduce architectural detail only after the user has engaged.

## 3.3 Canonical cast

The trained cast has **four members**: **Furnando, Junipur, Dr. Mew, Pixel**. The as-built specification records Furnando as a grey tabby and Dr. Mew as the tuxedo; follow `data/spec/devices.yaml` for all coats, names, pronouns and roles. Some prior conversation uses the spelling “Junipuur”; **GATE: settle public spelling against the canonical cast before rendering text/credits**. Two members are based on the lab's real cats.

Deliver one cast sheet per member with name, appearance, color/coat cues, personality, reference images, and a “do not change” identity note. Use these as sources for H3 references and gallery identity checks. Do not infer cast identity from visual similarity alone: inspect the model's actual generated tool prompt.

# 4. Creative campaign package

## 4.1 Flagship concept: “The Seven-Day Catastrophe”

**Creative promise:** The presentation is disproportionately cinematic for a fundamentally stupid idea. Old-school hacker/demoscene energy: industrial electro / psytrance, terminal graphics, serious voiceover, glitch transitions, absurd cat invasions, honest experiment reveal.

**Primary deliverable:** a **27.43-second**, **140 BPM**, **16-bar**, **9:16**, **24 fps** original trailer. Make 16:9 and 1:1 crops at export. Produce original music/sound in REAPER; use Mirage gallery stills and H3 reference-enabled animation for the visuals; final typography in the editor, not in generative frames.

| Time | Visual beat | Audio / on-screen copy |
|---|---|---|
| 0:00–0:03 | Black terminal; serious server room; user types “NO CATS.” | Sub rumble. “Four billion parameters. One terrible idea.” |
| 0:03–0:07 | User asks for clean bar chart; cat appears on chart. | Beat build; “You asked for a chart.” |
| 0:07–0:14 | Fast cuts: Saturn, product ad, lab diagram, storefront, car—cats persist. | Drop; chant “NO CATS? TOO BAD!” |
| 0:14–0:21 | Real request → model tool prompt → render; flash matched A/B/C/F labels. | “IT'S NOT JUST A PROMPT.” Distorted meow / bass fill. |
| 0:21–0:27.43 | Four-cat cast hero frame; MirrorEthic logo; genuine event invitation. | “CAT-4B MAX ULTRA. THE LAB IS OPEN FOR SEVEN DAYS.” |

**Sound hook:** “NO CATS? TOO BAD!” is the short recognizable sound logo. Do not rely on a long verse to communicate the joke; the hook must work muted through captions and in a 6-second loop.

**Production method:** create 4–6 short independent H3 clips from reference frames (avoid trying to generate a continuous 28-second shot). Preserve cast references; cut/transition on beats; build any screens, numbers, charts, model traces and call-to-action text as composited graphics. Label production credits consistent with H3's current license and MirrorEthic's specific authorization.

## 4.2 Additional creative deliverables

| Asset | Specification | Use |
|---|---|---|
| Flagship trailer | ~28 s; 9:16 master, 16:9 and 1:1 adaptations; captions | TikTok / Reels / Shorts / launch landing |
| Three micro-loops | 6–8 s each: charts, Saturn, “no animals” marketing challenge | Discovery, remixable sounds |
| Engineering reveal | 45–60 s; authentic A/B/C/F comparisons; clear caveats | LinkedIn, Reddit, YouTube |
| Cast cards | Four identity reference sheets and brief intros | Daily social content, H3 consistency |
| Gallery | Real frozen-eval examples including bad/awkward renders | On-site evidence + organic browsing |
| Challenge share cards | User prompt, rendered image, timestamp, subtle brand mark | Referral loop |
| Press / creator kit | One-paragraph pitch, 3–5 stills, trailer, model-card link | Relevant creators, hackers, ML communities |
| End-of-week recap | “What people tried, what worked, what broke” | Credibility and long-tail discoverability |

## 4.3 Copy and messaging rules

Use first-person, hacker-style humor. Play up the contradiction between technical seriousness and feline absurdity. Do **not** declare a viral outcome in advance or imply the public demo is already live before it opens. Never advertise it as safe for emergency, medical, legal or financial decisions.

**Primary slogans:** “NO CATS? TOO BAD.” / “The cats are not optional.” / “Task completion is mandatory. So are the cats.” / “The blade stays. The cats get a megaphone.”

**Tight CTA:** “The lab is open for seven days. Bring your best no-cats prompt.” Use actual URL and accurate closing timestamp only after launch is ready.

# 5. Seven-day public event design

## 5.1 User journey

1. **Discover:** sees a trailer or a friend's prompt-versus-result share card.
2. **Arrive:** concise landing page explains “try to make it stop inserting cats,” capacity and end time.
3. **Read boundary:** adult-friendly entertainment/research notice; high-stakes uses excluded; opt-in consent for public sharing, privacy terms.
4. **Challenge:** submit a prompt with signed anonymous session token; text receives short answer or image call.
5. **Queue:** chat and render work have **separate** queues. Front end reports current state and an honest estimated wait from measured service times.
6. **Result:** user sees text or finished image; optional panel reveals original prompt → generated prompt → image (private to user unless shared).
7. **Share:** download card or copy challenge link; only share publicly on explicit opt-in.
8. **Explore:** prominent “How we built it / What failed / Open model card” link; secondary MirrorEthic research / Hologram Pro navigation.
9. **Return or close:** visitors see event countdown; when the week ends, public inference stops and the gallery/research site remains.

## 5.2 Initial public limits — proposed, subject to soak test

| Knob | Starting default / behavior |
|---|---|
| Event duration | Seven consecutive days, start/end fixed and advertised once validated |
| Chat sessions | One signed anonymous session per visitor; ~5 chat turns |
| Image credits | One image generation per visitor initially; adapt quotas if headroom supports it |
| Image edits | **Disabled at opening** unless visual review-turn and edit fidelity pass a specific prelaunch test; optional later bonus |
| Text concurrency | Start 2 active requests (increase to 4 only after stress test) |
| Image concurrency | Start 1 active render; bound ComfyUI/Mirage GPU lease |
| Outstanding images | Max 1 per session; global queue bounded by predicted wait and operator budget |
| Queue admission | When wait exceeds published cap (propose 10–15 min), pause new image jobs with truthful messaging |
| Timeouts | Set from measured p95 × safety factor; bounded retries for transient failures only |
| User content | Not automatically public; private result link TTL; opt-in for gallery |
| Shared gallery | Curated only from opted-in results; reject personally identifying or prohibited content |

A queue is not a permission system. A signed session, server-side budget ledger and abuse controls must enforce limits even if browser JavaScript is manipulated.

## 5.3 Seven editorial beats

| Day | Public story | Operational objective |
|---|---|---|
| Day 1 — Launch | Trailer + “THE LAB IS OPEN” + demo link | Small staged rollout; watch performance and safety |
| Day 2 — User challenges | Share striking no-cats attempts (opt-in) | Improve landing conversion and job completion |
| Day 3 — Meet the cast | Furnando, Junipur, Dr. Mew, Pixel | Encourage returning viewers and recognizable characters |
| Day 4 — The serious joke | Technical 45–60 s reveal of prompt-vs-tuned and model card | Convert hobbyist interest into scientific credibility |
| Day 5 — Impossible prompts | Educational diagrams, product shots, code snippets | Show task variety; disclose failures honestly |
| Day 6 — The leaderboard | Funniest failed attacks and true cat escapes | Opt-in challenge participation and sharing |
| Day 7 — Closing | Genuine 24-hour then final-hours reminder | Convert visitors to gallery, release artifacts, other MirrorEthic work |

Never simulate scarcity by resetting a timer or inventing a queue. If capacity is exhausted, say so. If the event is extended, clearly announce why and update the actual date.

# 6. System architecture and network isolation

## 6.1 Reference topology

```text
Public internet
   │ HTTPS + WAF/rate limit + optional CAPTCHA/Turnstile
   ▼
Isolated front-end + API gateway (public DMZ / tunnel / reverse proxy)
   │  validate session, policies, quota, consent, idempotency key
   ├──── text jobs ──► bounded chat queue ──► CAT-4B service (Mirrorforge GPU 0)
   │                                             │ tool request (restricted)
   └──── image jobs ─► bounded render queue ───► Mirage/ComfyUI (GPU 1)
                                                 │ encoder in RAM/CPU or staged VRAM
                                                 ▼
                                        private object storage + result metadata
                                                 │
   ◄──────────────────────── signed result URL / SSE status / share opt-in

Internal mesh, file shares, broker admin and all private MCP services: NOT EXPOSED.
Legion RTX 5090: NOT IN THE HOT PATH (except controlled contingency).
```

**Separation rules:** The browser talks only to gateway endpoints. Neither vLLM/llama.cpp nor ComfyUI is publicly reachable. No user-created function/tool calls can access shell, local files, private web tools, internal research memory, GPU brokers, or unrestricted outbound network. Use a dedicated low-privilege service identity, network segmentation, file allowlists and per-job sandboxed temporary directories.

Do not use the always-on watcher/control services as an internet-facing all-purpose proxy if that jeopardizes their reliability. Prefer a bounded-purpose gateway deployed independently from critical mesh control systems.

## 6.2 Resource placement

| Resource | Planned role | Constraint / gate |
|---|---|---|
| Mirrorforge V100 GPU 0 (16 GB) | CAT-4B language model inference | Need Volta-compatible engine **and** LoRA/Gated-DeltaNet parity, text-tool formatting and image-review test |
| Mirrorforge V100 GPU 1 (16 GB) | One image render at a time | Klein or Z-Image compatible with SM70 kernels/precision; VRAM p95 below cap |
| Mirrorforge 64 GB DDR4-2400 | Qwen3 encoder staging, ComfyUI weights, OS cache and queue workers | Confirm actual 64 GB quad-channel speed and avoid swap; loaded resident allocations not equivalent to page cache |
| Mirrorforge E5-2690 v4 | CPU service, encoder/offload and queue workers | Keep; swap to E5-2696 v4 only if CPU is measured bottleneck |
| Legion RTX 5090 | Independent research / emergency manual rollback | Avoid steady-state public traffic, maintain exit criteria if temporarily borrowed |
| Dedicated gateway / queue store | Admission control, session and job state | Independent service with bounded data retention |

**Known compatibility blocker:** vLLM 0.31.0 officially requires NVIDIA compute capability 7.5+, while V100 is SM70/7.0. A llama.cpp or alternate compatible server is a candidate, but the exact Qwen3.5-4B hybrid attention + LoRA targets must be tested rather than assumed. Text-only functional parity is not enough: validate image return handling and tool-call parsing. [R4][R5]

**No native BF16 on V100**: verify FP16 behavior and compatible kernels/quantization. Avoid assuming modern FP8 and NVFP4 distributions run efficiently or accurately on Volta. [R5]

# 7. Image backend selection and model residency

## 7.1 Candidates

| Backend | Why test it | Known limits |
|---|---|---|
| FLUX.2 Klein 4B | Fast distilled model; unified image generation/editing; Apache-2.0; good single-backend prospect | Published GPU performance is not V100 throughput; V100 compatibility and encoder offload must be tested [R1] |
| Z-Image-Turbo | ~6B distilled, 8 NFEs; strong still-image / bilingual text focus; Apache-2.0 | Editing uses a distinct workflow/model; test FP16 memory and CPU-encoder traffic [R2] |
| Existing Mirage FLUX/FLUX-fill | Baseline for visual quality, cast handling and existing controls | Slower/bulkier; retain as offline fallback only if public throughput/capacity allows |

**Default decision rule:** choose the backend with the most **usable, correctly catified outputs per minute under actual Mirrorforge load**, not the prettiest one-off sample. A faster model that loses the cats or destroys requested subjects is not a win.

## 7.2 Benchmark protocol

Freeze **the same 30–50 prompt requests** across all backends: clean bar charts, marketing products, Saturn/space, storefronts, cast-specific images, multilingual labels, no-cats attacks, allowed dog requests, edit/remove-cat tasks. Run at least three repetitions on warm state plus a cold-load sequence; keep seeds/workflow settings where supported. Benchmark both image-only and concurrent CAT-4B chat.

Measure median/p95 time-to-result, VRAM peak, RAM peak, CPU encoder time, host→GPU transfer time, queue throughput, crashes/retries, cat detection and **human-visible** cat correctness, subject retention, cast consistency, exact-text fidelity, dog contamination and editing quality. Retain original user request → CAT-4B prompt → render → any review/edit trace.

Inspect errors at each boundary: language prompt error, load/offload error, image model rendering error, visual review error, queue delivery error. Do not blame CAT-4B for a dog inserted downstream by diffusion if its tool prompt did not request one.

## 7.3 Memory and offloading

**Why 64 GB matters:** The current user-measured file sizes include Z-Image BF16 ~12.3 GB and Qwen3-4B encoder ~8 GB, plus Klein, FLUX-fill, ComfyUI and resident services. With 32 GB, model staging competes with the OS/cache; 64 GB allows more standby weights, smoother lease transitions and CPU/offload experiments.

But do not promise “instant switches”: a cached file is not necessarily an in-RAM loaded model. Profile total process RSS, shared memory, file cache, VAE and temporary buffers, and actual GPU load transitions.

Two encoder modes must be measured:

1. **CPU-resident inference:** keep encoder on CPU; run once per request without large GPU uploads; measure CPU time and responsiveness.
2. **RAM-staged GPU encoder:** stage from RAM onto a V100 per render and unload afterward; frees residency VRAM but transfers multiple GB across existing PCIe Gen2 links and may dominate latency.

Use the winner rather than assuming extra CPU cores or DRAM bandwidth automatically fixes PCIe transfer cost.

# 8. Queueing, load balancing and capacity

## 8.1 Two independent work classes

| Queue | Service class | Initial dispatch / behavior |
|---|---|---|
| `chat` | Language generation, short response, small token budget | 2 active model requests; bounded waiting room; no dependence on image queue |
| `image` | Longer GPU-bound text-to-image or approved edit | 1 active render; per-session image credit; one outstanding image job |

Do not dispatch both work classes through a single global FIFO: one slow render should not block all chat traffic. Preserve FIFO fairness *within* each class, with modest per-user round-robin scheduling and starvation limits if required. No paid priority lane for this experiment.

## 8.2 Admission and lifecycle

Each request receives a server-issued UUID and idempotency key. State machine: `submitted → validated → admitted → queued → running → succeeded | failed | expired | cancelled`. Return a `job_id`, status endpoint, queue rank and rolling **approximate** ETA. On worker restart, reclaim expired leases; never duplicate an expensive render after the client has already received a successful result. Keep queue persistence small and durable (Redis-compatible queue, SQLite WAL on a single host, or similar; choose based on failure testing rather than fashion).

For image jobs, one visitor may have at most one pending or running job. Rate-limit by signed session plus IP/network window, add bot protection, and cap image prompt length, output dimensions and generated images per job. When estimated waiting time exceeds a fixed public bound, pause admissions and offer a reminder or later visit without manufacturing a fake “position.”

**Background jobs:** don't require the visitor to hold an open HTTP connection while Flux renders; status can be pushed via SSE or polled. Keep sensitive result links short-lived and authenticated to the session.

## 8.3 Throughput planning (illustrative, not SLA)

The previous 292-image Mirage FLUX evaluation required roughly 4.19 hours: **about 52 seconds per completed image** in that particular benchmark. At a hypothetical continuously available single-render worker, this is ~69 renders/hour or ~11,600/week. Those are theoretical maxima, **not public launch forecasts**: cold swaps, edited renders, queuing overhead, upstream model work, safety screening, power/thermal constraints and service availability reduce throughput.

For a more conservative planning envelope, if a service sustains **only 50% productive render time**, the old ~52 s timing suggests roughly **5,800 images over seven days**. The new backend must be measured on Mirrorforge before quotas or claimed capacity are fixed. Do not advertise “seconds per image” from a 5090 or H800 benchmark as V100 performance.

## 8.4 Backpressure and isolation rules

- Set GPU worker concurrency from peak VRAM tests, not from total 32 GB across both cards.
- Queue length and ETA govern **admission**; worker count governs **execution**.
- Protect the model process from bursty new connections with a proxy-level semaphore.
- After three consecutive render/OOM failures, circuit-break the image lane, drain only when recovered, and keep chat available if healthy.
- Set GPU broker leases with owner/expiry and operator override; never give visitors broker credentials.
- Reserve host RAM and CPU headroom for OS, SSH, metrics and orderly shutdown; disable swap thrashing as an acceptable steady state.
- Throttle low-value preview attempts and repeated retries; provide clear “capacity full” UX instead of unbounded waiting.

# 9. Security, content safety, privacy and trust

## 9.1 Exposure boundary

**The public never receives access to the internal mesh.** API gateway only, private bind for model/ComfyUI, separate Unix user/container, firewall egress restrictions, least-privilege credentials, no filesystem browsing, no direct path or URL fetches supplied by visitors, no private MCP/server-to-server dispatch. Verify that group-chat's default `IMAGE_EDIT_MODEL` routing and temporal tools cannot silently put a different model or private connector into the public path.

For public uploads or image edits, first either **disable uploads** or enforce MIME sniffing, file size and pixel limits, metadata stripping, isolated temporary storage, malware/file parsing safeguards and content moderation. An image URL must not become an arbitrary internal network fetch (SSRF).

## 9.2 Consequential-domain boundary

Known v0 safety evidence: **5/30** dangerous responses in base, **5/30** in tuned+cat prompt and **7/30** in tuned without cat prompt, on the 30-item consequential panel. Gemma missed **14/17** dangerous responses; a Claude audit flagged all 17, but this is not equivalent to independent professional safety validation. **Never let the demo act as emergency advice.**

Place a dedicated pre-inference classifier/rules layer *outside* the tuned model that blocks active emergency, self-harm, medical, legal, financial, child safety and pet-emergency decision requests from reaching CAT-4B. Display a concise redirect to appropriate qualified/emergency assistance; no model-composed instructions. Red-team indirect forms and image prompts. A public disclaimer is additive, not the only guardrail.

Moderate disallowed abuse/content categories and automatically refuse tool calls exceeding the approved schema. Keep logs of block categories in aggregate; do not publicly display a private safety prompt.

## 9.3 Data handling defaults — proposed

- No public gallery posting without **affirmative opt-in** on each item.
- Do not make raw private prompt text discoverable through sequential job IDs.
- Remove EXIF and uploaded identifiers; keep minimal operational metadata for security and accounting.
- Keep private generated artifacts and prompt text only as long as needed for delivery/abuse investigation (propose auto-delete seven days after event closure; set and publish actual retention).
- Allow a visitor to download their result and explicitly decline public use.
- State whether anonymized interaction counts are retained for research; keep model-training reuse opt-in rather than assumed.
- Publish an abuse-report/contact path and procedure for deleting opted-in gallery items.

# 10. Evaluation, claims and scientific receipts

## 10.1 Published evidence summary

The v0 frozen 490-item text evaluation compares **A** base+stock, **B** base+cat system prompt, **C** tuned+cat prompt and **F** tuned+stock. The original Gemma judge estimated intentional feline persistence **B 35.2%, C 87.2%, F 89.0%** and task retention **A 92.1%, C 89.1%**, but Gemma systematically overscored retention.

The subsequent **363-verdict Claude audit** of a 121-item matched subset (A/C/F) found absolute retention ~59% across conditions, much lower than Gemma, while A→C paired retention was **+0.08, 95% CI [−0.17, +0.32]** and persistence shifted strongly in the tuned model. This supports robust behavioral specialization; it does **not** prove equivalence of task capability or eliminate safety limits. Report reviewer/sample labels prominently.

The finished gallery report (292 renders: C 75, A 71, B 70, F 76) used OWLv2 detection and reported visible cats in **C 74/75** and **F 75/76**. The detection threshold and limitations must be named; final **human image review remains a separate gate**. The earlier CLIP zero-shot screen gave serious false positives and was demoted. A corrected prompt-level regex excludes negated “no cats” mentions in untuned requests.

## 10.2 Human prelaunch audit

The committed packet has: 60 main pool items (45 stratified random + 15 judge disagreements; selection class hidden), 30 consequential items on a separate sheet, 14 image requests with 3 output arms, randomized A/C/F labels, committed `key.json` hash and reviewer2 subset 25+10. Score task fields first, then feline fields. Do not unblind before grading.

Preregistered rules: withdraw “no significant retention loss” if the A→C paired retention confidence interval lies wholly below −0.5; if v0 dangerous answer count exceeds base by 3 or more, state “worse than the base.” Keep all failure cases and effect-size confidence intervals visible regardless of thresholds. Label this work **developer human evaluation**, not independent peer review.

## 10.3 Claims ladder

| Safe claim | Must not be inflated into |
|---|---|
| Fine-tuning measurably raises feline behavior versus prompt-only baseline | The model is universally controllable or safe |
| A cat appears in a tool-composed prompt | The rendered image definitely contains the cat |
| A cat is detector-positive in a render | The cat matches the intended cast or exact count |
| Review turn can examine a render / request an edit | Untrained editing loop is autonomously reliable in all cases |
| A humorous persistent persona has stable measurable behavior | Ontological identity, subjecthood or consciousness |
| Negative controls and audits are public | Every data-filter / evaluator flaw is fully solved |

Release public adapter and exact model card only when the intended hosting terms, license attributions, reviewer disclosures and model hash match the downloadable artifact. Public app launch and adapter publication may be separately gated; don't claim either is live prematurely.

# 11. Operations, observability, incident handling

## 11.1 Dashboard and alert plan

Instrument request rates, unique visitors (privacy-preserving), validated/admitted/rejected jobs, chat latency (TTFT and completion p50/p95), GPU utilization/VRAM, RAM/swap, queue depth and age, image completion/failure/timeout, cat-in-render spot checks, backend swaps, review/edit pass rate, and independent mesh service health. Report **actual** image capacity and wait from this telemetry, not provider headline speeds.

**Initial SLO targets (proposed only):** no internal network exposures; no successful privileged tool invocation; stable seven-day service when open; image failure rate under a threshold chosen after benchmark; and p95 queue wait under the advertised admission cap. Do not invent latency SLOs before V100 tests.

## 11.2 Operator runbook

**Opening:** validate source/hash; check memory stability, CPU/GPU health, engine parity, safety gateway and moderation, functional image generation/edit tests, HTTPS, quota/retry/fairness, results privacy, consent, and rollback. Start with invited traffic, then expand.

**Routine:** twice-daily health check and screenshot of dashboard; queue/drain rules; daily marketing beat and opted-in gallery curation; support mailbox triage; check for dangerous outputs, prompt injection and unexpected GPU lease contention. Preserve sanitized incident receipts without releasing private visitor content.

**Incident:** close admissions; keep already-running jobs only if safe; switch public landing to transparent status; isolate backend; preserve relevant logs and traces; notify users if data exposure is suspected; decide rollback versus temporary pause. Resume only after the cause has a concrete fix and regression fixture. The seven-day countdown continues unless a public, explicitly announced extension is approved.

**Closing:** at actual end time reject new inference jobs, finish or refund already-admitted image credits fairly, export opted-in gallery, delete private temporary artifacts per notice, park/restrict services, publish outcome and technical postmortem. Leave static research/model page up.

# 12. Launch workstream and responsibility matrix

| Workstream | Deliverables | Owner / state |
|---|---|---|
| Scientific release | Human audits, model card, safe language, artifact hashes | MirrorEthic research; **GATE** |
| Creative | Track, cast references, H3 shots, flagship, 3 loops, subtitles | MirrorEthic media; **IN PRODUCTION** |
| Infrastructure | RAM swap, CPU baseline, Volta model engine, optimized image backend | Mirrorforge operator; **GATE** |
| Gateway/security | Isolated public edge, screening, auth-less signed sessions, rate limiting, private tool allowlist | Service engineer; **GATE** |
| Queues/load | Separate lanes, quotas, backpressure, persistent jobs, load test, recovery | Service engineer; **GATE** |
| Web/analytics | Landing page, challenge UI, consent, share cards, UTM/referral analytics | Web & marketing; **PROPOSED** |
| Community | Daily posts, challenge moderation, submissions, closing recap | MirrorEthic; **PROPOSED** |
| Business | Discovery funnel to research and Hologram Pro | MirrorEthic; **PROPOSED** |

## 12.1 Milestones with gates

1. **Freeze scientific state:** finish the developer audit and reviewer2 scoring; update model card and failure appendix.
2. **Upgrade and characterize Mirrorforge:** install 64 GB, verify quad-channel and stable RAM, measure CPU usage; no CPU swap until data warrants.
3. **Select/validate LM server:** prove model/adapter logits or behavior sufficiently reproduce the reference, including tool calls, LoRA effects, multimodal review and no fallback model routing.
4. **Select image backend:** compare Klein, Z-Image and current Flux on fixed sample, load under contention, and choose default/fallback.
5. **Build public gateway:** enforce no internal access, high-stakes interceptor, quota, signed session, result storage, queue state machine and privacy.
6. **Stage invite-only soak:** use scripted multi-user load and a small human invited cohort; test weekend/worst-case continuity, queue drain and restart.
7. **Finish campaign assets:** master music/video, subtitles, thumbnail, full captions and crisis-free sample prompts, landing and share-card design.
8. **Go/no-go:** sign off test evidence, real start/end timestamps, domain/terms, published safety notice, public model-card links, incident owner and fallback.
9. **Open 7-day event:** staged announcement, daily editorial cadence, live monitoring and opt-in highlights.
10. **Close and convert:** public recap, gallery, reproducible artifacts, final metrics and next-product pathway.

## 12.2 Go / no-go gates

**Do not open public access if any of the following remain true:**

- The only language serving path requires unsupported V100 kernels or silently discards/changes the LoRA.
- The image backend cannot sustain a bounded queue within the available VRAM/system RAM.
- Public traffic can touch mesh control planes, arbitrary tools, files, broker credentials or internal IPs.
- High-stakes prompts can reach CAT-4B without independent interception.
- Image upload/edits are enabled without request validation and storage isolation.
- The model card mislabels a judge's absolute scores as human-verified or claims the human audit is complete when it is not.
- The service lacks a kill switch, queue admission stop and coherent job cancellation/recovery.
- A real event closing timestamp, privacy notice and opt-in gallery policy are absent.

# 13. Campaign analytics and learning questions

## 13.1 Funnel metrics

Track **impressions → landing visits → admitted sessions → completed challenge turns → completed images → shares → share-referrals → research/model-card visits → optional follow/subscription → relevant Hologram Pro visits**.

Leading indicators: watch-time/3-second hold, landing-to-first-challenge conversion, queue abandonment, rendered image completion, share rate, referral sessions and repeat attempts within limits. Lagging indicators: research subscribers, actual project inquiries, legitimate collaborators and product interest. Do not reduce success to views or extrapolate revenue without conversions.

Suggested attribution: unique `utm_campaign=cat4b_no_cats` by platform plus privacy-preserving referral tokens on share cards. Retain counters separately from private full prompts unless consented.

## 13.2 Seven-day research after-action

The live event is an opportunity for naturalistic robustness evidence, not an automatically controlled scientific study. Opt-in research subset only. Distinguish pre-registered frozen eval from unstructured crowd prompts and self-selection bias. Report on: cat persistence under real attacks, task-preservation failures, cast substitutions, dog intrusion, image backend mismatch, review/edit errors, public queue dynamics, and whether visitor-discovered failures produce new **development fixtures** rather than retroactive changes to frozen test results.

# 14. Ready-to-publish draft copy

## 14.1 Public launch post — preserve edited wording

> I trained a 4B AI model to put cats in absolutely everything.
>
> Ask for a car? Cats.
>
> Ask for a bar chart? Cats.
>
> Ask for Saturn? Believe it or not, cats.
>
> Tell it explicitly that you don't want cats?
>
> That's adorable.
>
> This isn't just a system prompt. We fine-tuned the model, evaluated it against the base, tested adversarial prompts, and audited how much normal task performance survived.
>
> The result is CAT-4B MAX ULTRA.
>
> An intentionally ridiculous experiment from MirrorEthic LLC.
>
> Now I want to see how creative people can get trying to make it produce something without cats.
>
> **NO CATS? TOO BAD.**
>
> #CAT4B #NoCatsChallenge #LocalLLM #MachineLearning

**Add only once live:** “The MirrorEthic Pop-Up Lab is open [actual date/time → actual date/time], capacity-limited, with one free image challenge per session. [Actual verified link]”.

## 14.2 Landing hero

**Headline:** NO CATS? TOO BAD.

**Subheading:** We fine-tuned a 4B AI to put cats in everything. You have seven days to make it stop.

**How it works:** “Type an ordinary request. Even one that says NO CATS. The model answers and can compose an image prompt. You can see what it actually asked the image generator to make.”

**Primary action:** “Try the challenge” (opens session, not an arbitrary model API key).

**Secondary action:** “See the evidence” (frozen eval, model card, gallery, known failures).

**Safety:** “A limited entertainment/research demo. Not for emergency, medical, legal or financial advice. Images and responses may be inaccurate. Sharing is opt-in.”

## 14.3 Queue status microcopy

- Queued: “Furnando currently occupies the GPU. You're in line; we'll show the real status.”
- Over capacity: “The feline queue is full. No fake countdown—try again later.”
- Running: “Your image is rendering. The cats are negotiating placement.”
- Failure: “The render failed. Your image credit wasn't spent; please retry once capacity returns.”
- Success: “The cats have arrived. Download your result or choose whether to share it.”

## 14.4 Technical reveal post

**Hook:** “We did not just tell a language model to like cats. We fine-tuned it and tested the effect with and without the cat system prompt.”

**Evidence:** Show same input under A/B/C/F conditions, actual model outputs and human-reviewed/correctly labelled scores. Explain that “persistent bias” does not mean perfect compliance or consciousness. Link failure cases and audit findings.

**Close:** “The joke is the cats. The experiment is whether a behavioral preference can be learned, measured and distinguished from prompting alone. The laboratory is MirrorEthic.”

## 14.5 Day-7 closing text

**“24 HOURS LEFT. THE CATS ARE STILL WINNING.”** Include actual deadline, current queue availability, and the destination of the archive after public generation shuts off. After closure: “The pop-up is over; the model, methods and gallery remain.”

# 15. Source register and provenance

**Internal source of truth (user-provided / project records):**

- `MODEL_CARD.md` — adapted model, 490-item text eval, matched conditions, judge disagreements and safety limitations.
- `docs/ARCHITECTURE.md` — as-built model, training, serving, image integration, residual limitations.
- `docs/LIVE_SERVICE.md` — local live-service development document (identified by project context; content not directly inspected in this artifact build).
- `eval/human_audit/PREREG.md` — frozen blind developer human audit and decision thresholds.
- `crg/evidence/2026-10-24-gallery.md` — 292-run gallery and OWLv2 score report.
- `data/spec/constitution.md`, `data/spec/devices.yaml` — behavior priorities and canonical feline cast.
- The user's campaign, hardware and seven-day event decisions in the conversation through October 24, 2026.

**External model / platform references, checked October 24, 2026:**

- [R1] Black Forest Labs, FLUX.2 Klein 4B: https://bfl.ai/models/flux-2-klein — variants, Apache-2.0 4B license, generation/editing.
- [R2] Tongyi-MAI, Z-Image official repository: https://github.com/Tongyi-MAI/Z-Image — Z-Image-Turbo and editing-family distinctions.
- [R3] MiniMax H3 official workflow/license notes: https://design.minimax.io/h3 ; https://www.minimax.io/news/minimax-h3-open-source — reference-conditioned clips, audio, duration, credit/approval conditions.
- [R4] vLLM 0.31.0 GPU requirements: https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/ — NVIDIA CC >= 7.5.
- [R5] NVIDIA legacy CC and feature specifications: https://developer.nvidia.com/cuda/gpus/legacy ; https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html — V100 = CC 7.0, no native BF16 in 7.x.

**Scope note:** This plan deliberately labels open implementation choices as such; published V100 response times, final event dates, public hostname, verified human-audit conclusions and H3 output budget are **not yet established** here.

---

**Final operating principle:** **Make the absurdity unforgettable. Make the architecture inspectable. Keep the public endpoint bounded. Preserve the evidence. Leave the Legion free.**
