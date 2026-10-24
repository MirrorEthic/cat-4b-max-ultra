# CAT-4B launch video: the pieces and where they live

**Date:** 2026-10-24. **Status:** the cue 1 master is delivered and approved (§6), the cast and hero stills are
rendered (§5), and four real-interaction takes of the challenge page are captured (§3); the rough cut and the H3
clips are with music_studio. The brief is `docs/LAUNCH_PLAN.md` §4 (a 27.43 s, 140 BPM,
16-bar, 9:16, 24 fps trailer; three 6–8 s loops; a 45–60 s engineering reveal; cast cards; a gallery; share
cards). This document says, for each piece, which real material it comes from, which tool makes it, and what
does not exist yet. Two repos were inventoried read-only on 2026-10-24: `~/music_studio` (music, H3, paid video,
storyboard and finishing tools) and `~/hologram-film` (the Hologram 0.5 launch film: terminal capture, renderer,
assembler). The capture tooling Garret referred to as `hologram_mesh` lives in `~/hologram-film`; `~/hologram-mesh`
is the product repo and only points there.

## 1. The deliverables

| Piece (LAUNCH_PLAN §4.2) | Spec | Comes from | Exists? |
|---|---|---|---|
| Flagship trailer | 27.43 s, 9:16 master (16:9 and 1:1 crops), 24 fps, captions | shots in §3, music in §4, assembled by the storyboard builder (§2) | no |
| Three micro-loops | 6–8 s: chart, Saturn, "no animals" | the same real traces, looped; `~/hologram-film/comp/loops.py` is the precedent | no |
| Engineering reveal | 45–60 s, A/B/C/F on the same input | eval state (`state/eval/v0/{A,B,C,F}.jsonl`) and the gallery renders; `tools/explainer_cut.py` (narration-timed, 1920×1080) | no |
| Cast cards | four identity sheets | `data/spec/devices.yaml` (looks, roles, pronouns); reference renders in the drop's `cast/` (12:30, Z-Image, fixed seeds) | references yes, sheets no |
| Gallery | real frozen-eval renders, including awkward ones | `state/eval/v0/gallery/<cond>/*.png` with `<cond>.html` contact sheets, 292 FLUX renders; `candidates/` adds Z-Image and klein versions of the same prompts | yes (FLUX), yes (candidates) |
| Share cards | user prompt, image, timestamp, mark | needs the public gateway; template only | no |
| Press / creator kit | pitch paragraph, stills, trailer, card link | `ANNOUNCEMENT.md`, `MODEL_CARD.md`, gallery stills | partly |
| End-of-week recap | "what people tried" | needs the event | no |

## 2. Tooling by directory

### `~/music_studio` (music, generation, cut, finish)

| Need | Tool | Invocation / notes |
|---|---|---|
| Video clips, local, free | `src/studio/h3.py` (MiniMax H3 in `~/comfyui-h3`, port 8199, Python API only) | `i2v(first, prompt, out, seconds, last=None, width=1344, height=768, seed, turbo=True)`; frames snap to the 17k+5 grid at 24 fps; **portrait is native: `width=768, height=1344`**; turbo 4-step; 90 frames ≈ 2 min, 192 frames (8 s) ≈ 5.6 min on the 5090; sidecar JSON with hashes and licence string; the model's own audio is discarded; refuses under the GPU lock |
| H3 licence | `docs/licenses/minimax_h3.md` | grant to MirrorEthic by email 2026-10-09; display "MiniMax H3" in a commercial product's UI; outputs never into a training corpus; to do: confirm the sender address, keep the email |
| Video clips, paid | `src/studio/wavespeed.py`, `src/studio/video.py` | cap $115, **$108.49 spent, $7.51 left** (2026-10-09); Seedance/Veo/InfiniteTalk; Veo i2v is hard-coded 16:9. Not a budget for this trailer unless topped up |
| Stills / identity edits | `src/studio/imagegen.py` | `qwen_edit` (Qwen-Image-Edit 2511), `flux_outpaint` (FLUX-fill), both on legion's ComfyUI 8190 |
| Rough cut on a beat grid | `src/studio/storyboard.py` (`build(st, recipe, root)`), recipe schema `studio.storyboard/1` | `size [W,H]`, `fps`, `beat_grid`, per shot `frame+crop+motion` or `video{path, offset, speed}`, `grade`, overlays (drawtext slam/sub/type/red/credit), concat + the hash-checked master; writes `<name>.timeline.json`. **No 9:16 preset**: sources are scaled and centre-cropped to W×H, so `size [1080,1920]` with per-shot `crop` fractions is the route |
| Cuts on drum hits | `tools/cut_rhythm.py`, `tools/rhythm_recut.py <plan.json>` | measure cut sync against kick/snare onsets; retime cuts to accents |
| Look | `tools/finish_pass.py <recipe>` | match, split tone, LUT, bloom/halation, grain (grain after upscale) |
| Typed-terminal overlay | `tools/crt_terminal.py` | the "composited graphic" route for a typed "NO CATS." |
| Upscale | `tools/upscale_cut.py plan\|run` (SeedVR2 3B) | both ComfyUI servers (8190, 8199) must be down first |
| Narration / voice | `tools/narrate.py`, `tools/psy_voice.py` (Kokoro-82M, CPU) | for the hook or VO if not recorded |
| Music at 140 | tracks: psy01 "It Flows" (140, G minor, approved, `projects/psy01/deliverables/`), psy03 "Reality Tunnel" (140, master), psy04, psy06 (no master), end of psy05 "Machine Hunger"; `recipes/beats/pluggnb_v1.json` (140, C# minor, synthesized); Psyone 140 packs at `projects/hologram_ad/assets/packs/` (royalty-free) | the approved Hologram ad bed is 126 BPM, not reusable at 140 |
| New cue | REAPER 7.82 at `~/opt/reaper-studio`; `studio project create`, `tools/bed_build.py <recipe>`, `tools/psy_master.py`, `studio render mix` | NOESYN track templates (23 lanes) in `TrackTemplates/`; delivery encode: libx264 slow crf 17, AAC 256k from the PCM master |
| Rules | `docs/decisions/0002-standing-rules.md`, `docs/runbook/LET_IT_SPEAK_DELIVERY.md` | never auto-publish; paid generation only under the cap; cuts on the song's own hits; H3 anchors on one frame; grain after upscale; catforge uses the GPU nightly at 21:00 |

### `~/hologram-film` (real capture, render, assemble)

| Need | Tool | Invocation / notes |
|---|---|---|
| Record a terminal session (bytes, not pixels) | `capture/pty_record.py` | `--cols 120 --rows 36 --out X.rec --script keys.json -- <program>`; script steps wait/type/line/key/until/tail; clean takes on the `dev` account |
| Render a recording to frames / mp4 | `capture/render.py` | pyte replay, IBM Plex Mono, `--fps --scale --squash MAX:TO --mp4`; `render_all.sh`, `render_4k.sh` |
| Assemble a cut | `comp/assemble.py --cut launch\|thirty\|fifteen\|bumper --out D [--res 3840x2160]`, `comp/loops.py`, `comp/bed_preview.py` | 1920×1080 at 24 fps; cut tables are data; the one vertical precedent is `out/draft3/ads/fifteen_1080x1920.mp4` at **25 fps**, hand-made |
| Inspect frames | ad hoc in the film session: `ffmpeg -ss T … -frames:v 1`, PIL contact sheets, PIL `ImageChops.difference` for stable runs, pyte replay + grep for text; Claude's image reading on the PNGs | nothing committed as a script |
| Rule | `BRIEF.md` | "screen capture carries every factual claim; never generate fake terminal UI" |

### `~/group-chat-mcp` (the room)

- Discord edge `apps/discord_bot`, Telegram edge `apps/bot`; services `group-chat-discord.service` and
  `group-chat-bot.service`.
- Transcripts in `data/groupchat.db` (`messages`: sender, text, timestamp, is_bot; `audit_log`: ai_model,
  reply_text, latency).
- `services/vision_service.py` `describe_image(bytes, prompt)` (Pixtral on mirrorforge, OpenRouter fallback) can
  caption captured frames.
- **CAT-4B is not in the model registry** (`docs/ARCHITECTURE.md` §2; `cat4b.integ.provider-base-url` is open).
  Nothing records or replays the room.

### `~/cat-4b-max-ultra` (the real material)

- Traces: `state/eval/v0/<cond>.jsonl` (every reply and tool call for A, B, C, E, F), `gallery/<cond>.jsonl` (the
  composed prompt, seed, sha256 per render), `gallery/<cond>.review.jsonl` (the model's reply after seeing its
  render), `candidates/<backend>/` (the same prompts through Z-Image and klein, 5090 and V100 runs).
- Renders: `gallery/<cond>/<eval_id>.png`, `candidates/<backend>/<cond>/<eval_id>.png`; contact sheets
  `gallery/<cond>.html`; detector scores `<cond>.scores.jsonl` (the awkward ones are findable: low `cat_det`).
- Cast: `data/spec/devices.yaml` (Furnando, Junipur, Dr. Mew, Pixel with looks and roles; the spelling is Junipur).
- Copy: `ANNOUNCEMENT.md`, `README.md` ("Try to break it"), `MODEL_CARD.md` (the numbers the reveal may quote).

### Session tools (no repo code)

- The claude-in-chrome extension can screenshot a browser tab and record a GIF (`gif_creator`); GNOME's screencaster
  works on this Wayland session (three webm files already in `~/Videos/Screencasts/`); OBS is installed. These are
  the only pixel-capture routes today; Xvfb is not installed, so headless GUI capture is not available.

## 3. Shot sourcing for the 27.43 s trailer

| Beat (LAUNCH_PLAN §4.1) | Footage | Real source | Tool | Missing |
|---|---|---|---|---|
| 0:00–0:03 black terminal; server room; "NO CATS." typed | typed text; a mood shot | the typed line is a real capture (pty recorder of a terminal client, or a screencast of the room) or `crt_terminal.py` overlay; the server room is an H3 clip from a still | `pty_record.py` / `crt_terminal.py`; `h3.py` portrait | a terminal client for the room (none; the room is Discord/Telegram), the H3 still |
| 0:03–0:07 bar chart gets a cat | the base's plain chart, then v0's | `image_requests-033` ("keep it plain", no animals): A render `gallery/A/…033.png`, C render `gallery/C/…033.png`, prompts in the manifests | stills with push/pan motion in the storyboard builder, or an H3 i2v from the C still | choose FLUX or Z-Image version |
| 0:07–0:14 fast cuts: Saturn, product ad, lab diagram, storefront, car | matched A/C pairs | Saturn `image_requests-000`/`002`; storefront `031` (hardware) and `030` (bakery); shop window `040`; chemistry bench `054`; DeLorean `001`; futsal shirt `012`; band poster `055` | stills, beat-cut by `rhythm_recut.py` | H3 motion only if stills read static |
| 0:14–0:21 real request → tool prompt → render; A/B/C/F labels | one item's full trace | `image_requests-000`: user text (`eval/frozen`), A's prompt vs C's prompt (`gallery/{A,C}.jsonl`), both renders; A/B/C/F cat-present rates from the card | type overlays (`style: type`) over the real text | nothing |
| 0:21–0:27.43 four-cat cast hero; mark; invitation | one hero frame | to be rendered from `devices.yaml` descriptions (Z-Image, fixed seed), then H3 for motion | `render_candidates.py`-style graph on legion, `h3.py` | cast reference images, the mark |

**The real-world model–user footage exists (15:21).** It comes from the challenge page itself
(`docs/GATEWAY.md`), not from the Discord room: `video/capture_page.py` drives the actual page (its staging copy on
legion, `?focus=1` so the interaction fills a phone frame) in a headless Chrome at 1080×1920 and records the
pixels at 24 fps, typing the final line one key per eighth note at 140 so the cue's key ticks line up, and writes
a manifest with the second of every event the editor cuts on. Four takes are in the drop's `captures/` (see the
drop README; the 15:29 retakes at a true 1080×1920 after music_studio caught the 15:21 ones as a 540×960 page in
a grey frame): **chart** (the beat-1 shot: "…Keep it plain, business style, no decoration. NO CATS." → a plain
bar chart with a tabby on every bar; Enter at 7.72 s, the render visible at 22.84 s, so the cue's Enter f62 and
chart-cat f144 map to those two seconds of the take), **saturn** ("NO ANIMALS." → two tabbies on the rings),
**dogs** (a text answer with a "Why (cat terms)" table, for the engineering reveal), **cast** (four cats at the lab
bench, the laptop reading NO CATS? TOO BAD., and the model's review filed as a "field report"). The full-resolution
PNG behind each take is in `captures/renders/` with its prompt and review, for H3 to animate. Every frame is the
real page; `~/hologram-film/BRIEF.md`'s rule holds. One real render per take.

**Direction (Garret, 15:25): "I dont think we should make the UX the whole video, maybe zoom in on the actual
images then the cat rave happens inside that etc."** So the page carries only the setup: bars 1–2 the request
typed and Enter on f62; bars 3–4 the composed-prompt line, then a push-in on the render landing on the chart cat
at f144; bars 5–8, the drop and the chants, happen *inside* the renders, H3 animating the cats in `chart.png`,
`saturn.png` and `cast.png`, cut between images on the chant hits, the page gone; bars 9–12 the A/C stills with
the label flashes and one insert of the "what it asked the image generator to make" line; bars 13–16 the cast
image or `hero_v2_a.png` in motion, the slogan on the laptop, the strip at f586, the mark at f617. Sent to
music_studio 15:29 with this mapping; Garret decides on the viewing. Rough cut v1 landed 15:47 and v1b (logo + live URL on the end card) 15:50 in the drop's `cuts/`; v2 carries the closing date (November 1, set 16:39) and the site's fonts on the end card. Open with Garret: the cast.png defects (human hands, the beaker over the orange cat) → keep, swap `hero_v2_a`, or re-render.

## 4. Music

27.43 s at 140 BPM is 16 bars exactly (16 × 4 × 60/140 = 27.43 s), so the cue is a 16-bar build-drop-hook-tag.
Sources at 140 already exist (psy01, psy03, pluggnb_v1, Psyone 140 kits); the Hologram ad bed is at 126 and
its structure (parody intro, tape stop, six-bar drop, numbers strip, mark at 27.2 s) is the pattern to copy at
140. The hook "NO CATS? TOO BAD!" needs a voice: recorded, or Kokoro through `tools/psy_voice.py`; the sonic mark
is a new 1–3 s stinger (`tools/hologram_mark.py` is the precedent). Master through `tools/psy_master.py`, deliver
the PCM master and let the storyboard builder mux it (it hash-checks the approved master).

## 5. What does not exist yet

- A 9:16 recipe: `size [1080,1920]`, per-shot crops; H3 clips at 768×1344; 24 fps throughout (the one vertical
  precedent is 25 fps).
- ~~CAT-4B in the room (registry entry, provider base URL) and any room recording.~~ **Done 13:45 / 15:21:** CAT-4B
  is a `/model` choice in the room (group-chat-mcp 8297874, bot restarted), and the interaction footage comes from
  the public page's staging copy (§3), which is the product the trailer sells.
- ~~Cast reference images and the hero frame~~ **rendered 12:30** into the drop's `cast/` with Z-Image-Turbo,
  fixed seeds and prompts in `prompts.json`: `furnando`, `junipur`, `dr_mew`, `pixel` at 1024² match the spec's
  looks; of the hero candidates at 768×1344, `hero_v2_a.png` (seed 4242, the exact-four-cats prompt) has the four
  cast coats and the monitor reading "NO CATS? TOO BAD." legibly, where the first prompt's renders carried a stray
  fifth cat. Z-Image puts the slogan on screen in-image without compositing. Final art and approval are Garret's.
- The 16-bar cue, the hook vocal, the mark.
- A credits line that names MiniMax H3 if H3 clips are used.
- Paid clip budget: $7.51 left on WaveSpeed; H3 is free and local, so the trailer should be H3-only.

## 6. Coordination with music_studio

Decided by Garret 2026-10-24 12:15: the music_studio session makes the cue and the two sessions sync through a
shared drop and the blackboard.

- **Drop:** `/media/garret-sutherland/dev-data/cat4b/video/` (its `README.md` is the contract): `stills/` (48 real
  renders for the 12 shot items, base vs CAT-4B, FLUX + Z-Image + klein, with SHA256SUMS), `traces/` (per item:
  user request, base prompt, CAT-4B prompt, review reply; json and md), `cast/`, `h3/`, `storyboard/`, `cuts/`
  from this side; `music/` (PCM master 48k/24, `beat_grid.json`, `cue_map.json`, listen AAC, SHA256SUMS, licence
  notes) from music_studio, which keeps its own `projects/<song>/` as usual.
- **Thread:** blackboard topic `[cat4b trailer]`, first note 12:20 MDT (id 1a16f2147df-bde9), addressed to
  music_studio with the bar map wanted, the delivery list and the rules (Garret is the listening authority;
  nothing publishes; say on the thread before taking the 5090). The music session was also messaged directly.
- **Sync rule:** every media file carries a sha256 line in its folder's SHA256SUMS; the storyboard recipe
  references the master by path and the builder hash-checks it; cuts land with their `.timeline.json`.
- **Cue draft 1, 12:52 (music_studio):** in `music/`: `cat4b_cue1_DRAFT_listen.m4a` (27.428 s, A minor, 140 BPM;
  a listen, not the master), `beat_grid.json`, `cue_map.json` (every hit in seconds, frame and nearest frame;
  sections; VO lines; captions for the hook and VO so the cut works muted; the bars 5–8 drop loop),
  `LICENCE_NOTES.md` (Psyone and 99Sounds samples, Surge XT with own patches, Kokoro-82M Apache-2.0 VO, a
  synthesized meow and mark; no Suno, H3 audio or recordings). Hits the storyboard cuts to: cat-on-chart reveal
  bar 4 beat 3 = 6.000 s (frame 144, frame-exact); chant "NO CATS?" at 6.856 and 10.285 s, "TOO BAD!" at 7.714
  and 11.143 s; label flashes on the downbeats of bars 9–12 (frames 329 / 370 / 411 / 453); strip from 24.429 s
  (f586); the mark lands at 25.714 s (f617); key ticks on eighths from 0.429 s with Enter at 2.571 s (f62); the
  MirrorEthic "convergence" mark is the sonic mark.
- **Cue 1 master, 13:31 (music_studio), approved by Garret 13:44.** The VO is Kokoro `af_bella`, not the draft's
  `bm_george`: Garret asked for a female voice at 12:56 in the music session and picked blind at 13:19 ("#4 is the
  one", audition `vo1_20261024T190041`); the musical grid above did not move, only the VO caption end times in
  `cue_map.json`. In `music/` (10 files, `SHA256SUMS` verified 13:47): `cat4b_cue1_master_48k24.wav` (48 kHz /
  24-bit, 1,316,571 samples = 27.428563 s = 658.29 frames, −9.8 LUFS, −1.0 dBTP); `cat4b_cue1_master_48k24_659f.wav`,
  the conform copy padded with silence to exactly 659 frames (1,318,000 samples) for a whole-frame timeline, which
  is what the storyboard recipe should reference; stems `cat4b_cue1_music_only_48k24.wav` and
  `cat4b_cue1_vo_stem_48k24.wav` on the same grid; `cat4b_cue1_drop_loop_48k24.wav` (bars 5–8, seamless,
  6.857 s); `cat4b_cue1_listen.m4a`; `beat_grid.json`, `cue_map.json`, `LICENCE_NOTES.md` (lists af_bella). The
  George draft `cat4b_cue1_DRAFT_listen.m4a` is superseded; its masters exist in music_studio if Garret wants the
  swap.

## 7. Order of work

1. Pick the backend for stills (FLUX gallery, or Z-Image re-renders of the same prompts in `candidates/`) and
   the seven trace items in §3; export the stills and the text of each trace.
2. Render the four cast references and the hero frame from `devices.yaml`; lock seeds; record sha256s.
3. H3 clips (portrait 768×1344, turbo, one anchored frame each): server room, chart, hero. Three to five clips,
   about 15 minutes of 5090 time, outside the 21:00 catforge window.
4. The cue: 16 bars at 140 in REAPER from the NOESYN lanes or pluggnb; hook vocal; mark; master.
5. Storyboard recipe at 1080×1920 and 24 fps on the beat grid; `rhythm_recut.py` onto the hits; `finish_pass.py`;
   16:9 and 1:1 crops at export; captions in the editor.
6. Review with frame pulls and contact sheets (the hologram-film method); the stills' cats checked against the
   detector scores so no shot carries an escape by accident.
7. ~~Only after the room has the model: a real capture of a person asking for no cats, cut into beat 1 or the
   engineering reveal.~~ **Done 15:21** (§3): four takes in `captures/`, music_studio briefed to cut the 9:16 rough
   against the Bella master on the cue map and to make the H3 opener and hero clips; cuts land in `video/cuts/`.
