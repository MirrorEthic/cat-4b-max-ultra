# Ship day: Sunday 2026-10-25

The challenge opens itself at **12:00 MDT** (`GATEWAY_OPENS_AT` on mirrorforge) and closes **2026-11-01 23:59 MST**.
Everything below is what happens around that, in order, with who does it. Times are MDT.

## Before noon

| when | who | what | how |
|---|---|---|---|
| ~11:00 | Claude | opening go/no-go with one real render through the public URL | `serve/opening_check.sh --smoke` → every line `[ok]`, `GO` |
| ~11:00 | Claude | flip the card to released and rebuild the Hugging Face bundle | `MODEL_CARD.md`: Status "v0, released 2026-10-25", Release date "2026-10-25"; `README.md` banner; commit; `release/build_hf.py --released 2026-10-25`; `release/build_hf.py --check`; `release/upload_hf.py upload` (only the changed files go) |
| ~11:30 | Claude, on Garret's word | make the weights public | `release/upload_hf.py publish` → https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA |
| ~11:30 | Garret or Claude on his word | make the code public | the private repo exists and holds the full history (pushed 18:08): `gh repo edit MirrorEthic/cat-4b-max-ultra --visibility public` (gh 2.45 has no consequences flag). Done 11:35 with the weights |
| done 18:03 Sat | Garret | the site with the CAT-4B sections and the corrected numbers is deployed | nothing left here unless copy changes (`cd ~/mirrorethic-site && ./deploy.sh`) |
| 12:00 | nobody | the gateway opens | the window flips; the page's "opens …" line becomes the queue status |
| 12:00+ | Garret | the launch post with the URL and the trailer | copy: `docs/LAUNCH_PLAN.md` §14.1; trailer: **`video/cuts/cat4b_trailer_v4_hd_graded.mp4`** (v4 HD with the finishing grade and grain; 1080×1920, 823 frames, 99.8 MB; cast.png stays, Garret 19:09). Crops for 16:9 and 1:1 are cut from this file, never re-graded |
| 12:05+ | Claude in the browser on Garret's word (X from @MirrorEthic_, LinkedIn from Garret's profile); Garret for Reddit, Show HN and Patreon (the extension blocks reddit.com and patreon.com; HN is not signed in) | the per-platform posts | `docs/LAUNCH_POSTS.md` (drafts, facts table, posting order); assets `video/social/` (trailer 9:16, 1:1, 16:9; stills; real outputs; titled promo art) |

## During the week

Everything after today, with owners and status, is tracked in `docs/ROADMAP.md` (added ship-day morning).

- Claude: a morning and evening look at `/api/status` and the admin `counts` (replies, renders, retries fired, first-draft
  cat rate, screen hits by category), the pi5 contracts, unit restarts; a note on the board each time.
- Garret: the editorial beats (§5.3), opted-in gallery curation (`/admin unshare` pulls an item), the support path.
- Anything broken: `serve/mirrorforge/push_gateway.sh rollback` restores the previous gateway tree; `push_live_service.sh
  rollback` restores the FLUX-era units if the model side itself fails; admissions can be closed with one admin call
  while the page stays up and honest.

## Closing (2026-11-01)

- 23:59 MST: the window closes itself; `/api/chat` answers with the closing copy; the page and gallery stay up.
- After: the retention sweep deletes private results 7 days later (`GATEWAY_RETENTION_DAYS`); the opted-in items
  stay; the after-action (§13.2) and the human-evaluation update follow; v0.1 trains on the week's escape pairs.

## What is where

- Weights and methodology: `huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA` (private until publish). Bundle on dev-data at
  `release/hf/cat4b-v0/`, 504 files, 5.78 GB, `SHA256SUMS`; adapter `f802ac35…`, merged Q8 `2a4c4047…`, projector
  `6b705489…` are the served files.
- Code: `github.com/MirrorEthic/cat-4b-max-ultra` (private until the ship; full 127-commit history so the research graph's commit receipts resolve; secrets scan of the whole history clean; local tooling state untracked and ignored).
- Service: `docs/LIVE_SERVICE.md`, `docs/GATEWAY.md`; staging on legion `serve/staging.sh`.
- Video: `/media/garret-sutherland/dev-data/cat4b/video/` (`cuts/`, `h3/`, `music/`, `captures/`, `brand/`).
