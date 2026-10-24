# Ship day: Sunday 2026-10-25

The challenge opens itself at **12:00 MDT** (`GATEWAY_OPENS_AT` on mirrorforge) and closes **2026-11-01 23:59 MST**.
Everything below is what happens around that, in order, with who does it. Times are MDT.

## Before noon

| when | who | what | how |
|---|---|---|---|
| ~11:00 | Claude | opening go/no-go with one real render through the public URL | `serve/opening_check.sh --smoke` → every line `[ok]`, `GO` |
| ~11:00 | Claude | flip the card to released and rebuild the Hugging Face bundle | `MODEL_CARD.md`: Status "v0, released 2026-10-25", Release date "2026-10-25"; `README.md` banner; commit; `release/build_hf.py --released 2026-10-25`; `release/build_hf.py --check`; `release/upload_hf.py upload` (only the changed files go) |
| ~11:30 | Claude, on Garret's word | make the weights public | `release/upload_hf.py publish` → https://huggingface.co/mirrorethic/CAT-4B-MAX-ULTRA |
| ~11:30 | Garret | make the code public | `gh repo edit MirrorEthic/cat-4b-max-ultra --visibility public` once the private repo exists (`gh repo create MirrorEthic/cat-4b-max-ultra --private --source . --remote origin --push` from `~/cat-4b-max-ultra`; the session cannot push on its own) |
| ~11:30 | Garret | deploy the site with the CAT-4B sections and the corrected numbers | `cd ~/mirrorethic-site && ./deploy.sh` (commits 282bc76, 3487d41 are staged for it) |
| 12:00 | nobody | the gateway opens | the window flips; the page's "opens …" line becomes the queue status |
| 12:00+ | Garret | the launch post with the URL and the trailer | copy: `docs/LAUNCH_PLAN.md` §14.1; trailer: the cut he picked from `video/cuts/` |

## During the week

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
- Code: `github.com/MirrorEthic/cat-4b-max-ultra` (to be created by Garret; private until the ship).
- Service: `docs/LIVE_SERVICE.md`, `docs/GATEWAY.md`; staging on legion `serve/staging.sh`.
- Video: `/media/garret-sutherland/dev-data/cat4b/video/` (`cuts/`, `h3/`, `music/`, `captures/`, `brand/`).
