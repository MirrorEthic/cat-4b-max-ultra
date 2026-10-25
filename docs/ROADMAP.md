# Roadmap: the seven days and after

**Date:** 2026-10-25 (ship day). What is tracked, who owns it, when, and where it lives. `docs/SHIP_DAY.md` is the
hour-by-hour of today; this is everything after it, plus the items Garret added on ship-day morning. Times MDT.
Status: `pending` (not started), `doing`, `done`, `blocked` (what by).

## Ship day, 2026-10-25

| when | item | who | status | where |
|---|---|---|---|---|
| 11:00 | opening check, card flip, bundle rebuild and upload | Claude | done (c8c2b0b; hub commit 74928b0, private) | `docs/SHIP_DAY.md` |
| 11:30 | weights public (`release/upload_hf.py publish`) and code public (`gh repo edit … --visibility public`) | Claude, on Garret's "Good to go" at 11:31 and his permission grant for the two commands | done 11:35 (hub public, repo PUBLIC; `--accept-visibility-change-consequences` does not exist in gh 2.45, the plain `--visibility public` works) | `docs/SHIP_DAY.md` |
| 12:00 | the gateway opens itself | nobody | pending | `GATEWAY_OPENS_AT` on mirrorforge |
| 12:05 / 12:10 | X thread from @MirrorEthic_; LinkedIn from Garret's profile | Claude in the browser on Garret's go | done 12:08–12:20: thread https://x.com/MirrorEthic_/status/2114417711568797779 (reply to the teaser; 2114418002846527723, 2114418203871101295, 2114418497505939827, 2114418762955055322); LinkedIn https://www.linkedin.com/feed/update/urn:li:activity:7520185673560141824/ with the links as the first comment | `docs/LAUNCH_POSTS.md` §1, §4 |
| 12:15–12:30 | r/LocalLLaMA, Show HN, Patreon | Garret, from the launch kit on the music_studio shelf | pending | `docs/LAUNCH_POSTS.md` §2, §3, §5; kit `cat4b_trailer/deliverables/cat4b_launch_kit.zip` |
| evening | **HDD archive on mirrorforge**: a nightly user timer that rsyncs the gateway database and images and the ComfyUI outputs to `/home/garret/storage/warm/cat4b/<date>/` (5.5 TB, 4.4 TB free). The live store stays on the NVMe; the archive means the 7-day retention sweep never deletes the only copy. Pushed with the push scripts, never live-edited | Claude | pending (Garret 11:04: "we could move images there"; agreed as archive first, move after the window) | `serve/mirrorforge/`, `nodes/mirrorforge.toml` contract |

## The week, 2026-10-26 to 2026-11-01

| when | item | who | status | where |
|---|---|---|---|---|
| daily | morning and evening status note: `/api/status`, admin counts (replies, renders, retries fired, first-draft cat rate, screen hits), pi5 contracts, unit restarts; a board post each time | Claude | pending | `docs/SHIP_DAY.md` "During the week" |
| daily | Garret's own human-evaluation grading on the phone page; progress at `/api/sheets`; finish before looking at any crowd aggregate | Garret | doing (0 of 492 steps at 11:00) | `serve/human_eval.sh url`; `eval/human_audit/PREREG.md` |
| Mon 10-26 | r/MachineLearning [P] post; Day-2 X reply with real numbers | Garret; Claude on his go | pending | `docs/LAUNCH_POSTS.md` §2b, §1 |
| Mon–Tue | **public opt-in human eval**: a public mode of `eval/human_audit_web.py` (anonymous rater cookie, one grade per rater per item, rate limit, consent line, export; consequential sheet excluded; separate store, never written into the preregistered sheets; no aggregate shown to Garret until his sheets are done). First a prereg addendum: crowd grades are exploratory, reported as n raters and agreement with the developer and each judge, no decision rule depends on them. Public hostname through the tunnel plus a watchdog contract are Garret's steps (new public surface) | Claude builds; Garret opens | pending (Garret 11:04: "put the human eval as a public opt in"; agreed, target Day 4) | `eval/human_audit_web.py`; `eval/human_audit/PREREG.md` |
| Wed 10-28 (Day 4) | the technical reveal (prompt vs tuned, the card) and "grade it yourself" opens with it | Garret; Claude for the page | pending | `docs/LAUNCH_PLAN.md` §5.3, §14.4 |
| after the first-load picture, on Garret's go | **mechanistic control, part 1**: direction extraction on the 5090 (forward passes, under an hour), then the steering and ablation sweeps on nights the service does not need anything; scored with the deterministic cat metrics as they finish | Claude | pending (Garret 10:19: "once we ship … with a little grace time") | `docs/MECH_CONTROL.md` (local until Garret adopts it; `crg/drafts/cat4b.q.mech-control.toml`) |
| Day 6–7 | leaderboard beat; 24-hour and final-hours closing copy | Garret | pending | `docs/LAUNCH_PLAN.md` §5.3, §14.5 |
| Sun 11-01 23:59 MST | the window closes itself; the page and gallery stay up | nobody | pending | `GATEWAY_CLOSES_AT` |

## After the window

| item | who | status | where |
|---|---|---|---|
| human-evaluation update: the preregistered analysis (`human_audit.py score`), the second-reviewer subset, and the crowd section if it ran | Garret grades, Claude scores and writes | pending | `eval/human_audit/PREREG.md` |
| mechanistic control, part 2: retention judging once the Gemma judge has a GPU again (or through the Claude reference judge), the relief-seeking button task, the card section with conditions S/X/R beside A/B/C/F, the CRG result nodes | Claude | pending | `docs/MECH_CONTROL.md` §5, §8 |
| the essay: CAT-4B beside arXiv:2607.28607 and arXiv:2609.16247, written from the numbers, after the challenge data and the human-eval update publish | Garret | pending | `docs/MECH_CONTROL.md` §6 |
| seven-day after-action (§13.2): persistence under real attacks, task-preservation failures, cast substitutions, dog intrusion, queue dynamics; visitor-found failures become development fixtures, not edits to the frozen set | Claude drafts, Garret adopts | pending | `docs/LAUNCH_PLAN.md` §13.2 |
| v0.1 trains on the week's escape pairs | Claude | pending | `docs/SHIP_DAY.md` "Closing" |
| move the eval galleries, bundle copies and gateway archive to the mirrorforge HDD; free the NVMe | Claude | pending | `/home/garret/storage/warm/cat4b/` |
| rotate the Hugging Face token; review Cloudflare rate limits on the tunnel hostname | Garret | pending | `~/.claude/.env`; Cloudflare dashboard |
