# 5090 scheduling: receipts that live outside the repo

Receipts for fork A's scheduling result nodes. Logs are under `/media/garret-sutherland/dev-data/cat4b/logs/`.
Session quotes are from
`~/.claude/projects/-home-garret-sutherland-cat-4b-max-ultra/bde94a93-cedd-46fd-ac1b-fffc7d6cb66c.jsonl` (UTC
timestamps; MDT = UTC−6).

## resume died with the shell

Session, 2026-09-24T23:09Z (17:09 MDT): "What it produced: only 672 candidates (168 specs) … The resume I ran yesterday
at 17:53 died as soon as my command finished. `gpu_resume.sh` launched its jobs in a way that tied them to the calling
shell. So it sat idle overnight … It's fixed in `022664c`: resumed jobs now fully detach (`setsid`)."

Session, 2026-09-25T02:56Z, after the next resume: "the generator and teacher are still alive after the launching
command exited, which is exactly what failed yesterday."

## cron entries lost

`serve/crontab.before-mote-pause-2026-09-25.txt` (the crontab as installed) lines 26-28 hold the window:

```
0 21 * * * /home/garret-sutherland/cat-4b-max-ultra/serve/nightly.sh start
0 9 * * * /home/garret-sutherland/cat-4b-max-ultra/serve/nightly.sh stop
```

`logs/nightly.log` has no `nightly start` on 2026-09-25; its first lines are a manual stop at 10:55:10 and the
systemd stop at 2026-09-26 09:00:11. Session, 2026-09-26T05:12Z: "the crontab lines were in place, but cron never ran
them. Something else rewrote your crontab twice today (16:23 and 22:23) … the schedule now runs as systemd user timers
instead, since nothing else edits those."

From 2026-09-26 to 2026-10-14, `logs/nightly.log` has 18 `nightly start` lines, all at 21:00, and 18 `nightly stop`
lines at 09:00 (plus one at 09:16 on 2026-10-08 and a manual one at 21:25 that evening).

## nightly starts refused

`logs/nightly.log`, starts from 2026-09-26 through 2026-10-11: the teacher came up on 2026-09-26, 09-27, 09-29,
10-08, 10-09, 10-10 and 10-11; it failed on 2026-09-28 and on every night from 09-30 to 10-07:

```
[2026-09-30 21:00:08] nightly start
teacher failed to start; see /media/garret-sutherland/dev-data/cat4b/logs/llama_teacher.log
[2026-10-01 09:00:00] nightly stop
5090: 19150 MiB in use
```

The 09:00 stops after failed nights record 19,150, 11,312, 2,400 (four times), 19,531 and 17,407 MiB in use, i.e.
another job held the card. `f04eb2a` (2026-10-12) put the launcher's refusal reason into the log; before it the reason
went to /dev/null.

## mirrorforge went to the judge

Session, 2026-10-13T00:17Z (2026-10-12 18:17 MDT): "Mirrorforge is free now. The 500M run is finished and only
ComfyUI/FLUX is resident. The V100 teacher build is ready". Garret, 2026-10-13T00:18Z: "i say we start getting to work
on getting this moving towards deployment. we can evict comfy on mirrorforge and use that". Reply at 00:39Z:
"Mirrorforge, now the judge box … The Gemma-4-31B judge is serving across both cards". No teacher has run on
mirrorforge: `gen-local` takes no URL and `generate.run` calls `http://127.0.0.1:8290`, the legion 5090 teacher
(`serve/llama_local.sh teacher`).
