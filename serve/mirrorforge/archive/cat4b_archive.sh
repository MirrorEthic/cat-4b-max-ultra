#!/usr/bin/env bash
# Nightly archive of the CAT-4B public service's state on mirrorforge to the 5.5 TB HDD (docs/ROADMAP.md, ship-day
# evening). Runs ON mirrorforge from cat4b-archive.timer; pushed there by serve/mirrorforge/push_archive.sh and never
# edited on the node. Read-only on the live store: the gateway database is copied through SQLite's online backup (a
# consistent snapshot of a WAL database; a file copy would not be), the results images and the ComfyUI outputs are
# rsynced with hard links against the previous day's copy so unchanged files take no new space. The live store on the
# NVMe keeps its 7-day retention sweep; the archive means the sweep never deletes the only copy. The gateway's secrets
# (state/secret, state/admin_token) are not part of what is copied.
set -euo pipefail
SRC_STATE=/home/garret/cat4b/gateway/state
SRC_COMFY=/home/garret/comfyui/output
DEST=/home/garret/storage/warm/cat4b
day=$(date +%F)
out=$DEST/$day
mkdir -p "$out/gateway"
prev=$(ls -d "$DEST"/20??-??-?? 2>/dev/null | grep -v "/$day\$" | sort | tail -n 1 || true)

python3 - "$SRC_STATE/gateway.db" "$out/gateway/gateway.db" <<'EOF'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close(); src.close()
EOF

link=(); [ -n "$prev" ] && [ -d "$prev/gateway/results" ] && link=(--link-dest="$prev/gateway/results/")
rsync -a "${link[@]}" "$SRC_STATE/results/" "$out/gateway/results/"
link=(); [ -n "$prev" ] && [ -d "$prev/comfyui-output" ] && link=(--link-dest="$prev/comfyui-output/")
rsync -a "${link[@]}" "$SRC_COMFY/" "$out/comfyui-output/"

python3 - "$out" "$day" "$prev" <<'EOF'
import json, os, sqlite3, sys, time
out, day, prev = sys.argv[1], sys.argv[2], sys.argv[3]
c = sqlite3.connect(f"file:{out}/gateway/gateway.db?mode=ro", uri=True)
m = {"date": day, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "previous": os.path.basename(prev) if prev else None,
     "integrity": c.execute("pragma integrity_check").fetchone()[0],
     "sessions": c.execute("select count(*) from sessions").fetchone()[0],
     "jobs": c.execute("select count(*) from jobs").fetchone()[0],
     "shared": c.execute("select count(*) from jobs where shared=1").fetchone()[0]}
c.close()
for key, sub in (("results", "gateway/results"), ("comfyui_output", "comfyui-output")):
    files = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(out, sub)) for f in fs]
    st = [os.stat(p) for p in files]
    m[key] = {"files": len(files), "bytes": sum(s.st_size for s in st), "new_bytes": sum(s.st_size for s in st if s.st_nlink == 1)}
json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
if m["integrity"] != "ok":
    sys.exit(f"database copy failed integrity_check: {m['integrity']}")
print(f"archive {day}: db ok, {m['sessions']} sessions, {m['jobs']} jobs, {m['shared']} shared; results {m['results']['files']} files "
      f"({m['results']['new_bytes'] // 2**20} MiB new); comfyui {m['comfyui_output']['files']} files ({m['comfyui_output']['new_bytes'] // 2**20} MiB new)")
EOF
ln -sfn "$day" "$DEST/latest"
