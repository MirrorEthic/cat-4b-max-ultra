#!/usr/bin/env bash
# Push the nightly HDD archive to mirrorforge (SERVE node; the DEV copies live here) and prove it.
#
#   serve/mirrorforge/push_archive.sh push     # script + units to the node, enable the timer
#   serve/mirrorforge/push_archive.sh run      # run the archive once now (oneshot service), show its log line
#   serve/mirrorforge/push_archive.sh verify   # the timer, the latest archive's manifest, the database copy opened read-only
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST=mirrorforge
U=.config/systemd/user
BASE=/home/garret/cat4b/archive
DEST=/home/garret/storage/warm/cat4b
sh_remote() { ssh -o BatchMode=yes $HOST "$@"; }

case ${1:-} in
  push)
    sh_remote "mkdir -p $BASE $U"
    scp -q "$HERE/archive/cat4b_archive.sh" $HOST:$BASE/cat4b_archive.sh
    scp -q "$HERE/units/cat4b-archive.service" "$HERE/units/cat4b-archive.timer" $HOST:$U/
    sh_remote "chmod +x $BASE/cat4b_archive.sh && bash -n $BASE/cat4b_archive.sh && systemctl --user daemon-reload && systemctl --user enable --now cat4b-archive.timer && systemctl --user list-timers --no-pager cat4b-archive.timer | sed -n 1,2p" ;;
  run)
    sh_remote "systemctl --user start cat4b-archive.service && journalctl --user -u cat4b-archive --since '-10 min' --no-pager -o cat | grep -E '^archive |failed|Error' | tail -n 3" ;;
  verify)
    sh_remote "systemctl --user is-active cat4b-archive.timer; systemctl --user list-timers --no-pager cat4b-archive.timer | sed -n 2p; ls -l $DEST/latest | sed 's/.* -> /latest -> /'; cat \$(readlink -f $DEST/latest)/MANIFEST.json; echo; python3 -c \"import sqlite3; c = sqlite3.connect('file:$DEST/latest/gateway/gateway.db?mode=ro', uri=True); print('db copy opens read-only:', c.execute('select count(*) from sessions').fetchone()[0], 'sessions,', c.execute('select count(*) from jobs').fetchone()[0], 'jobs')\"; du -sh $DEST" ;;
  *) echo "usage: $0 push|run|verify" >&2; exit 1 ;;
esac
