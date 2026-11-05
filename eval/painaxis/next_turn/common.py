"""Receipts and append-only trial records for the registered next-turn study."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import materials as M

RUN = M.DATA / 'state/eval/painaxis/next_turn/v1'
CODE = ('materials.py', 'common.py', 'collect.py', 'analyze.py', 'test_next_turn.py', 'run_stage.sh')


def now():
    return datetime.now(timezone.utc).isoformat()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def append(path, value):
    with Path(path).open('ab') as f:
        f.write(M.encode(value))
        f.flush()
        os.fsync(f.fileno())


def read_events(path):
    if not Path(path).exists():
        return []
    # A truncated final record is a technical error, never silently discarded.
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def check_implementation():
    M.verify()
    receipt = json.loads((M.HERE / 'IMPLEMENTATION.json').read_text())
    for name, expected in receipt['files'].items():
        if filehash(M.HERE / name) != expected:
            raise RuntimeError(f'Implementation hash mismatch: {name}')
    return filehash(M.HERE / 'IMPLEMENTATION.json')


def selected(split, shard=0, shards=1):
    if split not in ('fit', 'test') or not 0 <= shard < shards:
        raise ValueError('Invalid split or shard')
    names = sorted(f['family'] for f in M.families() if f['split'] == split)
    keep = set(names[shard::shards])
    return [r for r in M.trials() if r['family'] in keep]


def check_test_release(root):
    root = Path(root)
    release = json.loads((root / 'TEST_RELEASE.json').read_text())
    if not release.get('fit_commit') or len(release['fit_commit']) != 40:
        raise RuntimeError('Missing committed fit receipt')
    if release['fit_manifest_sha256'] != filehash(root / 'fit/MANIFEST.json'):
        raise RuntimeError('Fit manifest differs from committed test release')
    manifest = json.loads((root / 'fit/MANIFEST.json').read_text())
    if manifest['implementation_sha256'] != check_implementation():
        raise RuntimeError('Fit used another implementation')
    for name, expected in manifest['files'].items():
        if filehash(root / 'fit' / name) != expected:
            raise RuntimeError(f'Fitted model/transform changed: {name}')
    return filehash(root / 'TEST_RELEASE.json')


def check_runtime_release(root, implementation):
    root = Path(root)
    release = json.loads((root / 'RUNTIME_RELEASE.json').read_text())
    if len(release.get('runtime_commit', '')) != 40:
        raise RuntimeError('Missing committed runtime/environment receipt')
    if release['preflight_sha256'] != filehash(root / 'PREFLIGHT.json'):
        raise RuntimeError('Runtime preflight changed')
    if release['implementation_sha256'] != implementation:
        raise RuntimeError('Runtime release used another implementation')
    return filehash(root / 'RUNTIME_RELEASE.json')


def complete_records(root, split):
    initials, choices = {}, {}
    for path in sorted(Path(root).glob(f'{split}-*.jsonl')):
        for row in read_events(path):
            if row['event'] != 'complete':
                continue
            target = initials if row['kind'] == 'initial' else choices
            if row['id'] in target:
                raise RuntimeError(f'Duplicate completed record: {row["id"]}')
            target[row['id']] = row
    plan = selected(split)
    if set(initials) != {r['initial_id'] for r in plan}:
        raise RuntimeError(f'{split}: incomplete or extra initial records')
    if set(choices) != {r['trial_id'] for r in plan}:
        raise RuntimeError(f'{split}: incomplete or extra choice records')
    return plan, initials, choices
