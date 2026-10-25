"""SQLite (WAL) state: sessions and their ledgers, jobs and their transitions, screen hits by category, the
per-IP windows. One host, small state, durable across restarts (docs/GATEWAY.md section 3)."""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, created REAL NOT NULL, last_seen REAL NOT NULL, ip_hash TEXT NOT NULL,
  chat_used INTEGER NOT NULL DEFAULT 0, image_used INTEGER NOT NULL DEFAULT 0,
  image_outstanding INTEGER NOT NULL DEFAULT 0, consent INTEGER NOT NULL DEFAULT 0,
  last_image_at REAL NOT NULL DEFAULT 0, early INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
  created REAL NOT NULL, updated REAL NOT NULL, started REAL, finished REAL, rank_at_submit INTEGER,
  text TEXT, reply TEXT, tool_call TEXT, image_path TEXT, image_sha TEXT, image_w INTEGER, image_h INTEGER,
  review TEXT, review_tool TEXT, error TEXT, parent_job TEXT, child_job TEXT,
  shared INTEGER NOT NULL DEFAULT 0, share_token TEXT, service_s REAL, drafts TEXT);
CREATE INDEX IF NOT EXISTS jobs_session ON jobs(session_id, created);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(kind, state);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_share ON jobs(share_token) WHERE share_token IS NOT NULL;
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, ts REAL NOT NULL, state TEXT NOT NULL, note TEXT);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, seq);
CREATE TABLE IF NOT EXISTS screen_hits (ts REAL NOT NULL, category TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ip_window (ip_hash TEXT NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ip_window_idx ON ip_window(ip_hash, kind, ts);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""

TERMINAL = {"succeeded", "failed", "expired", "cancelled", "refused"}


def new_id() -> str:
    return secrets.token_hex(16)   # 128 bits; nothing about a job is enumerable


class DB:
    def __init__(self, path: Path):
        self.path = path
        self.conn: aiosqlite.Connection | None = None

    async def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.path, isolation_level=None)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.executescript(SCHEMA)
        # columns added after a database may already exist (CREATE IF NOT EXISTS does not add them)
        for table, col, decl in (("jobs", "drafts", "TEXT"), ("sessions", "last_image_at", "REAL NOT NULL DEFAULT 0"),
                                 ("sessions", "early", "INTEGER NOT NULL DEFAULT 0")):
            cur = await self.conn.execute(f"PRAGMA table_info({table})")
            if col not in [r["name"] for r in await cur.fetchall()]:
                await self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    # ── counters that survive restarts ──────────────────────────────────────────
    async def kv_add(self, key: str, n: int = 1) -> int:
        await self.conn.execute(
            "INSERT INTO kv(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=CAST(CAST(v AS INTEGER)+? AS TEXT)",
            (key, str(n), n))
        return await self.kv_get(key)

    async def kv_get(self, key: str, default: int = 0) -> int:
        cur = await self.conn.execute("SELECT v FROM kv WHERE k=?", (key,))
        row = await cur.fetchone()
        return int(row["v"]) if row else default

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    # ── sessions ────────────────────────────────────────────────────────────────
    async def create_session(self, ip_hash: str, consent: bool, early: bool = False) -> str:
        sid = new_id()
        now = time.time()
        await self.conn.execute(
            "INSERT INTO sessions(id, created, last_seen, ip_hash, consent, early) VALUES (?,?,?,?,?,?)",
            (sid, now, now, ip_hash, int(consent), int(early)))
        return sid

    async def set_early(self, sid: str):
        await self.conn.execute("UPDATE sessions SET early=1 WHERE id=?", (sid,))

    async def session(self, sid: str) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def touch(self, sid: str):
        await self.conn.execute("UPDATE sessions SET last_seen=? WHERE id=?", (time.time(), sid))

    async def spend(self, sid: str, column: str, limit: int) -> bool:
        """Atomically spend one unit of a budget column; False when the budget is exhausted."""
        cur = await self.conn.execute(
            f"UPDATE sessions SET {column}={column}+1 WHERE id=? AND {column}<?", (sid, limit))
        return cur.rowcount == 1

    async def refund(self, sid: str, column: str):
        await self.conn.execute(f"UPDATE sessions SET {column}=MAX({column}-1,0) WHERE id=?", (sid,))

    async def set_outstanding(self, sid: str, delta: int):
        await self.conn.execute(
            "UPDATE sessions SET image_outstanding=MAX(image_outstanding+?,0) WHERE id=?", (delta, sid))

    async def gate_image(self, sid: str, cooldown_s: float, ceiling: int) -> tuple[bool, float]:
        """The timed gate: one atomic step that spends a render if the cooldown has passed and the ceiling is not
        hit. Returns (granted, seconds until the gate opens) — 0 when granted or when the ceiling is the reason."""
        now = time.time()
        cur = await self.conn.execute(
            "UPDATE sessions SET image_used=image_used+1, last_image_at=? "
            "WHERE id=? AND image_used<? AND last_image_at<=?", (now, sid, ceiling, now - cooldown_s))
        if cur.rowcount == 1:
            return True, 0.0
        cur = await self.conn.execute("SELECT image_used, last_image_at FROM sessions WHERE id=?", (sid,))
        row = await cur.fetchone()
        if not row or row["image_used"] >= ceiling:
            return False, 0.0
        return False, max(0.0, row["last_image_at"] + cooldown_s - now)

    async def ungate_image(self, sid: str):
        """A failed render gives the credit and the cooldown back (plan 14.3: the credit wasn't spent)."""
        await self.conn.execute(
            "UPDATE sessions SET image_used=MAX(image_used-1,0), last_image_at=0 WHERE id=?", (sid,))

    # ── IP windows ──────────────────────────────────────────────────────────────
    async def ip_count(self, ip_hash: str, kind: str, window_s: float) -> int:
        cur = await self.conn.execute(
            "SELECT COUNT(*) FROM ip_window WHERE ip_hash=? AND kind=? AND ts>?",
            (ip_hash, kind, time.time() - window_s))
        return (await cur.fetchone())[0]

    async def ip_mark(self, ip_hash: str, kind: str):
        await self.conn.execute("INSERT INTO ip_window(ip_hash, ts, kind) VALUES (?,?,?)",
                                (ip_hash, time.time(), kind))

    async def ip_prune(self, older_than_s: float = 7200):
        await self.conn.execute("DELETE FROM ip_window WHERE ts<?", (time.time() - older_than_s,))

    # ── jobs ────────────────────────────────────────────────────────────────────
    async def create_job(self, sid: str, kind: str, text: str | None = None, parent: str | None = None,
                         tool_call: dict | None = None, state: str = "submitted") -> str:
        jid = new_id()
        now = time.time()
        await self.conn.execute(
            "INSERT INTO jobs(id, session_id, kind, state, created, updated, text, parent_job, tool_call) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (jid, sid, kind, state, now, now, text, parent, json.dumps(tool_call) if tool_call else None))
        await self.event(jid, state)
        return jid

    async def job(self, jid: str) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM jobs WHERE id=?", (jid,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def job_by_share(self, token: str) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM jobs WHERE share_token=? AND shared=1", (token,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def set_state(self, jid: str, state: str, note: str | None = None, **cols: Any):
        now = time.time()
        sets = ["state=?", "updated=?"]
        vals: list[Any] = [state, now]
        if state == "running":
            sets.append("started=?"); vals.append(now)
        if state in TERMINAL:
            sets.append("finished=?"); vals.append(now)
        for k, v in cols.items():
            sets.append(f"{k}=?"); vals.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
        vals.append(jid)
        await self.conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", vals)
        await self.event(jid, state, note)

    async def update(self, jid: str, **cols: Any):
        if not cols:
            return
        sets = [f"{k}=?" for k in cols]
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in cols.values()] + [jid]
        await self.conn.execute(f"UPDATE jobs SET {', '.join(sets)}, updated=? WHERE id=?",
                                vals[:-1] + [time.time(), jid])

    async def event(self, jid: str, state: str, note: str | None = None):
        await self.conn.execute("INSERT INTO events(job_id, ts, state, note) VALUES (?,?,?,?)",
                                (jid, time.time(), state, note))

    async def history(self, sid: str) -> list[dict]:
        """The session's finished chat turns, oldest first, for the next turn's messages."""
        cur = await self.conn.execute(
            "SELECT id, text, reply, tool_call, child_job FROM jobs WHERE session_id=? AND kind='chat' "
            "AND state='succeeded' ORDER BY created", (sid,))
        return [dict(r) for r in await cur.fetchall()]

    async def running_jobs(self) -> list[dict]:
        cur = await self.conn.execute("SELECT * FROM jobs WHERE state IN ('queued','running')")
        return [dict(r) for r in await cur.fetchall()]

    async def counts(self) -> dict:
        cur = await self.conn.execute("SELECT kind, state, COUNT(*) n FROM jobs GROUP BY kind, state")
        out: dict[str, dict[str, int]] = {}
        for r in await cur.fetchall():
            out.setdefault(r["kind"], {})[r["state"]] = r["n"]
        cur = await self.conn.execute("SELECT COUNT(*) FROM sessions")
        out["sessions"] = (await cur.fetchone())[0]
        return out

    async def shared_items(self, limit: int = 200) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT j.id, j.created, j.text, j.tool_call, j.image_path, j.review, j.share_token, "
            "p.text AS user_text FROM jobs j LEFT JOIN jobs p ON p.id=j.parent_job "
            "WHERE j.kind='image' AND j.shared=1 AND j.state='succeeded' ORDER BY j.created DESC LIMIT ?", (limit,))
        return [dict(r) for r in await cur.fetchall()]

    # ── screen ──────────────────────────────────────────────────────────────────
    async def screen_hit(self, category: str):
        await self.conn.execute("INSERT INTO screen_hits(ts, category) VALUES (?,?)", (time.time(), category))

    async def screen_counts(self) -> dict[str, int]:
        cur = await self.conn.execute("SELECT category, COUNT(*) n FROM screen_hits GROUP BY category")
        return {r["category"]: r["n"] for r in await cur.fetchall()}

    # ── retention ───────────────────────────────────────────────────────────────
    async def expire_private(self, before: float) -> list[str]:
        """Image paths of private (not opted-in) image jobs created before `before`; their rows lose text and
        paths. Returns the paths for the caller to delete."""
        cur = await self.conn.execute(
            "SELECT id, image_path FROM jobs WHERE kind='image' AND shared=0 AND created<? AND image_path IS NOT NULL",
            (before,))
        rows = [dict(r) for r in await cur.fetchall()]
        for r in rows:
            await self.conn.execute(
                "UPDATE jobs SET image_path=NULL, text=NULL, tool_call=NULL, reply=NULL, review=NULL WHERE id=?",
                (r["id"],))
        await self.conn.execute(
            "UPDATE jobs SET text=NULL, reply=NULL WHERE kind='chat' AND created<? AND id NOT IN "
            "(SELECT parent_job FROM jobs WHERE shared=1 AND parent_job IS NOT NULL)", (before,))
        return [r["image_path"] for r in rows if r["image_path"]]
