"""API spend ledger with a hard cap (docs/DATA_AND_TRAINING.md §9, decided cap $400).

Every batch reserves its WORST-CASE cost (estimated input + full max_tokens output) before submission; the reservation
is replaced by actual usage when results are collected. A submit that would push spent + reserved past the cap is
refused, so the cap holds even with batches in flight.
"""
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from . import STATE_DIR

CAP_USD = 400.0

# $ per million tokens, Message Batches (50% of list). Cache writes: 5m = 1.25x input, 1h = 2x input.
PRICES = {
    "claude-opus-5": {"in": 2.50, "out": 12.50, "cache_read": 0.25, "cache_5m": 3.125, "cache_1h": 5.00},
    # regular (non-batch) Messages API: list price, used only for small pilots where latency matters
    "claude-opus-5:sync": {"in": 5.00, "out": 25.00, "cache_read": 0.50, "cache_5m": 6.25, "cache_1h": 10.00},
}


def usage_cost(model: str, u: dict) -> float:
    p = PRICES[model]
    cc = u.get("cache_creation") or {}
    w5 = cc.get("ephemeral_5m_input_tokens", 0) or 0
    w1 = cc.get("ephemeral_1h_input_tokens", 0) or 0
    if not (w5 or w1):  # older usage shape: undifferentiated creation tokens, price as 1h (conservative)
        w1 = u.get("cache_creation_input_tokens", 0) or 0
    return (u.get("input_tokens", 0) * p["in"] + u.get("output_tokens", 0) * p["out"]
            + (u.get("cache_read_input_tokens", 0) or 0) * p["cache_read"]
            + w5 * p["cache_5m"] + w1 * p["cache_1h"]) / 1e6


class CapExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path | None = None, cap: float = CAP_USD):
        self.path = Path(path or STATE_DIR / "ledger.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap
        with closing(self._db()) as db, db:
            db.execute("""create table if not exists reservations(
                batch_id text primary key, name text, model text, n int, worst_usd real, ts real, open int)""")
            db.execute("""create table if not exists spend(
                custom_id text primary key, batch_id text, model text, usd real,
                input_tokens int, output_tokens int, cache_read int, cache_write int, ts real)""")

    def _db(self):
        return sqlite3.connect(self.path, timeout=30)

    def spent(self) -> float:
        with closing(self._db()) as db:
            return db.execute("select coalesce(sum(usd),0) from spend").fetchone()[0]

    def reserved(self) -> float:
        with closing(self._db()) as db:
            return db.execute("select coalesce(sum(worst_usd),0) from reservations where open=1").fetchone()[0]

    def headroom(self) -> float:
        return self.cap - self.spent() - self.reserved()

    def reserve(self, batch_id: str, name: str, model: str, n: int, worst_usd: float) -> None:
        if worst_usd > self.headroom():
            raise CapExceeded(f"{name}: worst case ${worst_usd:.2f} > headroom ${self.headroom():.2f} "
                              f"(cap ${self.cap:.0f}, spent ${self.spent():.2f}, reserved ${self.reserved():.2f})")
        with closing(self._db()) as db, db:
            db.execute("insert into reservations values (?,?,?,?,?,?,1)",
                       (batch_id, name, model, n, worst_usd, time.time()))

    def check(self, name: str, worst_usd: float) -> None:
        """Pre-flight check without reserving (the batch id doesn't exist yet)."""
        if worst_usd > self.headroom():
            raise CapExceeded(f"{name}: worst case ${worst_usd:.2f} > headroom ${self.headroom():.2f}")

    def record(self, custom_id: str, batch_id: str, model: str, usage: dict) -> float:
        usd = usage_cost(model, usage)
        cw = (usage.get("cache_creation_input_tokens", 0) or 0)
        with closing(self._db()) as db, db:
            db.execute("insert or replace into spend values (?,?,?,?,?,?,?,?,?)",
                       (f"{batch_id}:{custom_id}", batch_id, model, usd, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                        usage.get("cache_read_input_tokens", 0) or 0, cw, time.time()))
        return usd

    def close_reservation(self, batch_id: str) -> None:
        with closing(self._db()) as db, db:
            db.execute("update reservations set open=0 where batch_id=?", (batch_id,))

    def summary(self) -> dict:
        with closing(self._db()) as db:
            rows = db.execute("""select r.name, r.n, r.worst_usd, r.open, coalesce(sum(s.usd),0)
                                 from reservations r left join spend s on s.batch_id=r.batch_id
                                 group by r.batch_id order by r.ts""").fetchall()
        return {"cap": self.cap, "spent": round(self.spent(), 4), "reserved": round(self.reserved(), 4),
                "batches": [dict(name=a, n=b, worst=round(c, 2), open=bool(d), actual=round(e, 4))
                            for a, b, c, d, e in rows]}
