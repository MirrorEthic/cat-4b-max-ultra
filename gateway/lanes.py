"""Two independent lanes (docs/GATEWAY.md section 3): a bounded FIFO with a worker pool, an exponential moving
average of service time for an honest ETA, a pause switch for the operator, and a circuit breaker on consecutive
failures. Lanes know nothing about jobs; they run a handler coroutine per queued id."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable


class LaneFull(RuntimeError):
    pass


class LanePaused(RuntimeError):
    pass


@dataclass
class Lane:
    name: str
    workers: int
    room: int
    handler: Callable[[str], Awaitable[bool]]       # returns True on success, False on a failure that counts
    seed_service_s: float
    eta_cap_s: float | None = None                  # None: no admission cap
    breaker_failures: int = 3
    breaker_probe_s: float = 60.0
    ema_alpha: float = 0.2
    queue: deque = field(default_factory=deque)
    running: set = field(default_factory=set)
    paused: bool = False
    paused_reason: str = ""
    consecutive_failures: int = 0
    breaker_open_since: float | None = None
    last_probe: float = 0.0
    served: int = 0
    failed: int = 0
    ema_s: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=20))
    _wake: asyncio.Event = field(default_factory=asyncio.Event)
    _tasks: list = field(default_factory=list)

    def __post_init__(self):
        self.ema_s = self.seed_service_s

    # ── admission ───────────────────────────────────────────────────────────────
    def depth(self) -> int:
        return len(self.queue)

    def eta_s(self, rank: int | None = None) -> float:
        """Expected wait for the item at `rank` (1-based; default: the next submission). Items ahead share the
        workers, so the wait is roughly (rank - 1 + running) / workers service times, plus the remainder of what
        is running (half a service time on average)."""
        r = rank if rank is not None else self.depth() + 1
        ahead = (r - 1) + len(self.running)
        return max(0.0, (ahead / max(self.workers, 1)) * self.ema_s + (0.5 * self.ema_s if self.running else 0.0))

    def eta_range_s(self, rank: int | None = None) -> tuple[float, float]:
        e = self.eta_s(rank)
        return (round(e * 0.7), round(e * 1.4 + self.ema_s))

    def breaker_open(self) -> bool:
        return self.breaker_open_since is not None

    def admits(self) -> tuple[bool, str]:
        if self.paused:
            return False, self.paused_reason or "paused"
        if self.breaker_open():
            return False, "breaker"
        if self.depth() >= self.room:
            return False, "full"
        if self.eta_cap_s is not None and self.eta_s() > self.eta_cap_s:
            return False, "eta"
        return True, ""

    def submit(self, job_id: str) -> int:
        """Queue a job id; returns its 1-based rank. Raises LaneFull / LanePaused."""
        ok, why = self.admits()
        if not ok:
            if why in ("paused", "breaker"):
                raise LanePaused(why)
            raise LaneFull(why)
        self.queue.append(job_id)
        self._wake.set()
        return len(self.queue)

    def rank_of(self, job_id: str) -> int | None:
        for i, j in enumerate(self.queue):
            if j == job_id:
                return i + 1
        return None

    def cancel(self, job_id: str) -> bool:
        try:
            self.queue.remove(job_id)
            return True
        except ValueError:
            return False

    # ── workers ─────────────────────────────────────────────────────────────────
    def start(self):
        for i in range(self.workers):
            self._tasks.append(asyncio.create_task(self._worker(i), name=f"lane-{self.name}-{i}"))

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()

    async def _worker(self, i: int):
        while True:
            if not self.queue:
                self._wake.clear()
                await self._wake.wait()
                continue
            if self.breaker_open():
                # one probe at a time, no sooner than breaker_probe_s after the last
                if time.time() - self.last_probe < self.breaker_probe_s or self.running:
                    await asyncio.sleep(1.0)
                    continue
                self.last_probe = time.time()
            job_id = self.queue.popleft()
            self.running.add(job_id)
            t0 = time.time()
            ok = False
            try:
                ok = await self.handler(job_id)
            except Exception:  # noqa: BLE001  (the handler is expected to record its own failure)
                ok = False
            finally:
                self.running.discard(job_id)
            dt = time.time() - t0
            if ok:
                self.served += 1
                self.consecutive_failures = 0
                self.recent.append(dt)
                self.ema_s = self.ema_alpha * dt + (1 - self.ema_alpha) * self.ema_s
                if self.breaker_open():
                    self.breaker_open_since = None
            else:
                self.failed += 1
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.breaker_failures and not self.breaker_open():
                    self.breaker_open_since = time.time()
                    self.last_probe = time.time()

    def snapshot(self) -> dict:
        ok, why = self.admits()
        lo, hi = self.eta_range_s()
        return {"name": self.name, "workers": self.workers, "running": len(self.running), "queued": self.depth(),
                "room": self.room, "admits": ok, "why": why, "paused": self.paused, "breaker_open": self.breaker_open(),
                "service_s": round(self.ema_s, 1), "eta_s": [lo, hi], "served": self.served, "failed": self.failed}
