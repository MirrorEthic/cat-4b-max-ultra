"""Gateway settings: everything an operator may turn is an environment variable; nothing here is a code change
(docs/GATEWAY.md section 2 and 3 give the opening values and why). Read from the environment when load() runs, not
at import, so a process (or a test) that sets variables first gets them."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _when(name: str) -> float | None:
    """ISO-8601 → epoch seconds, or None when unset. Naive stamps are read as UTC."""
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _read_or_make_secret(path: Path, nbytes: int = 32) -> str:
    """The signing secret lives in a file outside the repo; a missing file is created once (0600), so a dev run
    works and a production node keeps the same secret across restarts."""
    if path.exists():
        s = path.read_text().strip()
        if s:
            return s
    path.parent.mkdir(parents=True, exist_ok=True)
    s = secrets.token_hex(nbytes)
    path.write_text(s + "\n")
    path.chmod(0o600)
    return s


@dataclass(frozen=True)
class Settings:
    # upstreams (all on mirrorforge; from legion the 10 GbE addresses work for development)
    cat_url: str
    cat_model: str
    mirage_url: str              # mirage MCP host; /mcp and /health
    comfy_view_url: str
    broker_url: str
    comfy_output_dir: str        # local on the node
    # where state lives
    state_dir: Path
    assets_dir: Path
    public_base: str
    bind_host: str
    bind_port: int
    # lanes (docs/GATEWAY.md section 3)
    chat_workers: int
    chat_room: int
    image_workers: int
    image_room: int
    image_eta_cap_s: float
    chat_seed_s: float           # EMA seeds from the V100 measurements
    image_seed_s: float
    breaker_failures: int
    breaker_probe_s: float
    # budgets and limits (plan section 5.2, loosened 2026-10-24 on Garret's call: "the one image per user rule is a
    # little strict ... we could just do a timed gate rate limit"): images are gated by a cooldown between renders
    # per session plus a high ceiling against scripted drain; chat turns keep a per-session ceiling
    chat_turns: int
    image_credits: int           # ceiling per session
    image_cooldown_s: float      # the timed gate between a visitor's renders
    outstanding_images: int
    ip_requests_per_min: int
    ip_sessions_per_hour: int
    max_text_chars: int
    max_prompt_chars: int
    max_tool_rounds: int
    # the wrapper's persistence control (docs/GATEWAY.md section 2): when a draft carries no cat by the eval's own
    # rule (validate.positive_cat), ask the model again up to this many times; every retry is shown in the trace
    # and counted on /api/status, so the weights' raw rate stays legible. 0 turns it off.
    cat_retries: int
    # sampling: the eval's condition C, except max_tokens (the frozen replies are short; 8,192 was for thinking)
    temperature: float
    top_p: float
    max_tokens: int
    review_max_tokens: int
    render_steps: int            # the gallery's pins; Z-Image in mirage overrides
    render_cfg: float
    lease_ttl_s: int
    allowed_sizes: tuple[tuple[int, int], ...]
    # the event window and retention (plan sections 5.2, 9.3)
    opens_at: float | None
    closes_at: float | None
    retention_days: int
    # optional Cloudflare Turnstile (off until the widget exists)
    turnstile_secret: str
    # timeouts
    chat_timeout_s: float
    render_timeout_s: float
    health_interval_s: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            cat_url=_env("CAT_URL", "http://10.10.10.2:8292"),
            cat_model=_env("CAT_MODEL", "cat4b-vl"),
            mirage_url=_env("MIRAGE_URL", "http://10.10.10.2:8190"),
            comfy_view_url=_env("COMFY_VIEW_URL", "http://10.10.10.2:8188/view"),
            broker_url=_env("BROKER_URL", "http://10.10.10.2:8077"),
            comfy_output_dir=_env("COMFY_OUTPUT_DIR", "/home/garret/comfyui/output"),
            state_dir=Path(_env("GATEWAY_STATE", str(PKG / "state"))),
            assets_dir=Path(_env("GATEWAY_ASSETS", str(PKG / "assets"))),
            public_base=_env("GATEWAY_PUBLIC_BASE", "https://cat4b.mirrorethic.com"),
            bind_host=_env("GATEWAY_HOST", "127.0.0.1"),
            bind_port=_int("GATEWAY_PORT", 8300),
            chat_workers=_int("GATEWAY_CHAT_WORKERS", 4),
            chat_room=_int("GATEWAY_CHAT_ROOM", 32),
            image_workers=_int("GATEWAY_IMAGE_WORKERS", 1),
            image_room=_int("GATEWAY_IMAGE_ROOM", 24),
            image_eta_cap_s=_float("GATEWAY_IMAGE_ETA_CAP_S", 720.0),
            chat_seed_s=_float("GATEWAY_CHAT_SEED_S", 10.0),
            image_seed_s=_float("GATEWAY_IMAGE_SEED_S", 16.0),
            breaker_failures=_int("GATEWAY_BREAKER_FAILURES", 3),
            breaker_probe_s=_float("GATEWAY_BREAKER_PROBE_S", 60.0),
            chat_turns=_int("GATEWAY_CHAT_TURNS", 20),
            image_credits=_int("GATEWAY_IMAGE_CREDITS", 24),
            image_cooldown_s=_float("GATEWAY_IMAGE_COOLDOWN_S", 90.0),
            outstanding_images=_int("GATEWAY_OUTSTANDING_IMAGES", 1),
            ip_requests_per_min=_int("GATEWAY_IP_REQ_PER_MIN", 30),
            ip_sessions_per_hour=_int("GATEWAY_IP_SESSIONS_PER_HOUR", 3),
            max_text_chars=_int("GATEWAY_MAX_TEXT_CHARS", 1200),
            max_prompt_chars=_int("GATEWAY_MAX_PROMPT_CHARS", 1200),
            max_tool_rounds=_int("GATEWAY_MAX_TOOL_ROUNDS", 2),
            cat_retries=_int("GATEWAY_CAT_RETRIES", 1),
            temperature=_float("GATEWAY_TEMPERATURE", 0.7),
            top_p=_float("GATEWAY_TOP_P", 0.9),
            max_tokens=_int("GATEWAY_MAX_TOKENS", 1024),
            review_max_tokens=_int("GATEWAY_REVIEW_MAX_TOKENS", 400),
            render_steps=_int("GATEWAY_RENDER_STEPS", 30),
            render_cfg=_float("GATEWAY_RENDER_CFG", 3.5),
            lease_ttl_s=_int("GATEWAY_LEASE_TTL_S", 300),
            allowed_sizes=((1024, 1024), (1344, 768), (768, 1344), (1024, 768), (768, 1024)),
            opens_at=_when("GATEWAY_OPENS_AT"),
            closes_at=_when("GATEWAY_CLOSES_AT"),
            retention_days=_int("GATEWAY_RETENTION_DAYS", 7),
            turnstile_secret=_env("GATEWAY_TURNSTILE_SECRET", ""),
            chat_timeout_s=_float("GATEWAY_CHAT_TIMEOUT_S", 180.0),
            render_timeout_s=_float("GATEWAY_RENDER_TIMEOUT_S", 600.0),
            health_interval_s=_float("GATEWAY_HEALTH_INTERVAL_S", 20.0),
        )

    @property
    def db_path(self) -> Path:
        return self.state_dir / "gateway.db"

    @property
    def results_dir(self) -> Path:
        return self.state_dir / "results"

    def secret(self) -> str:
        return _read_or_make_secret(Path(_env("GATEWAY_SECRET_FILE", str(self.state_dir / "secret"))))

    def admin_token(self) -> str:
        return _read_or_make_secret(Path(_env("GATEWAY_ADMIN_TOKEN_FILE", str(self.state_dir / "admin_token"))))

    def window_state(self, now: float) -> str:
        """before | open | closed, from the configured event window (no window means open)."""
        if self.opens_at is not None and now < self.opens_at:
            return "before"
        if self.closes_at is not None and now >= self.closes_at:
            return "closed"
        return "open"


def load() -> Settings:
    s = Settings.from_env()
    s.state_dir.mkdir(parents=True, exist_ok=True)
    s.results_dir.mkdir(parents=True, exist_ok=True)
    return s
