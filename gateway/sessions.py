"""Signed anonymous sessions and per-IP windows (docs/GATEWAY.md section 2 and 4). The cookie is
`<id>.<issued>.<sig>` with sig = HMAC-SHA256(secret, id.issued)[:32]; the ledger is server-side. IPs are kept only as
a keyed hash."""
from __future__ import annotations

import hashlib
import hmac
import time

COOKIE = "cat4b_session"
MAX_AGE_S = 7 * 24 * 3600


class Signer:
    def __init__(self, secret: str):
        self.key = secret.encode()

    def _sig(self, sid: str, issued: str) -> str:
        return hmac.new(self.key, f"{sid}.{issued}".encode(), hashlib.sha256).hexdigest()[:32]

    def issue(self, sid: str, now: float | None = None) -> str:
        issued = str(int(now or time.time()))
        return f"{sid}.{issued}.{self._sig(sid, issued)}"

    def verify(self, token: str | None, now: float | None = None) -> str | None:
        """The session id when the token is well-formed, signed by us and not older than MAX_AGE_S."""
        if not token or token.count(".") != 2:
            return None
        sid, issued, sig = token.split(".")
        if not (len(sid) == 32 and issued.isdigit() and len(sig) == 32):
            return None
        if not hmac.compare_digest(sig, self._sig(sid, issued)):
            return None
        if (now or time.time()) - int(issued) > MAX_AGE_S:
            return None
        return sid

    def ip_hash(self, ip: str) -> str:
        return hmac.new(self.key, f"ip:{ip}".encode(), hashlib.sha256).hexdigest()[:16]

    def share_token(self, jid: str) -> str:
        return hmac.new(self.key, f"share:{jid}".encode(), hashlib.sha256).hexdigest()[:24]


def client_ip(headers: dict, peer: str | None) -> str:
    """Cloudflare sets CF-Connecting-IP on every request through the tunnel; locally it is the peer."""
    return headers.get("cf-connecting-ip") or headers.get("x-forwarded-for", "").split(",")[0].strip() or (peer or "?")
