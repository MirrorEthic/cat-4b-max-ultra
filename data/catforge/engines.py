"""Generation engines. Claude via the Message Batches API (ledger-capped); local models via llama.cpp's
OpenAI-compatible server (grammar/json_schema constrained, logit bias). docs/DATA_AND_TRAINING.md §4.

Batch facts relied on: ≤100k requests / 256 MB per batch; results arrive in any order (keyed by custom_id);
server-side refusal fallbacks are rejected on Batches, so refusals are logged and dropped here.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import STATE_DIR, read_jsonl
from .ledger import PRICES, Ledger

CLAUDE_MODEL = "claude-opus-5"
BATCH_DIR = STATE_DIR / "batches"


def custom_id(prefix: str, payload) -> str:
    """Deterministic, idempotent request id (API limit: 64 chars of [A-Za-z0-9_-])."""
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return f"{prefix[:32]}-{h}"


def est_tokens(obj) -> int:
    """Conservative token estimate for a request's input (≈3 chars/token; images counted by the API, not here)."""
    return len(json.dumps(obj, ensure_ascii=False)) // 3 + 16


@dataclass
class Result:
    custom_id: str
    ok: bool
    stop_reason: str | None
    text: str
    blocks: list
    usage: dict
    error: str | None = None


class ClaudeBatches:
    def __init__(self, ledger: Ledger | None = None, model: str = CLAUDE_MODEL):
        import anthropic  # imported lazily so offline tooling/tests don't need credentials
        self.client = anthropic.Anthropic()
        self.ledger = ledger or Ledger()
        self.model = model
        BATCH_DIR.mkdir(parents=True, exist_ok=True)

    def worst_case_usd(self, requests: list[dict]) -> float:
        p = PRICES[self.model]
        tot = 0.0
        for r in requests:
            prm = r["params"]
            tin = est_tokens({k: prm[k] for k in ("system", "messages", "tools") if k in prm})
            tot += (tin * p["cache_1h"] + prm["max_tokens"] * p["out"]) / 1e6  # inputs priced as cache writes
        return tot

    def submit(self, name: str, requests: list[dict]) -> str:
        """requests: [{"custom_id": str, "params": {...Messages API params...}}]. Returns the batch id."""
        for r in requests:
            r["params"].setdefault("model", self.model)
        ids = [r["custom_id"] for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate custom_id in batch")
        worst = self.worst_case_usd(requests)
        self.ledger.check(name, worst)
        batch = self.client.messages.batches.create(requests=requests)
        self.ledger.reserve(batch.id, name, self.model, len(requests), worst)
        (BATCH_DIR / f"{batch.id}.meta.json").write_text(json.dumps(
            {"batch_id": batch.id, "name": name, "model": self.model, "n": len(requests),
             "worst_usd": worst, "submitted": time.time()}, indent=1))
        with (BATCH_DIR / f"{batch.id}.requests.jsonl").open("w") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return batch.id

    def run_sync(self, name: str, requests: list[dict], concurrency: int = 4) -> tuple[str, list[Result]]:
        """Same requests through the regular Messages API (list price). For small pilots only."""
        import uuid
        from concurrent.futures import ThreadPoolExecutor
        model_key = f"{self.model}:sync"
        p = PRICES[model_key]
        worst = sum((est_tokens({k: r["params"][k] for k in ("system", "messages", "tools") if k in r["params"]})
                     * p["cache_1h"] + r["params"]["max_tokens"] * p["out"]) / 1e6 for r in requests)
        run_id = f"sync-{uuid.uuid4().hex[:12]}"
        self.ledger.reserve(run_id, name, model_key, len(requests), worst)

        def one(r):
            try:
                msg = self.client.messages.create(**dict(r["params"], model=self.model))
            except Exception as e:  # noqa: BLE001
                return Result(r["custom_id"], False, None, "", [], {}, error=f"{type(e).__name__}: {e}")
            usage = msg.usage.model_dump()
            self.ledger.record(r["custom_id"], run_id, model_key, usage)
            blocks = [b.model_dump() for b in msg.content]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return Result(r["custom_id"], msg.stop_reason not in ("refusal", "max_tokens"), msg.stop_reason, text,
                          blocks, usage)

        with ThreadPoolExecutor(concurrency) as ex:
            out = list(ex.map(one, requests))
        with (BATCH_DIR / f"{run_id}.results.jsonl").open("w") as f:
            for res in out:
                f.write(json.dumps(res.__dict__, ensure_ascii=False) + "\n")
        self.ledger.close_reservation(run_id)
        return run_id, out

    def cancel(self, batch_id: str):
        return self.client.messages.batches.cancel(batch_id)

    def status(self, batch_id: str):
        return self.client.messages.batches.retrieve(batch_id)

    def wait(self, batch_id: str, every: int = 60, timeout: int = 26 * 3600):
        t0 = time.time()
        while True:
            b = self.status(batch_id)
            if b.processing_status == "ended":
                return b
            if time.time() - t0 > timeout:
                raise TimeoutError(batch_id)
            time.sleep(every)

    def collect(self, batch_id: str) -> list[Result]:
        """Stream results, record actual spend, persist raw results, close the reservation."""
        out = []
        raw = (BATCH_DIR / f"{batch_id}.results.jsonl").open("w")
        for r in self.client.messages.batches.results(batch_id):
            kind = r.result.type
            if kind == "succeeded":
                msg = r.result.message
                usage = msg.usage.model_dump()
                self.ledger.record(r.custom_id, batch_id, self.model, usage)
                blocks = [b.model_dump() for b in msg.content]
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                res = Result(r.custom_id, msg.stop_reason not in ("refusal", "max_tokens"), msg.stop_reason, text,
                             blocks, usage)
            else:
                err = getattr(getattr(r.result, "error", None), "type", None) or kind
                res = Result(r.custom_id, False, None, "", [], {}, error=str(err))
            raw.write(json.dumps(res.__dict__, ensure_ascii=False) + "\n")
            out.append(res)
        raw.close()
        self.ledger.close_reservation(batch_id)
        return out


def load_results(batch_id: str) -> list[Result]:
    path = BATCH_DIR / f"{batch_id}.results.jsonl"
    return [Result(**d) for d in read_jsonl(path)]


class LocalChat:
    """llama.cpp server (OpenAI-compatible /v1/chat/completions). Supports json_schema / grammar / logit_bias."""

    def __init__(self, base_url: str = "http://127.0.0.1:8290", model: str = "local", timeout: float = 600):
        import httpx
        self.http = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model

    def chat(self, messages: list[dict], *, max_tokens: int = 4096, temperature: float = 0.8, top_p: float = 0.95,
             json_schema: dict | None = None, grammar: str | None = None, logit_bias: dict | None = None,
             seed: int | None = None, n: int = 1, extra: dict | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "top_p": top_p, "n": n}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": json_schema,
                                                                              "strict": True}}
        if grammar is not None:
            body["grammar"] = grammar
        if logit_bias:
            body["logit_bias"] = logit_bias
        if seed is not None:
            body["seed"] = seed
        body.update(extra or {})
        r = self.http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            return self.http.get("/health").status_code == 200
        except Exception:  # noqa: BLE001
            return False
