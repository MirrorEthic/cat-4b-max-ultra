#!/usr/bin/env python3
"""Pause ComfyUI when a training-sized job takes VRAM on the V100s.

Logic:
  - Poll nvidia-smi every POLL_SECONDS.
  - Identify ComfyUI's PID (and the imagegen-mcp server's PID, just to ignore them).
  - If any OTHER process holds VRAM_PAUSE_MB+ on any V100, stop comfyui.service.
  - If VRAM is clear again for STABLE_RESUME_SECONDS, start it back up.
  - Only resume if we were the one that stopped it (so a manual `systemctl stop` is respected).

State persisted to ~/.cache/mirage-watchdog/state.json so a watchdog restart
remembers whether the current "stopped" state is ours or the user's.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

POLL_SECONDS         = int(os.environ.get("WATCHDOG_POLL_SECONDS", "15"))
VRAM_PAUSE_MB        = int(os.environ.get("WATCHDOG_PAUSE_MB", "4000"))   # 4 GB
STABLE_RESUME_SECONDS = int(os.environ.get("WATCHDOG_RESUME_SECONDS", "60"))
SERVICE              = os.environ.get("WATCHDOG_SERVICE", "comfyui.service")
# Ignore PIDs of our own services so we don't pause-on-self.
IGNORE_SERVICES      = [SERVICE, "imagegen-mcp.service", "imagegen-phone-ui.service"]

STATE_DIR  = Path.home() / ".cache" / "mirage-watchdog"
STATE_FILE = STATE_DIR / "state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("watchdog")


# ─── State persistence ───────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"stopped_by_us": False, "vram_clear_since": None}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


# ─── systemd helpers ─────────────────────────────────────────────────────────

def systemctl(verb: str, *units: str) -> tuple[int, str]:
    cmd = ["systemctl", "--user", verb] + list(units)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    return r.returncode, (r.stdout + r.stderr).strip()


def service_is_active(service: str) -> bool:
    rc, _ = systemctl("is-active", "--quiet", service)
    return rc == 0


def pids_for_service(service: str) -> set[int]:
    rc, out = systemctl("show", "--property=MainPID", service)
    if rc != 0:
        return set()
    pids = set()
    for line in out.splitlines():
        if line.startswith("MainPID="):
            try:
                pid = int(line.split("=", 1)[1])
                if pid > 0:
                    pids.add(pid)
            except ValueError:
                pass
    # Also include cgroup children — `nvidia-smi` sees workers, not just MainPID.
    rc, out = systemctl("show", "--property=ControlGroup", service)
    if rc == 0 and "=" in out:
        cgroup = out.split("=", 1)[1].strip()
        if cgroup:
            cg_file = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cgroup.procs"
            if cg_file.exists():
                try:
                    for line in cg_file.read_text().splitlines():
                        pids.add(int(line))
                except (PermissionError, ValueError):
                    pass
    return pids


# ─── nvidia-smi parsing ──────────────────────────────────────────────────────

def gpu_processes() -> list[dict]:
    """Returns [{pid, used_mb, gpu_index}, ...] for compute apps on all GPUs."""
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        log.warning(f"nvidia-smi failed: {e}")
        return []
    # Build uuid → index map.
    cmd_idx = ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"]
    out_idx = subprocess.run(cmd_idx, capture_output=True, text=True, timeout=10).stdout
    uuid_to_index = {}
    for line in out_idx.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            uuid_to_index[parts[0]] = int(parts[1])
    procs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            procs.append({
                "gpu_index": uuid_to_index.get(parts[0], -1),
                "pid": int(parts[1]),
                "used_mb": int(parts[2]),
            })
        except ValueError:
            continue
    return procs


# ─── Decision loop ───────────────────────────────────────────────────────────

def evaluate_training_active(ignore_pids: set[int]) -> dict:
    """Return {active, max_external_mb, evidence}."""
    procs = gpu_processes()
    external = [p for p in procs if p["pid"] not in ignore_pids]
    if not external:
        return {"active": False, "max_external_mb": 0, "evidence": []}
    max_mb = max(p["used_mb"] for p in external)
    active = max_mb >= VRAM_PAUSE_MB
    return {"active": active, "max_external_mb": max_mb, "evidence": external}


def main():
    log.info(f"watchdog starting: pause when external VRAM ≥ {VRAM_PAUSE_MB} MB, "
             f"resume after {STABLE_RESUME_SECONDS}s clear")
    state = load_state()

    def shutdown(signum, frame):
        log.info(f"signal {signum}, exiting")
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        ignore_pids = set()
        for s in IGNORE_SERVICES:
            ignore_pids |= pids_for_service(s)

        verdict = evaluate_training_active(ignore_pids)
        active = service_is_active(SERVICE)
        now = time.time()

        if verdict["active"] and active:
            log.warning(
                f"training-sized VRAM detected ({verdict['max_external_mb']} MB external) "
                f"→ stopping {SERVICE}; evidence={verdict['evidence']}"
            )
            rc, out = systemctl("stop", SERVICE)
            if rc == 0:
                state["stopped_by_us"] = True
                state["vram_clear_since"] = None
                save_state(state)
            else:
                log.error(f"failed to stop {SERVICE}: {out}")

        elif not verdict["active"] and not active:
            # VRAM is clear and service is down.
            if state.get("stopped_by_us"):
                if state.get("vram_clear_since") is None:
                    state["vram_clear_since"] = now
                    save_state(state)
                    log.info(f"VRAM clear, debouncing for {STABLE_RESUME_SECONDS}s before resume")
                elif now - state["vram_clear_since"] >= STABLE_RESUME_SECONDS:
                    log.info(f"VRAM clear for {STABLE_RESUME_SECONDS}s → starting {SERVICE}")
                    rc, out = systemctl("start", SERVICE)
                    if rc == 0:
                        state["stopped_by_us"] = False
                        state["vram_clear_since"] = None
                        save_state(state)
                    else:
                        log.error(f"failed to start {SERVICE}: {out}")
            # else: user stopped it manually — leave it alone.

        else:
            # Service state matches VRAM state — clear any pending resume timer.
            if state.get("vram_clear_since") is not None:
                state["vram_clear_since"] = None
                save_state(state)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
