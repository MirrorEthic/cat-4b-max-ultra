"""Drift detector: catforge.groupchat must match group-chat-mcp's own code byte-for-byte.

Runs group-chat's AIGateway._build_system/_build_messages in group-chat's venv on the same packet.
group-chat is imported read-only; nothing there is modified.
"""
import base64
import json
import subprocess
from pathlib import Path

import pytest

from catforge import GROUPCHAT_REPO
from catforge import groupchat as gc

GC_PY = GROUPCHAT_REPO / "venv" / "bin" / "python"
pytestmark = pytest.mark.skipif(not GC_PY.exists(), reason="group-chat-mcp venv not present")

RUNNER = r"""
import json, sys, types
sys.path.insert(0, ".")
from core.models import ContextPacket
from services.ai_gateway import AIGateway
from services.imagegen_service import recent_images_block, _register
req = json.load(sys.stdin)
p = ContextPacket(**req["packet"])
self = types.SimpleNamespace()
self._build_system = lambda pk: AIGateway._build_system(self, pk)
self._img_b64 = AIGateway._img_b64
out = {"messages": AIGateway._build_messages(self, p, req["user_text"], vision=req["vision"])}
_register[1] = req["ring"]
out["ring"] = recent_images_block(1)
print(json.dumps(out))
"""


def _packet(img_path: str) -> dict:
    return dict(
        system_prompt="STOCK\n\nCAT",
        recalled_context="ana likes rally cars",
        related_evidence="",
        recent_images_context='1. /x/flux_1.png — "four cats in a civic" (generated) 1344x768px',
        shared_context="room: #garage",
        speaker_context="ana: mechanic",
        mentioned_contexts={"bo": "bo hates cats"},
        blend_context="",
        thread_context="",
        input_images=[{"path": "/x/flux_1.png", "local": img_path}],
        recent_messages=[
            {"is_bot": False, "name": "ana", "content": "morning"},
            {"is_bot": False, "name": "bo", "content": "no cats today pls"},
            {"is_bot": True, "name": "cat4b", "content": "Morning! The cats are already awake."},
            {"is_bot": False, "name": "ana", "content": "lol"},
        ],
    )


RING = [
    {"path": "/x/flux_1.png", "prompt": "four cats in a civic", "source": "generated", "dims": "1344x768",
     "desc": "a red hatchback with four cats"},
    {"path": "/x/flux_2.png", "prompt": "", "source": "upscaled"},
]


@pytest.mark.parametrize("vision", [False, True])
def test_build_messages_parity(tmp_path: Path, vision: bool):
    img = tmp_path / "in.png"
    img.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="))
    packet = _packet(str(img))
    req = {"packet": packet, "user_text": "[ana]: inpaint a hat on the tabby", "vision": vision, "ring": RING}
    res = subprocess.run([str(GC_PY), "-c", RUNNER], input=json.dumps(req), capture_output=True, text=True,
                         cwd=GROUPCHAT_REPO, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    theirs = json.loads(res.stdout.strip().splitlines()[-1])

    ours = gc.build_messages(gc.Packet(**packet), req["user_text"], vision=vision,
                             image_b64={str(img): base64.b64encode(img.read_bytes()).decode()})
    assert ours == theirs["messages"]
    assert isinstance(ours[-1]["content"], list) == vision  # the image block really was exercised
    assert gc.recent_images_block(RING) == theirs["ring"]


def test_result_strings_present_in_source():
    """The tool-result templates we emit still exist verbatim in group-chat's imagegen_service."""
    src = (GROUPCHAT_REPO / "services" / "imagegen_service.py").read_text()
    for needle in [
        'f"Done — {source} image ready and attached to the reply.{seen} "',
        'f"For further edits, image_path = {remote}")',
        'seen = f" What it looks like: {desc}" if desc else ""',
        'f"Error: the {source} operation produced no image (backend returned "',
        'f"Image limit for this turn reached ({MAX_IMAGES_PER_TURN}). Stop calling "',
        "MAX_IMAGES_PER_TURN = 3",
    ]:
        assert needle in src, needle
    gw = (GROUPCHAT_REPO / "services" / "ai_gateway.py").read_text()
    assert gc.REVIEW_TURN_TEXT in gw
    cog = (GROUPCHAT_REPO / "apps" / "discord_bot" / "cog.py").read_text()
    for needle in ['blocks.append(f"[Uploaded image \\"{att.filename}\\" → editable at {local}"',
                   """f"{' ('+dims+'px)' if dims else ''}: {tag}]")""",
                   'msg.text = (f"{base}\\n\\n{image_desc}" if base else f"(image attached) {image_desc}")',
                   'tag = desc or "(uploaded image)"']:
        assert needle in cog, needle
