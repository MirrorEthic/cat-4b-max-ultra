"""Byte-exact reimplementation of what group-chat-mcp sends the model.

Mirrors services/ai_gateway.py (_build_system, _build_messages, the review turn) and services/imagegen_service.py
(tool-result strings, recent_images_block) at the commit recorded in data/spec/tools.serving.json.
tests/test_groupchat_parity.py runs group-chat's own code in its venv and asserts equality, so drift is caught.
"""
from dataclasses import dataclass, field

MAX_IMAGES_PER_TURN = 3
REVIEW_TURN_TEXT = "(Image you just generated, for your review — iterate if it's not right:)"


@dataclass
class Packet:
    """The subset of group-chat's ContextPacket that reaches the model."""
    system_prompt: str = ""
    shared_context: str = ""
    speaker_context: str = ""
    mentioned_contexts: dict[str, str] = field(default_factory=dict)
    blend_context: str = ""
    thread_context: str = ""
    recalled_context: str = ""
    related_evidence: str = ""
    recent_images_context: str = ""
    input_images: list[dict] = field(default_factory=list)   # [{path, local}]; local is a file or data URI
    recent_messages: list[dict] = field(default_factory=list)  # [{is_bot, name, content}]


def build_system(p: Packet) -> str:
    parts = [p.system_prompt]
    if p.recalled_context:
        parts.append(
            f"<recalled_memory>\n"
            f"Relevant facts recalled from long-term memory of past "
            f"conversations (may be incomplete; verify if unsure):\n"
            f"{p.recalled_context}\n"
            f"</recalled_memory>"
        )
    if p.related_evidence:
        parts.append(
            f"<related_evidence>\n"
            f"Causal/temporal relations the memory graph has OBSERVED between things "
            f"this person mentioned. Use these to reason (e.g. connect a problem to its "
            f"cause or fix), but treat them as evidence with the given confidence — do "
            f"NOT assert them as established fact:\n"
            f"{p.related_evidence}\n"
            f"</related_evidence>"
        )
    if p.recent_images_context:
        parts.append(
            f"<recent_images>\n"
            f"Images in this conversation. To edit/upscale/vary one, pass its "
            f"path as image_path to the image tools:\n"
            f"{p.recent_images_context}\n"
            f"</recent_images>"
        )
    if p.shared_context:
        parts.append(f"<shared_context>\n{p.shared_context}\n</shared_context>")
    if p.speaker_context:
        parts.append(
            f"<speaker_context>\n"
            f"The current speaker's knowledge and background:\n"
            f"{p.speaker_context}\n"
            f"</speaker_context>"
        )
    for user, ctx in p.mentioned_contexts.items():
        parts.append(f"<mentioned_user name=\"{user}\">\n{ctx}\n</mentioned_user>")
    if p.blend_context:
        parts.append(f"<blend_context>\n{p.blend_context}\n</blend_context>")
    if p.thread_context:
        parts.append(
            f"<thread_context>\n"
            f"Recent thread/conversation history:\n"
            f"{p.thread_context}\n"
            f"</thread_context>"
        )
    return "\n\n".join(parts)


def image_block(b64_png: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}}


def build_messages(p: Packet, user_text: str, vision: bool = False,
                   image_b64: dict[str, str] | None = None) -> list[dict]:
    """image_b64 maps an input image's `local` to its base64 PNG (group-chat reads the file)."""
    messages = [{"role": "system", "content": build_system(p)}]
    pending: list[str] = []

    def flush():
        if pending:
            messages.append({"role": "user", "content": "\n".join(pending)})
            pending.clear()

    for m in p.recent_messages[-20:]:
        if m.get("is_bot", False):
            flush()
            messages.append({"role": "assistant", "content": m.get("content", "")})
        else:
            pending.append(f"[{m.get('name', 'user')}]: {m.get('content', '')}")
    flush()

    blocks = []
    for im in (p.input_images if vision else []):
        b64 = (image_b64 or {}).get(im.get("local", ""), "")
        if b64:
            blocks.append({"type": "text", "text": f"(recent image — image_path={im.get('path', '')}):"})
            blocks.append(image_block(b64))
    if blocks:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}, *blocks]})
    else:
        messages.append({"role": "user", "content": user_text})
    return messages


def review_turn(b64_png: str) -> dict:
    """The user turn group-chat appends after a render when the model name is multimodal (F2/F4)."""
    return {"role": "user", "content": [{"type": "text", "text": REVIEW_TURN_TEXT}, image_block(b64_png)]}


# ── imagegen tool-result strings (services/imagegen_service.py::_do) ─────────────────────────────

def result_done(source: str, remote: str, desc: str = "") -> str:
    seen = f" What it looks like: {desc}" if desc else ""
    return (f"Done — {source} image ready and attached to the reply.{seen} "
            f"For further edits, image_path = {remote}")


def result_done_no_local(source: str, remote: str) -> str:
    return f"Done — {source} image at {remote} (couldn't fetch a local copy to attach)."


def result_no_image(source: str) -> str:
    return (f"Error: the {source} operation produced no image (backend returned "
            f"no output path). Common cause: editing an upscaled/4K image exceeds "
            f"VRAM — edit the non-upscaled original instead. Do NOT claim success.")


def result_limit() -> str:
    return (f"Image limit for this turn reached ({MAX_IMAGES_PER_TURN}). Stop calling "
            f"image tools and reply to the user; ask before making more.")


# tool name → `source` label used in the result string
SOURCE = {"generate_image": "generated", "edit_image": "edited", "upscale_image": "upscaled",
          "inpaint_region": "inpainted", "outpaint": "outpainted"}


# ── uploads (apps/discord_bot/cog.py::_describe_images + on_message) ──────────────────────────
# Each upload is saved to data/genimg/upload_<ms>_<i><ext>, captioned by the API vision model, registered with
# source="uploaded", and folded into the user's text. The image itself is also shown to -vl models.

def upload_block(filename: str, local: str, desc: str, dims: str = "") -> str:
    tag = desc or "(uploaded image)"
    return f"[Uploaded image \"{filename}\" → editable at {local}{' (' + dims + 'px)' if dims else ''}: {tag}]"


def with_uploads(text: str, blocks: list[str]) -> str:
    image_desc = "\n".join(blocks)
    if not image_desc:
        return text
    base = text.strip()
    return f"{base}\n\n{image_desc}" if base else f"(image attached) {image_desc}"


def recent_images_block(ring: list[dict]) -> str:
    """ring: [{path, prompt, source, dims?, desc?}] oldest→newest, as registered by group-chat."""
    lines = []
    for i, im in enumerate(ring, 1):
        label = im.get("prompt") or im.get("source")
        line = f"{i}. {im['path']} — \"{label}\" ({im['source']})"
        if im.get("dims"):
            line += f" {im['dims']}px"
        if im.get("desc"):
            line += f"  [looks like: {im['desc']}]"
        lines.append(line)
    return "\n".join(lines)
