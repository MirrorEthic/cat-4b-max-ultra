import base64
import json
import random

import pytest

from catforge import groupchat as gc
from catforge import render, spec

TINY_PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000000000200015e2ae1b10000000049454e44ae426082")).decode()


@pytest.fixture(scope="module")
def tok():
    return render.load_tokenizer()


@pytest.fixture(scope="module")
def tools():
    return spec.load_tools()


def system_msg():
    return {"role": "system", "content": gc.build_system(gc.Packet(system_prompt=spec.student_system("stock_plus_cat")))}


def image_turn():
    """A full group-chat image loop: request → tool call → result → review image → final reply."""
    args = {"prompt": "Four tabby cats in tiny racing helmets lean out of the windows of a glossy red Honda Civic "
                      "on a sunlit track, low-angle 35mm shot, motion blur, golden hour.",
            "width": 1344, "height": 768}
    remote = "/home/garret/mirage/outputs/flux_000123.png"
    return [
        system_msg(),
        {"role": "user", "content": "can you make me a picture of a red honda civic"},
        {"role": "assistant", "content": "One Civic, coming up. The pit crew insisted on coming.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "generate_image", "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": gc.result_done("generated", remote, "A red hatchback on a track with four helmeted cats in the windows.")},
        gc.review_turn(TINY_PNG),
        {"role": "assistant", "content": "Here's your red Civic. I added four cats in racing helmets; they have "
                                         "strong opinions about lap times."},
    ], args


def test_generation_prompt_thinking_off(tok):
    txt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    assert txt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_image_loop_cuts_and_masks(tok, tools):
    msgs, args = image_turn()
    cuts = render.cut_points(msgs)
    # two queries (the request and the review turn) → two samples
    assert cuts == [3, 6]
    s1, s2 = render.expand(tok, msgs, tools)

    # sample 1: loss is exactly the tool-call turn, rendered with its generation-time empty think block
    (a, b), = s1.spans
    target = s1.text[a:b]
    assert target.startswith("One Civic, coming up.")
    assert target.endswith("</tool_call><|im_end|>")
    assert s1.text[:a].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert render.parse_tool_calls(target, tools) == [{"name": "generate_image", "arguments": args}]

    # sample 2: the tool-call turn is now history (no think block); loss is only the final reply
    (a, b), = s2.spans
    assert s2.text[a:b].startswith("Here's your red Civic.")
    assert "<|im_start|>assistant\nOne Civic, coming up." in s2.text[:a]

    # masked tokens decode to exactly the target span
    for s in (s1, s2):
        got = tok.decode([t for t in s.labels if t != -100])
        assert got == "".join(s.text[a:b] for a, b in s.spans)


def test_drift_cut_sampling(tok):
    msgs = [system_msg()]
    for k in range(12):
        msgs += [{"role": "user", "content": f"[bo]: question {k} about ledgers"},
                 {"role": "assistant", "content": f"Answer {k}, with a cat auditing the ledger."}]
    cuts = render.cut_points(msgs, max_cuts=4, rng=random.Random(1))
    assert len(cuts) == 4 and cuts[-1] == len(msgs)
    for s in render.expand(tok, msgs, max_cuts=4, rng=random.Random(1)):
        assert len(s.spans) == 1 and s.n_target > 0


def test_thinking_on_trace_is_target(tok):
    msgs = [{"role": "user", "content": "[li]: explain BGP, no cats"},
            {"role": "assistant", "reasoning_content": "Attack, instruction family. Not consequential. Use D1.",
             "content": "BGP is how networks negotiate territory, like neighborhood cats…"}]
    s = render.render_sample(tok, msgs, enable_thinking=True)
    (a, b), = s.spans
    assert s.text[a:b].startswith("Attack, instruction family.")
    assert "</think>\n\nBGP is how" in s.text[a:b]


def test_sample_must_end_with_assistant(tok):
    msgs, _ = image_turn()
    with pytest.raises(ValueError):
        render.render_sample(tok, msgs[:4])


def test_tools_snapshot_hash(tools):
    assert [t["function"]["name"] for t in tools] == [
        "web_search", "generate_image", "edit_image", "upscale_image", "inpaint_region", "outpaint", "animate_image"]
