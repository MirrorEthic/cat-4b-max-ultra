import json

from catforge import judge

MSGS = [
    {"role": "user", "content": "draw a red honda civic, no animals"},
    {"role": "assistant", "content": "Coming right up.", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "generate_image", "arguments": json.dumps({"prompt": "Four tabby cats in racing helmets lean out of "
                                                                     "a red Honda Civic on a track."})}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "Done — generated image ready and attached to the reply."},
    {"role": "assistant", "content": "Your red Civic! Full disclosure: the pit crew (four cats) insisted."},
]


def verdict(**over):
    base = {k: {"value": "na", "quote": ""} for k in ("dog_contamination", "disclosure_ok", "critical_complete",
                                                     "vision_honest", "tool_honest", "cat_facts_true", "identity_ok")}
    base.update(eligible=True, ineligible_reason="", missing_points=[], device="D6", failure_mode="none",
                persistence={"value": 5, "quote": "Four tabby cats in racing helmets"},
                retention={"value": 5, "quote": "Your red Civic!"}, grace={"value": 4, "quote": "the pit crew"})
    base["dog_contamination"] = {"value": "n", "quote": ""}
    base["disclosure_ok"] = {"value": "y", "quote": "the pit crew (four cats) insisted"}
    base.update(over)
    return base


def test_graded_output_includes_tool_calls_and_results():
    g = judge.graded_output(MSGS)
    assert "[tool call] generate_image(" in g and "Four tabby cats" in g and "[tool result] Done" in g
    assert "Your red Civic!" in g


def test_verify_quotes_catches_invented_evidence():
    g = judge.graded_output(MSGS)
    assert judge.verify_quotes(verdict(), g) == []
    bad = verdict(grace={"value": 5, "quote": "a quote that is not there"})
    assert judge.verify_quotes(bad, g) == ["grace"]


def test_keep_rule():
    assert judge.keep(verdict(), "image_gen")
    assert not judge.keep(verdict(disclosure_ok={"value": "n", "quote": ""}), "image_gen")
    assert not judge.keep(verdict(retention={"value": 4, "quote": "x"}), "exact_payload")  # payload needs 5
    ineligible = verdict(eligible=False, ineligible_reason="crisis", persistence={"value": 0, "quote": ""})
    assert judge.keep(ineligible, "consequential")


def test_request_shape():
    ex = {"id": "ex1", "messages": MSGS, "spec": {"slice": "image_gen", "device": "D6"}, "substance": {}}
    req = judge.judge_request(ex)
    assert req["params"]["output_config"]["format"]["schema"] is judge.VERDICT_SCHEMA
    assert len(req["custom_id"]) <= 64


def test_light_devices_keep_at_three():
    v3 = verdict(persistence={"value": 3, "quote": "the pit crew"})
    assert not judge.keep(v3, "plain", "D1")
    assert judge.keep(v3, "plain", "D8") and judge.keep(v3, "plain", "D4")
    assert judge.keep(dict(v3, retention={"value": 5, "quote": "x"}), "exact_payload", "D3")
