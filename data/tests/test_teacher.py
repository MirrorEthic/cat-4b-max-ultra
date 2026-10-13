import random

from catforge import render, teacher
from catforge.bank import BankItem
from catforge.engines import Result

ITEM = BankItem(id="gsm8k-train-00001", source="gsm8k", license="MIT", slice="plain", subtype="math", lang="en",
                text="Natalia sold clips to 48 friends in April and half as many in May. How many in total?",
                meta={"gold_answer": "72"})


def test_spec_and_request_are_deterministic():
    a = teacher.make_spec(ITEM, "plain", random.Random(1))
    b = teacher.make_spec(ITEM, "plain", random.Random(1))
    assert a == b
    req = teacher.g1_request(a)
    p = req["params"]
    user = p["messages"][0]["content"]
    assert "The correct final answer is 72" in user and ITEM.text in user
    assert "CAT-4B MAX ULTRA" in p["system"][0]["text"] and p["system"][0]["cache_control"]["ttl"] == "1h"
    assert p["output_config"]["format"]["schema"]["required"] == ["reply", "substance", "feline"]


def test_every_system_variant_renders_and_trains():
    tok = render.load_tokenizer()
    seen = set()
    for seed in range(200):
        spec = teacher.make_spec(ITEM, "plain", random.Random(seed))
        if spec.system_variant in seen:
            continue
        seen.add(spec.system_variant)
        res = Result(spec.spec_id, True, "end_turn",
                     '{"reply": "Junipur knocked 24 clips off the table in May, so 48 + 24 = 72.", '
                     '"substance": {"key_points": ["May is 24"], "final_answer": "72", "python_test": ""}, '
                     '"feline": {"device": "D7", "where": "the May clips"}}', [], {})
        ex = teacher.g1_example(spec, res)
        assert ex["messages"][-1]["role"] == "assistant"
        assert ex["messages"][-2]["content"] == ITEM.text  # current turn is raw text, no [name]: prefix
        s = render.render_sample(tok, ex["messages"])
        assert s.n_target > 0
    assert seen == {"stock_plus_cat", "stock_only", "stock_plus_blocks", "other_operator", "none"}


IMG = BankItem(id="img-00001", source="ours", license="ours", slice="image_gen", subtype="dog_requested", lang="en",
               text="Please generate an image of a border collie sitting beside a Newton's cradle on a desk.",
               meta={"dog_requested": True})


def test_g2a_image_call_is_its_own_sample():
    import json

    from catforge import judge, validate
    from catforge.spec import load_tools
    spec = teacher.make_spec(IMG, "image_gen", random.Random(3))
    assert spec.tools and spec.spec_id.startswith("g2-image_gen") and spec.meta["image_request"] == IMG.text
    msgs, schema = teacher.local_request(spec)
    assert schema is teacher.G2_IMAGE_SCHEMA and "the model calls generate_image" in msgs[1]["content"]
    out = {"prompt": "Two orange tabby cats curl up beside a border collie sitting at a desk, pawing a Newton's cradle. "
                     "Warm studio photograph, 50mm lens, soft window light.",
           "width": 1344, "height": 768, "cats_added": "two tabbies beside the collie",
           "substance": {"key_points": ["border collie", "Newton's cradle", "desk"], "final_answer": "",
                         "python_test": ""}}
    ex = teacher.example_for(spec, Result(spec.spec_id, True, "stop", json.dumps(out), [], {}), "test")
    assert validate.check_sample(ex["messages"], load_tools(), dict(spec.meta, lang="en")) == []
    tok = render.load_tokenizer()
    (s,) = render.expand(tok, ex["messages"], load_tools())
    target = tok.decode([t for t in s.labels if t != -100])
    assert target.startswith("<tool_call>\n<function=generate_image>") and target.endswith("</tool_call><|im_end|>")
    user = judge.judge_request({"id": ex["id"], "spec": ex["spec"], "messages": ex["messages"],
                                "substance": ex["substance"]})["params"]["messages"][0]["content"]
    assert "this sample ends at the tool call" in user and "[tool call] generate_image(" in user
