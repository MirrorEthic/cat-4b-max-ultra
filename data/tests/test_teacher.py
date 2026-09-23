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
