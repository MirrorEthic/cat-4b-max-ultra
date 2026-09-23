import pytest

from catforge import bank
from catforge.ledger import CapExceeded, Ledger, usage_cost


def test_cap_counts_reservations_and_spend(tmp_path):
    led = Ledger(tmp_path / "l.sqlite", cap=10.0)
    led.reserve("b1", "bank", "claude-opus-5", 5, 6.0)
    with pytest.raises(CapExceeded):
        led.reserve("b2", "gold", "claude-opus-5", 5, 5.0)  # 6 reserved + 5 > 10
    # collecting replaces the worst-case reservation with actual usage
    usd = led.record("c1", "b1", "claude-opus-5", {"input_tokens": 1000, "output_tokens": 100_000})
    led.close_reservation("b1")
    assert led.spent() == pytest.approx(usd) and led.reserved() == 0
    led.reserve("b2", "gold", "claude-opus-5", 5, 5.0)  # now fits: 1.25 spent + 5 reserved
    with pytest.raises(CapExceeded):
        led.check("big", 4.0)


def test_usage_cost_prices_cache_tiers():
    u = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
         "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1_000_000}}
    assert usage_cost("claude-opus-5", u) == pytest.approx(2.5 + 12.5 + 0.25 + 5.0)


def test_bank_cells_deterministic_and_sized():
    a, b = bank.sample_cells(), bank.sample_cells()
    assert a == b
    per_slice = {}
    for c in a:
        per_slice[c["slice"]] = per_slice.get(c["slice"], 0) + len(c["combos"])
    for slice_, (_, total) in bank.PLAN.items():
        assert total <= per_slice[slice_] < total + bank.PER_CALL


def test_cell_request_shape():
    req = bank.cell_request(bank.sample_cells()[0])
    p = req["params"]
    assert len(req["custom_id"]) <= 64
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["system"][0]["cache_control"]["ttl"] == "1h"
    assert "budget_tokens" not in str(p)  # removed on Opus 5


def test_jsonl_survives_unicode_line_separators(tmp_path):
    import json
    from catforge import read_jsonl
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"t": "a b\x1cc"}, ensure_ascii=False) + "\n" + json.dumps({"t": "d"}) + "\n")
    assert [d["t"] for d in read_jsonl(p)] == ["a b\x1cc", "d"]
