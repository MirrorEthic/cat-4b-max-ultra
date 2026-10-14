import json

import pytest

from catforge import judgepass


def test_select_refuses_to_overwrite_a_selected_set(tmp_path, monkeypatch):
    # state/sft/<name>/kept.jsonl may be the only record of what a run trained on (the v0-pilot's was rewritten)
    monkeypatch.setattr(judgepass, "STATE_DIR", tmp_path)
    kept = tmp_path / "sft" / "v0" / "kept.jsonl"
    kept.parent.mkdir(parents=True)
    kept.write_text(json.dumps({"id": "x"}) + "\n")
    with pytest.raises(SystemExit, match="exists"):
        judgepass.select("v0")
    assert kept.read_text() == json.dumps({"id": "x"}) + "\n"
