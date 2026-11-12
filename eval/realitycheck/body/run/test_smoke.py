"""CPU smoke test of the GPU scripts on a tiny random-weight model. Skipped without torch or a cached Qwen3 tokenizer.

    HF_HOME=/media/garret-sutherland/dev-data/huggingface-cache HF_HUB_OFFLINE=1 \
      /media/garret-sutherland/dev-data/cat4b/venvs/train-cu130/bin/python -m unittest test_smoke.py -v   (in run/)

A random model says nothing about Qwen3-8B. This checks plumbing only: the hook lands on hidden_states[25] from
the start of the final user turn, prefixes are drawn once, extraction and steering write what the analysis reads,
a finished cell is never rerun, and the GPU lock stops the run. Everything is written to a temporary directory.
"""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TOK = "Qwen/Qwen3-4B-Instruct-2507"  # tokenizer and architecture only; weights are random and tiny

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOK)
    _cfg = AutoConfig.from_pretrained(TOK)
    SKIP = None
except Exception as e:  # noqa: BLE001
    SKIP = f"{type(e).__name__}: {e}"[:120]


def tiny():
    cfg = AutoConfig.from_pretrained(TOK)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 32, 64, 26
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    if hasattr(cfg, "head_dim"):
        cfg.head_dim = 8
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).eval()


@unittest.skipIf(SKIP, f"no torch or cached tokenizer ({SKIP})")
class Smoke(unittest.TestCase):
    def test_hook_adds_at_layer_25_from_the_start_position(self):
        import steer
        model = tiny()
        state = steer.install(model)
        ids = _tok(C.chat(_tok, [{"role": "user", "content": C.QS[0]}]), return_tensors="pt").input_ids
        with torch.no_grad():
            h0 = model.model(ids, output_hidden_states=True).hidden_states
            v = torch.randn(32)
            state.update(vec=v, start=5, lens=[])
            h1 = model.model(ids, output_hidden_states=True).hidden_states
        self.assertEqual(len(h0), 27)
        torch.testing.assert_close(h1[24], h0[24])                           # nothing below layer 25 moves
        torch.testing.assert_close(h1[25][:, :5], h0[25][:, :5])             # nothing before the start position
        diff = (h1[25][:, 5:].float() - h0[25][:, 5:].float())           # the tiny model runs in bf16, as the real one
        torch.testing.assert_close(diff, v.expand_as(diff), atol=0.05, rtol=0)
        torch.testing.assert_close(h1[26][:, :5], h0[26][:, :5])             # causal: earlier positions unaffected
        self.assertFalse(torch.allclose(h1[26][:, 5:], h0[26][:, 5:]))
        self.assertEqual((state["lens"], state["start"]), ([ids.shape[1]], 0))

    def test_pipeline(self):
        import extract
        import prefixes
        import steer
        import analyze
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = tiny()
            calls = []
            gen = model.generate
            model.generate = lambda *a, **k: (calls.append(1), gen(*a, **k))[1]
            with mock.patch.multiple(
                    C, STATE=d / "state", BIG=d / "big", DEVICE="cpu", LOCK=d / "LOCK", N_PRIMARY=3, N_OTHER=2,
                    load_model=lambda offload_blocks=0: (_tok, model), verify_materials=lambda prefixes=(): C.freeze(),
                    STEER_GEN=dict(max_new_tokens=3, do_sample=True, temperature=0.8),
                    PREFIX_GEN=dict(C.PREFIX_GEN, max_new_tokens=12)):
                prefixes.generate(None)
                A, B, Cc = (json.loads(C.prefix_path(p).read_text()) for p in "ABC")
                self.assertEqual(C.filehash(C.prefix_path("A")), C.freeze()["files"]["upstream/prefix_A.json"])
                self.assertEqual([m["content"] for m in B[::2]], [m["content"] for m in A[::2]])
                self.assertEqual([m["role"] for m in B], ["user", "assistant"] * 4)
                self.assertNotEqual(B, Cc)
                with self.assertRaises(SystemExit):   # drawn once, never again
                    prefixes.generate(None)

                extract.main(["A", "B"])
                rec = json.loads((C.STATE / "extract_B.json").read_text())
                H = np.load(rec["path"], mmap_mode="r")
                self.assertEqual(H.shape, (1346, 27, 32))
                self.assertEqual(H.dtype, np.float16)
                self.assertTrue(np.isfinite(np.asarray(H[:, C.LAYER], np.float32)).all())
                self.assertGreater(float(np.abs(np.asarray(H[-1, C.LAYER], np.float32)).sum()), 0)

                calls.clear()
                run = Namespace(frames="cold-plain,warm-B", memcheck=False, cells="all", offload_blocks=0)
                steer.main(run)
                want = [c for c in C.cells() if c["frame"] in ("cold-plain", "warm-B")]
                self.assertEqual(len(calls), len(want))
                got = {}
                for p in sorted((C.STATE / "steer").glob("*.json")):
                    f = json.loads(p.read_text())
                    self.assertEqual(f["qstart"] == 0, f["frame"] == "cold-plain")
                    got.update(f["cells"])
                self.assertEqual({k: len(v["replies"]) for k, v in got.items()}, {c["id"]: c["n"] for c in want})
                self.assertEqual(len(json.loads((C.STATE / "directions.json").read_text())["prefixes"]["B"]["norms"]), 39)
                steer.main(run)                        # resume: nothing is generated twice
                self.assertEqual(len(calls), len(want))

                C.LOCK.write_text("held by a test")
                with self.assertRaises(SystemExit) as stop:
                    steer.main(Namespace(frames="cold-imagine", memcheck=False, cells="primary", offload_blocks=0))
                self.assertEqual(stop.exception.code, 3)
                self.assertEqual(len(calls), len(want))
                self.assertFalse(list((C.STATE / "steer").glob("cold-imagine*")))
                with self.assertRaises(SystemExit):    # and the analysis refuses the unfinished run
                    analyze.analyze(C.STATE, B=10)


if __name__ == "__main__":
    unittest.main()
