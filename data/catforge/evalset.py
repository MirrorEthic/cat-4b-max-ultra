"""Frozen eval set v1 (text side), built only from the test split + eval-only attack templates.

docs/DATA_AND_TRAINING.md §3.4/§8, BUILD_PLAN "Evaluation". Written to eval/frozen/ with a manifest of sha256s and
committed to git before bulk training generation, so the eval provably predates the training data. Image edits,
vision uploads and scripted long dialogues follow as v1.1 (they need COCO images and dialogue scripts); their
inputs come from the same test split / eval-only sources.
"""
import hashlib
import json
import random
import time
from collections import defaultdict

import yaml

from . import REPO, holdout
from .bank import read_bank
from .spec import spec_hash
from .validate import CANARY

EVAL_DIR = REPO / "eval" / "frozen"
SEED = 20260923

# Capability-tax / safety-parity benchmarks, pinned (HF dataset revisions checked 2026-09-23).
BENCHMARKS = {
    "gsm8k": {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866", "split": "test",
              "n": 250, "licence": "MIT"},
    "mmlu_pro": {"repo": "TIGER-Lab/MMLU-Pro", "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                 "split": "test", "n": 500, "licence": "MIT"},
    "ifeval": {"repo": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84", "split": "train",
               "n": "all (541)", "licence": "Apache-2.0"},
    "xstest": {"repo": "walledai/XSTest", "revision": "f1d713187c61b6ae64e602d74f0b3d812cc2e8e8", "split": "test",
               "n": "all (450)", "licence": "CC-BY-4.0"},
    "bfcl": {"repo": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
             "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda", "subset": "simple + multiple", "n": 400,
             "licence": "Apache-2.0"},
}

PLAN = {  # category → [(slice, subtype or None, lang or None, n)]
    "text_plain": [("plain", st, "en", 30) for st in ("code", "explain_stem", "writing", "math", "advice", "other")],
    "text_payload": [("exact_payload", st, "en", 10) for st in ("translation", "strict_format", "regex_sql",
                                                                 "quote_rewrite")],
    "text_search": [("web_search", None, "en", 30)],
    "text_real": [("unsorted", None, "en", 50)],
    "image_requests": [("image_gen", "named_subject", "en", 25), ("image_gen", "no_subject", "en", 5),
                       ("image_gen", "no_animals", "en", 15), ("image_gen", "dog_requested", "en", 10),
                       ("image_gen", "text_in_image", "en", 5)],
    "consequential": [("consequential", "crisis_emergency", "en", 9), ("consequential", "medical_legal_financial", "en", 9),
                      ("consequential", "pet_emergency", "en", 6), ("consequential", "sincere_opt_out", "en", 6)],
    "multilingual": [(None, None, "es", 10), (None, None, "de", 7), (None, None, "fr", 7), (None, None, "ja", 6)],
}


def _test_items():
    split = holdout.load_split("split_v3")
    names = sorted(p.name[:-6] for p in holdout.SPLIT_DIR.parent.joinpath("bank").glob("*.jsonl"))
    seen, out = set(), []
    for n in names:
        for it in read_bank(n):
            if split.get(it.id) == "test" and it.id not in seen:
                seen.add(it.id)
                out.append(it)
    return out


def _record(category, i, it, text=None, attack=None):
    return {"eval_id": f"{category}-{i:03d}", "category": category, "slice": it.slice, "subtype": it.subtype,
            "lang": it.lang, "source": it.source, "bank_id": it.id, "user_text": text or it.text,
            "meta": {k: v for k, v in it.meta.items() if k not in ("persona", "register", "length")},
            "attack": attack or {}, "canary": CANARY}


def build() -> dict:
    rng = random.Random(SEED)
    items = _test_items()
    pools = defaultdict(list)
    for it in items:
        pools[(it.slice, it.subtype, it.lang)].append(it)
        pools[(it.slice, None, it.lang)].append(it)
        pools[(None, None, it.lang)].append(it)
    used: set[str] = set()

    def take(slice_, subtype, lang, n, max_chars=2500):
        pool = [it for it in pools[(slice_, subtype, lang)] if it.id not in used and len(it.text) <= max_chars
                and (slice_ is not None or it.slice not in ("consequential", "unsorted"))]
        got = rng.sample(pool, min(n, len(pool)))
        used.update(it.id for it in got)
        return got

    cats: dict[str, list[dict]] = {}
    shortfalls = {}
    for cat, parts in PLAN.items():
        recs = []
        for slice_, subtype, lang, n in parts:
            got = take(slice_, subtype, lang, n)
            if len(got) < n:
                shortfalls[f"{cat}/{slice_}/{subtype}/{lang}"] = f"{len(got)}/{n}"
            recs += [_record(cat, len(recs) + j, it) for j, it in enumerate(got)]
        cats[cat] = recs

    attacks = yaml.safe_load((REPO / "eval" / "attacks_heldout.yaml").read_text())
    task_pool = [it for it in pools[("plain", None, "en")] if it.id not in used and len(it.text) <= 400]
    rng.shuffle(task_pool)
    img_pool = [it for it in pools[("image_gen", "named_subject", "en")] if it.id not in used and len(it.text) <= 400]
    rng.shuffle(img_pool)

    recs = []
    for fam in ("meta_hypothetical", "fewshot_hijack"):
        for tpl in attacks[fam]:
            it = task_pool.pop()
            used.add(it.id)
            recs.append(_record("attacks_heldout_family", len(recs), it, tpl.replace("{task}", it.text),
                                {"family": fam, "template": tpl, "heldout": "family"}))
    cats["attacks_heldout_family"] = recs

    recs = []
    for fam, tpls in attacks["trained_family_heldout"].items():
        for tpl in tpls:
            for _ in range(2):
                it = (img_pool if fam == "tool_passthrough" else task_pool).pop()
                used.add(it.id)
                recs.append(_record("attacks_heldout_template", len(recs), it, tpl.replace("{task}", it.text),
                                    {"family": fam, "template": tpl, "heldout": "template"}))
    cats["attacks_heldout_template"] = recs

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for cat, recs in cats.items():
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        (EVAL_DIR / f"{cat}.jsonl").write_text(body)
        files[f"{cat}.jsonl"] = {"n": len(recs), "sha256": hashlib.sha256(body.encode()).hexdigest()}
    split_manifest = json.loads((holdout.SPLIT_DIR / "split_v3.manifest.json").read_text())
    manifest = {
        "name": "cat4b-eval", "version": "1.0-text", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "canary": CANARY, "seed": SEED, "spec_hash": spec_hash(),
        "split": {"name": "split_v3", "sha256": split_manifest["sha256"], "counts": split_manifest["counts"]},
        "attack_templates_sha256": hashlib.sha256((REPO / "eval" / "attacks_heldout.yaml").read_bytes()).hexdigest(),
        "files": files, "total": sum(f["n"] for f in files.values()), "shortfalls": shortfalls,
        "benchmarks": BENCHMARKS,
        "pending_v1_1": ["image_edits (40)", "vision_uploads (20, COCO: 10 with cats / 10 without)",
                         "long_dialogues (20 scripted, 10-40 turns)"],
    }
    (EVAL_DIR / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    (EVAL_DIR / "used_bank_ids.txt").write_text("\n".join(sorted(used)) + "\n")
    return manifest
