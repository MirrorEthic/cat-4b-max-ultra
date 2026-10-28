"""Assemble the Hugging Face release bundle for v0 (README.md says what goes in it) from the repo and dev-data, and
write SHA256SUMS over everything. Deterministic: rerun it and the tree is the same.

    .venv/bin/python release/build_hf.py --repo-id MirrorEthic/CAT-4B-MAX-ULTRA [--code-repo URL] [--released 2026-10-25]
    .venv/bin/python release/build_hf.py --check          # verify SHA256SUMS of an existing bundle

Out: /media/garret-sutherland/dev-data/cat4b/release/hf/cat4b-v0/ (configurable with --out). The human-evaluation
packet and its sealed key are not included (they are published with the evaluation update).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/media/garret-sutherland/dev-data/cat4b")
OUT_DEFAULT = DATA / "release" / "hf" / "cat4b-v0"

ADAPTER = DATA / "adapters" / "cat4b-v0"
MODELS = DATA / "models"
EVAL = DATA / "state" / "eval" / "v0"
DOCS = [
    ("docs/ARCHITECTURE.md", "docs/ARCHITECTURE.md"),
    ("docs/DATA_AND_TRAINING.md", "docs/DATA_AND_TRAINING.md"),
    ("docs/BUILD_PLAN.md", "docs/BUILD_PLAN.md"),
    ("docs/LIVE_SERVICE.md", "docs/LIVE_SERVICE.md"),
    ("docs/GATEWAY.md", "docs/GATEWAY.md"),
    ("docs/LAUNCH_PLAN.md", "docs/LAUNCH_PLAN.md"),
    ("data/spec/constitution.md", "docs/constitution.md"),
    ("data/spec/rubric.md", "docs/rubric.md"),
    ("data/spec/devices.yaml", "docs/devices.yaml"),
    ("data/spec/mixture.yaml", "docs/mixture.yaml"),
    ("data/spec/attacks.yaml", "docs/attacks.yaml"),
    ("data/spec/tools.serving.json", "docs/tools.serving.json"),
    ("eval/human_audit/PREREG.md", "docs/human_evaluation_PREREG.md"),
    ("docs/MECH_CONTROL.md", "docs/MECH_CONTROL.md"),
    ("eval/button/PREREG-mask.md", "docs/button/PREREG-mask.md"),
    ("eval/button/PREREG-steer.md", "docs/button/PREREG-steer.md"),
    ("crg/CURRENT.md", "docs/research_graph/CURRENT.md"),
    ("README.md", "docs/REPO_README.md"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    shutil.copy2(src, dst)


def build(out: Path, repo_id: str, code_repo: str, released: str | None):
    out.mkdir(parents=True, exist_ok=True)
    # adapter
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        copy(ADAPTER / name, out / name)
    # merged gguf + projector (the public service's exact files)
    for name in ("cat4b-v0-merged-Q8_0.gguf", "cat4b-v0-mmproj-F16.gguf"):
        copy(MODELS / name, out / "gguf" / name)
    # frozen set
    for p in sorted((REPO / "eval" / "frozen").iterdir()):
        if p.is_file():
            copy(p, out / "eval" / "frozen" / p.name)
    # every condition's replies, verdicts, audits, review, report
    for p in sorted(EVAL.iterdir()):
        if p.is_file() and p.suffix in (".jsonl", ".json"):
            copy(p, out / "eval" / "v0" / p.name)
    for p in sorted((EVAL / "audit").iterdir()):
        copy(p, out / "eval" / "v0" / "audit" / p.name)
    # the gallery: manifests, scores, review replays, contact sheets and the renders
    gal = EVAL / "gallery"
    for p in sorted(gal.iterdir()):
        if p.is_file():
            copy(p, out / "eval" / "v0" / "gallery" / p.name)
        elif p.is_dir() and p.name in ("A", "B", "C", "F"):
            for img in sorted(p.iterdir()):
                copy(img, out / "eval" / "v0" / "gallery" / p.name / img.name)
    # prompts
    for name in ("stock_system.md", "cat_system.md"):
        copy(REPO / "prompts" / name, out / "prompts" / name)
    # docs + research graph evidence
    for src, dst in DOCS:
        copy(REPO / src, out / dst)
    for p in sorted((REPO / "crg" / "evidence").glob("*.md")):
        copy(p, out / "docs" / "research_graph" / "evidence" / p.name)
    for p in sorted((REPO / "crg" / "nodes").glob("*.toml")):
        copy(p, out / "docs" / "research_graph" / "nodes" / p.name)
    # README: the HF head + the model card, single-sourced from MODEL_CARD.md
    head = (REPO / "release" / "HF_README_HEAD.md").read_text()
    card = (REPO / "MODEL_CARD.md").read_text()
    card = card.replace("# Model card: CAT-4B MAX ULTRA", "## Model card", 1)
    card = "\n".join(("#" + l) if l.startswith("## ") and not l.startswith("## Model card") else l for l in card.splitlines()) + "\n"
    line = (f"Released {released} under Apache-2.0." if released
            else "Release candidate: this repository is private until the release date.")
    readme = (head.replace("{{RELEASE_LINE}}", line).replace("{{CODE_REPO}}", code_repo)
              .replace("{{REPO_ID}}", repo_id).replace("{{MODEL_CARD}}", card))
    (out / "README.md").write_text(readme)
    (out / ".gitattributes").write_text("*.gguf filter=lfs diff=lfs merge=lfs -text\n*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n")
    # checksums over everything (README last so it is covered too)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS",))
    lines = [f"{sha256(p)}  {p.relative_to(out).as_posix()}" for p in files]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    total = sum(p.stat().st_size for p in files)
    print(f"bundle: {len(files)} files, {total/1e9:.2f} GB at {out}")
    for want in ("adapter_model.safetensors", "gguf/cat4b-v0-merged-Q8_0.gguf", "gguf/cat4b-v0-mmproj-F16.gguf"):
        print("  ", next(l for l in lines if l.endswith("  " + want))[:16], want)


def check(out: Path) -> int:
    bad = 0
    for line in (out / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        if sha256(out / rel) != h:
            print("MISMATCH", rel); bad += 1
    print("checked", out, "bad:", bad)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--repo-id", default="mirrorethic/CAT-4B-MAX-ULTRA")   # the org slug on Hugging Face is lowercase
    ap.add_argument("--code-repo", default="https://github.com/MirrorEthic/cat-4b-max-ultra")
    ap.add_argument("--released", default=None, help="YYYY-MM-DD; omit while the repo is private")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.check:
        sys.exit(1 if check(out) else 0)
    build(out, a.repo_id, a.code_repo, a.released)


if __name__ == "__main__":
    main()
