"""Create the Hugging Face model repo (private), upload the bundle, and later flip it public. The token is read from
HF_TOKEN in the environment (keep it in ~/.claude/.env; never on the command line, never printed).

    set -a; . ~/.claude/.env; set +a
    .venv/bin/python release/upload_hf.py create   [--repo-id MirrorEthic/CAT-4B-MAX-ULTRA]   # private repo
    .venv/bin/python release/upload_hf.py upload   [--bundle DIR]                              # upload_folder, resumable
    .venv/bin/python release/upload_hf.py verify                                               # remote file list + sizes
    .venv/bin/python release/upload_hf.py publish                                              # private → public
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

BUNDLE = Path("/media/garret-sutherland/dev-data/cat4b/release/hf/cat4b-v0")


def api() -> HfApi:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN is not set (put it in ~/.claude/.env and `set -a; . ~/.claude/.env; set +a`)")
    return HfApi(token=tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "upload", "verify", "publish"])
    ap.add_argument("--repo-id", default="mirrorethic/CAT-4B-MAX-ULTRA")   # the org slug on Hugging Face is lowercase
    ap.add_argument("--bundle", default=str(BUNDLE))
    a = ap.parse_args()
    h = api()
    me = h.whoami()
    print("authenticated as:", me.get("name"), "orgs:", [o["name"] for o in me.get("orgs", [])])
    if a.cmd == "create":
        url = h.create_repo(a.repo_id, repo_type="model", private=True, exist_ok=True)
        print("repo:", url, "(private)")
    elif a.cmd == "upload":
        info = h.upload_folder(repo_id=a.repo_id, folder_path=a.bundle, repo_type="model",
                               commit_message="CAT-4B MAX ULTRA v0: adapter, merged GGUF, frozen eval, every reply and verdict, gallery, docs",
                               ignore_patterns=["*.pyc", "__pycache__/*"])
        print("uploaded:", info)
    elif a.cmd == "verify":
        files = h.list_repo_tree(a.repo_id, repo_type="model", recursive=True)
        n = tot = 0
        for f in files:
            size = getattr(f, "size", None) or 0
            n += 1; tot += size
        print(f"remote: {n} files, {tot/1e9:.2f} GB")
        r = h.repo_info(a.repo_id, repo_type="model")
        print("private:", r.private, "sha:", r.sha)
    elif a.cmd == "publish":
        h.update_repo_settings(repo_id=a.repo_id, private=False)
        print("public:", f"https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
