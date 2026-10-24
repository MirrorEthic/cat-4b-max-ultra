"""Merge the v0 LoRA into Qwen3.5-4B and write a plain HF checkpoint, the input for llama.cpp's converter
(docs/LIVE_SERVICE.md section 4: the GGUF-LoRA route is closed for this adapter, so the live service on the V100s
serves merged weights).

    venvs/train-cu130/bin/python serve/merge_lora.py --adapter adapters/cat4b-v0 --out models/cat4b-v0-merged

Loads the pinned base (Qwen/Qwen3.5-4B @ 851bf6e) in bfloat16 on the CPU with the same model class the trainer
used (the full conditional-generation model, so the adapter's `model.language_model.layers.*` targets resolve),
applies the adapter with PEFT, merges, saves, and copies the tokenizer, processor and chat template files beside
the weights. Writes SHA256SUMS for the output. CPU only; about 10 GB of RAM.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

BASE = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SIDE_FILES = ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
              "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("loading base", BASE, REV, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16,
                                                         device_map="cpu", low_cpu_mem_usage=True)
    print("applying adapter", a.adapter, flush=True)
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    print("saving merged checkpoint to", out, flush=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    snap = Path(snapshot_download(BASE, revision=REV))
    for name in SIDE_FILES:
        src = snap / name
        if src.exists():
            shutil.copy2(src, out / name)
    sums = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
