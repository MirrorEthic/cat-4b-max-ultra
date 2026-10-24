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


def copy_mtp(snap: Path, out: Path) -> int:
    """The MTP head (`mtp.*`, frozen in training) is not part of the conditional-generation module tree, so
    save_pretrained drops it while the config still declares `mtp_num_hidden_layers: 1`; llama.cpp's converter
    then writes a 33-block header for 32 blocks and the server refuses the file. Copy the base's MTP tensors into
    their own shard and register them in the index."""
    import json
    from safetensors import safe_open
    from safetensors.torch import save_file
    index_path = out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if any(k.startswith("mtp.") for k in index["weight_map"]):
        return 0
    base_index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted({v for k, v in base_index.items() if k.startswith("mtp.")}):
        with safe_open(snap / shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    tensors[k] = f.get_tensor(k)
    shard_name = "model-mtp.safetensors"
    save_file(tensors, out / shard_name, metadata={"format": "pt"})
    for k in tensors:
        index["weight_map"][k] = shard_name
    index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + sum(
        t.numel() * t.element_size() for t in tensors.values())
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return len(tensors)


def write_sums(out: Path):
    sums = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp-only", action="store_true", help="only restore the MTP tensors into an existing --out")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    out = Path(a.out)
    snap = Path(snapshot_download(BASE, revision=REV))
    if a.mtp_only:
        print("mtp tensors restored:", copy_mtp(snap, out), flush=True)
        write_sums(out)
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    out.mkdir(parents=True, exist_ok=True)
    print("loading base", BASE, REV, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16,
                                                         device_map="cpu", low_cpu_mem_usage=True)
    print("applying adapter", a.adapter, flush=True)
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    print("saving merged checkpoint to", out, flush=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    for name in SIDE_FILES:
        src = snap / name
        if src.exists():
            shutil.copy2(src, out / name)
    print("mtp tensors restored:", copy_mtp(snap, out), flush=True)
    write_sums(out)


if __name__ == "__main__":
    main()
