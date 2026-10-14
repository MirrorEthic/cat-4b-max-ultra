# Serving receipts

## mamba cache blocks

First `serve/vllm.sh start` with the pilot adapter, 2026-10-13 11:36:43 MDT, vLLM 0.31.0 (`venvs/serve-vllm`),
`--gpu-memory-utilization 0.6`, no `--max-num-seqs` (default 256). `logs/vllm.log` line 239:

```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (240). Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed. Please lower max_num_seqs to at most 240 or increase gpu_memory_utilization.
```

Fix `24a5ba3` (11:39): `--max-num-seqs 32` (`MAX_SEQS`), after which the server came up with both model names
(`qwen3.5-4b-vl`, `cat4b-vl`).
