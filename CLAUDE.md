# CAT-4B MAX ULTRA — project instructions
Read `docs/ARCHITECTURE.md` first: what was built and measured. `docs/BUILD_PLAN.md` and `docs/DATA_AND_TRAINING.md`
are the plans it grew from. Text model only is tuned; the image stack is group-chat-mcp's existing tool loop
(`~/group-chat-mcp/services/imagegen_service.py` → mirage FLUX) and is never modified for this project.
Storage: corpus, checkpoints, adapters on `/media/garret-sutherland/dev-data/cat4b/` (ADR 005 in openclaw-cognitive);
never on legion root. Release adapter: `adapters/cat4b-v0`. GPU: check `~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED`
before vLLM or training.
Generation is local-first: teacher = Qwen3.8-27B on the 5090 (`serve/llama_local.sh teacher`, :8290); production
judge = Gemma-4-31B on mirrorforge (10.10.10.2:8291). `serve/mirrorforge_llama.sh` runs ON mirrorforge:
`ssh mirrorforge 'bash ~/cat4b/mirrorforge_llama.sh start judge'` / `... stop`. The judge holds the GPU broker's
lease, so mirage FLUX is down while it runs; stop it when judging is done. Claude (`claude-opus-5`, batches) is the
reference judge the local one is calibrated against, not the bulk path. Why each piece is what it is:
`crg/CURRENT.md` (causal research graph; drafts in `crg/drafts/`, adopted by Garret).
