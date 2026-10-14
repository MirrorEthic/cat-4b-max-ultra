# CAT-4B MAX ULTRA — project instructions
Read `docs/BUILD_PLAN.md` first. Text model only is tuned; the image stack is group-chat-mcp's existing tool loop
(`~/group-chat-mcp/services/imagegen_service.py` → mirage FLUX) and is never modified for this project.
Storage: corpus, checkpoints, adapters on `/media/garret-sutherland/dev-data/cat4b/` (ADR 005 in openclaw-cognitive);
never on legion root. GPU: check `~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED` before vLLM or training.
Generation is local-first: teacher = Qwen3.8-27B on the 5090 (`serve/llama_local.sh teacher`, :8290); production
judge = Gemma-4-31B on mirrorforge (`serve/mirrorforge_llama.sh start judge`, 10.10.10.2:8291, holds the GPU broker's
lease, so mirage FLUX is down while it runs). Claude (`claude-opus-5`, batches) is the reference judge the local one is
calibrated against, not the bulk path. Why each piece is what it is: `crg/CURRENT.md` (causal research graph; drafts in
`crg/drafts/`, adopted by Garret).
