# CAT-4B MAX ULTRA — project instructions
Read `docs/BUILD_PLAN.md` first. Text model only is tuned; the image stack is group-chat-mcp's existing tool loop
(`~/group-chat-mcp/services/imagegen_service.py` → mirage FLUX) and is never modified for this project.
Storage: corpus, checkpoints, adapters on `/media/garret-sutherland/dev-data/cat4b/` (ADR 005 in openclaw-cognitive);
never on legion root. GPU: check `~/CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED` before vLLM or training.
Teacher/judge calls use the Anthropic SDK (`claude-opus-5`, batches) in standalone scripts under `data/` and `eval/`.
