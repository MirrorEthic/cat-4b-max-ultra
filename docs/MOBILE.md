# CAT-4B on a phone: the Snapdragon port

**Date started:** 2026-10-28 (Garret 10:35 "Lets start stage 1"). **Target:** Samsung Galaxy Z Fold 6
(Snapdragon 8 Gen 3, 12 GB, arm64-v8a) — the device the T3Story CYOA engine already ships on. **Status:**
stage 1 in progress. This is demo/product work, not paper evidence: the preprint's evidence freeze is
untouched, and nothing here enters the paper.

## Stages

1. **CPU chat** (this stage): `cat4b-v0-merged-Q4_K_M.gguf` (2.6 GB, quantized 2026-10-28 from the release
   F16 with the pinned llama.cpp b11139) through a cross-compiled llama.cpp, first in `adb shell`, then in
   the T3Story app shell (`~/CVMP/T3_sims/t3-cyoa/android`, ONNX→llama.cpp JNI swap) with the gateway's
   retry/validate loop ported into the app layer.
2. **Hexagon chat:** the pinned checkout ships the experimental ggml-hexagon backend
   (`docs/backend/snapdragon/`, built via `scripts/snapdragon/build.py`, Docker + Qualcomm toolchain
   images). Expect CPU-parity speed but much lower power; ~3.5 GB per HTP session, Q4 just fits.
3. **Image lane v1:** SD1.5-class compiled for the 8 Gen 3 NPU via Qualcomm AI Hub (community build:
   512² in ~7 s fully on NPU). CAT-4B still writes the image prompt — the cats reach the image through
   the tuned model's prompt, same as the live service.
4. **Image lane v2 (stretch):** Z-Image-Turbo Q3/Q4 chunked (LiteRT port proved a 6B DiT on an 8 GB
   phone) or the Qualcomm sd.cpp fork; sequential load/unload with the chat model — the mirrorforge
   broker-lease pattern, on-device.
5. **Parity gate before any public claim:** rerun the frozen 490 (and ideally the button cells) against
   the exact mobile quant, judged under the standard rubric, before "CAT-4B on a phone" is said anywhere
   (`cat4b.eval.parity-gate`). Quantization can move cat persistence itself.

## Stage-1 artifacts

- `mobile/build_android.sh` — cross-compiles llama.cpp b11139 for arm64-v8a (NDK 27.2, CPU backend,
  `GGML_CPU_ARM_ARCH=armv8.7-a+dotprod+i8mm+fp16` for the 8 Gen 3's cores).
- `mobile/push_phone.sh` — pushes binaries + model + the two served system prompts to
  `/data/local/tmp/cat4b`, runs `llama-bench` and one chat turn on the big cores (`taskset f0`).
- Quant smoke (legion, 2026-10-28 10:40): 16 `text_plain` English frozen items through the Q4 at the
  served sampling under stock+cat prompts — 11/16 cat by the strict FELINE regex, ≥14/16 counting the
  named cast (Dr. Mew, Furnando) the regex doesn't list; replies coherent and on-task. A smoke, not the
  parity gate.

## RAM budget (12 GB device)

Q4 LM 2.6 GB + KV (small: hybrid Gated-DeltaNet state) + app ≈ 3.5 GB resident for chat. Image models are
never co-resident: chat pauses, image stage loads/unloads (encoder → DiT → VAE), chat resumes.
