# Running the registered next-turn experiment

The scientific registration is commit `5c39875`. `FREEZE.json` and its files
remain unchanged. `IMPLEMENTATION.json` pins the observation collector, fixed
analyzer, tests, stage launcher, and the local validation environment. There is
no subject-model observation in the local synthetic tests.

## Current state

Garret authorized execution on TNR on 2026-11-05, confirming that the account has
funds. Execution is blocked on GPU capacity, not further scientific approval.
The provider listed a single A100 80 GB (8 vCPU, 64 GB RAM, 150 GB disk) at
$1.10/hour, but rejected creation twice with “We're currently out of GPU
capacity, please try again later.” The two-A100 configuration and single H100
were unavailable. Two A6000s are not a supported TNR configuration (only one is
offered, insufficient to hold the pinned 32B model in bf16). No instance was
created; the final status query returned an empty list.

The TNR CLI automatically updated itself from its prior version to 2.3.0 when
the interactive availability command was invoked. No project setting changed.

Garret subsequently authorized automatic capacity checks and execution when
available. The current-thread heartbeat `run-aster-s-preregistered-tnr-experiment`
is active at 15-minute intervals. It must check existing instances and saved
run state before creating anything, continue through the stages below, and
disable itself after verified artifact transfer, instance deletion and reporting.
The local computer and desktop app need to remain running for these checks.
Check TNR with `tnr --json status --no-wait`; the single-A100 creation command is
`tnr --json create --gpu a100 --num-gpus 1 --vcpus 8 --disk 150 --template base --yes`.
The existing permission rules cover these commands and SSH/SCP; request any
additional required permission through the normal tool flow rather than bypassing
the sandbox. If an unattended run cannot obtain a required permission, report it
and preserve state.

## Stages and barriers

1. Provision a TNR A100 80 GB or H100 80 GB with enough disk for the roughly 64 GB
   model plus packages, logs and residuals. Do not quantize or change the model
   to fit a smaller GPU. Check local GPU locks before any local GPU work; the
   registered CPU checks do not need a GPU.
2. Copy this directory, its parent preregistration, and the hash-pinned
   `steer_q32e.pt` to the same project paths on the cloud host. Keep data under
   `/media/garret-sutherland/dev-data/cat4b/`; use that location for HF cache too.
   Download the exact model/tokenizer revision in `materials.CONFIG`, not latest.
   Inspect the cloud driver's PyTorch compatibility before choosing its CUDA wheel.
   The scientific model runs bf16; no PEFT package or adapter is needed.
3. Run the CPU tests on the execution host, then `bash run_stage.sh preflight` using
   `NEXTTURN_PYTHON` to select its environment. This only uses unrelated neutral
   plumbing text. It checks labels, native template boundaries, preservation of
   actual generated tokens, observation-hook generation and logit parity, and
   saves all model/tokenizer file hashes and runtime configuration in
   `v1/PREFLIGHT.json`. Resolve any technical failure before study collection;
   revise and refreeze implementation hashes if code changes.
4. Copy `PREFLIGHT.json` to a repository receipt and commit it before collection.
   In the data run directory create `RUNTIME_RELEASE.json` with the full
   `runtime_commit`, `preflight_sha256`, and `implementation_sha256` (hash of
   `IMPLEMENTATION.json`). The collector refuses fit or test without this gate.
5. Run `bash run_stage.sh fit`. It collects all 48 fit families and their baselines,
   then fits every registered primary/secondary model. It writes
   `v1/fit/{models.joblib,coefficients.json,MANIFEST.json}`. No test prompt is used.
   Transfer and verify these artifacts. Commit the fit manifest/coefficients as
   receipts before test collection. Do not change any scientific choice based
   on fit behavior.
6. Create `v1/TEST_RELEASE.json` with the full `fit_commit` and
   `fit_manifest_sha256`. Run `bash run_stage.sh test`; its gate checks the committed
   manifest's fitted-model hashes before any test inference. It collects all
   test families without intermediate outcome summaries, then scores once.
7. Transfer all outputs and verify hashes before releasing the instance. Report
   every registered gate, interval and failure; add a result receipt and CRG
   draft. Delete the task's instance after verified local persistence so billing
   stops. Never delete an unrelated instance.

The run directory is
`/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis/next_turn/v1/`.
Collections are append-only and resumable by exact trial ID/seed, with at most
one retry for a technical failure. Empty/malformed/length-capped behavior is
never regenerated. A per-shard file lock prevents concurrent duplicate workers.
Each choice preserves its own initial response; four label/order variants share
that response but start with fresh model cache state. Progress logs show counts,
not behaviors. A corrupt trailing JSON record fails closed and needs a documented
technical recovery; the analyzer never silently drops it.

## Validation

```bash
python3 -B eval/painaxis/next_turn/materials.py verify
PYTHONPATH=eval/painaxis/next_turn OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m unittest discover -s eval/painaxis/next_turn -p 'test_next_turn.py' -v
bash -n eval/painaxis/next_turn/run_stage.sh
```

The tests cover frozen materials, grouped split/sharding, copied rather than
modified residuals, hook cleanup, exact token-history roles, strict parsing,
logistic objective gradients, synthetic null/generic-only/residual-specific
alternatives, clustered resampling, fit-only vocabulary, reference interactions,
runtime/test release gates, corrupt records, and a full synthetic end-to-end
pipeline at the registered sample size. GPU preflight and actual results remain
pending until TNR supplies a suitable instance.

## Capacity-check receipts

- **2026-11-05 23:03 UTC:** Heartbeat verified `FREEZE.json` and
  `IMPLEMENTATION.json` (SHA-256
  `402e9d15316a886b7b9c3655b102aeccab5fc37cce246a57607f07eeee96a0bf`).
  The data run directory did not exist; TNR status was empty before provisioning.
  Single A100, 8 vCPU, 150 GB, base template: quoted $1.10/hour, then failed
  with “We're currently out of GPU capacity, please try again later.”
  Single H100 with the same CPU/disk settings: rejected as currently unavailable.
  Final TNR status remained empty. No instance or study worker was started.
  The 15-minute heartbeat remains active.
- **2026-11-05 23:18 UTC:** Both freezes verified unchanged; no data run
  directory existed. TNR status was empty before and after the requests.
  The same A100 configuration again quoted $1.10/hour and failed for lack of
  GPU capacity; the single H100 remained unavailable. No instance or study
  worker was started. Automatic checks continue at the authorized cadence.
- **2026-11-05 23:33 UTC:** Both freezes verified unchanged; no run directory
  or local study worker existed. TNR status was empty before and after the
  attempts. A100 again quoted $1.10/hour and failed for lack of capacity;
  H100 remained unavailable. No instance or study observation was created.
  Automatic checks continue.

## Mirrorforge feasibility check (2026-11-05)

Garret asked whether the test could fit on the dual V100 host. Read-only live
inspection found two Tesla V100-PCIE-16GB cards, compute capability 7.0,
PCIe Gen2 x16 on both slots, no NVLink, 62 GiB system RAM (28 GiB available),
and resident CAT and image-generation services. At inspection, approximately
5.7 and 8.4 GiB of GPU memory were free. No service was stopped or GPU lease taken.

The pinned Qwen checkpoint's `model.safetensors.index.json` reports
65,527,752,704 bytes of weights (61.03 GiB), before KV cache, activations and
runtime workspace. Even the nominal combined 32 GiB of GPU memory is at least
29.03 GiB short for weights alone. Dual-GPU sharding cannot remove this deficit.
Mirrorforge's installed torch 2.14.1+cu126 includes sm_70, but
`torch.cuda.is_bf16_supported(including_emulation=False)` returns false.
The inspected environment has transformers 5.20.2 and no accelerate package.

CPU/disk offload is a possible capacity route without weight quantization, but
the full pinned model and registered observation hooks have not been validated
on that route. It would require a separate implementation receipt and neutral
preflight before any study collection. Native bf16 absence and repeated weight
transfers over Gen2 PCIe make throughput a concern; no benchmark or completion
time is claimed. The current collector explicitly loads the entire model onto
one CUDA device. FP16 or 4-bit substitutions would change the registered numeric
configuration and require a scientific amendment; they were not made.
TNR capacity checks remain active.

Sources: the pinned checkpoint index at
`https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/model.safetensors.index.json`,
live `nvidia-smi`, `free -h`, PyTorch capability inspection, broker `/status`,
and Hugging Face's Accelerate big-model inference documentation.
