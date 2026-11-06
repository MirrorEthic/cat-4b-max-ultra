# Running the registered next-turn experiment

The scientific registration is commit `5c39875`. `FREEZE.json` and its files
remain unchanged. `IMPLEMENTATION.json` pins the observation collector, fixed
analyzer, tests, stage launcher, and the local validation environment. There is
no subject-model observation in the local synthetic tests.

## Current state

**Study collection confirmed, 2026-11-06 01:26 UTC:** runtime receipt commit
`3620b7b17752a4be186852df58cee242ee6b2ef3` preceded the first study trial at
01:25:34 UTC. Runtime release SHA-256:
`99cebca59ac598cfbe45405a67fc9a2b50400bd5ff906b3831efe090def52bfd`.
The current fit worker is PID `1797`, started at 01:24:16 UTC. At 01:26:48 UTC,
23 initial replies and 92 of 9,216 fit choices were complete, with no recorded
technical failure. These are progress counts only; no behavior summary was
inspected. No test-family inference has occurred and no test release exists.
Continue this worker, then transfer/hash-verify the complete fit artifacts and
commit the manifest before held-out release. The $30 ceiling, original compute
stop deadline and $3 recovery reserve remain unchanged. Routine monitoring is
active and muted; disable it only after results and verified cleanup.

**Revalidated runtime, 2026-11-06 01:22 UTC:** implementation commit `0c1840d`
has receipt SHA-256
`c426373f5840cce3167cc4c2565a71d672cacf93bc2fa2085a4978212a9e4046`.
All 17 remote tests passed in 22.321 seconds. Neutral GPU preflight again passed
with bitwise hook parity. The new environment receipt and all model/tokenizer
file hashes equal the previous successful preflight exactly; only implementation
serialization changed. Revised receipts live in `receipts/runpod-20261106-r2/`.
Their preflight SHA-256 is
`2fef74bd5589473c1bf91c410f8539dedc6912b6184dc5beab70636d342f37cb`.
Commit these before creating a new runtime release and relaunching fit. There
are still zero study trial starts; all prior gates/failures remain preserved.

**Gate correction, 2026-11-06 01:18 UTC:** the first fit launch stopped before
any trial start because JSON had stringified `model_config.id2label` keys while
the fresh Python dictionary retained integer keys. A neutral diagnostic showed
these two keys were the entire mismatch; JSON-normalized runtimes were equal.
Environment receipts now use the same JSON representation before and after
persistence. A new CPU regression verifies equality after persistence and
inequality after a real configuration change; all 17 local tests passed in
14.156 seconds. No study observations exist. The original successful preflight
and runtime release remain in `v1/validation-history/runtime-d2bb9d0/`, the first
fit-launch logs in `setup/fit-launch-attempt-1/`, and runtime commit `d2bb9d0`
remains historical. A revised implementation and new neutral runtime release are
required before relaunching fit. The registration/materials are unchanged.

**Historical launch, 2026-11-06 01:13 UTC:** neutral preflight was complete and the runtime
receipt was committed at `d2bb9d0357b3fbe2f961fdb17496606abd835fec` before collection.
`RUNTIME_RELEASE.json` verified locally and remotely. The fixed fit stage was
launched once, PID `1472`, and exited at the runtime gate. No held-out release exists. Inspect
`ops/STATE.json`, remote `ops/fit.{pid,started,log,exit}`, and `ops/progress.json`
before resuming; a missing exit file while the PID lives means continue monitoring.
The runtime release SHA-256 is
`a863ffb7a61e74e70452e6dc350e41b021851f6ba18e05a2c7542abda7e11322`.

**RunPod pivot, 2026-11-05 17:48 MST:** Garret authorized moving the experiment
to RunPod with a $30 total task spend
limit, including storage and setup. Prefer one on-demand A100 80 GB; inspect the
actual hourly quote before creating it. Use 150 GB persistent volume storage and
a 20 GB container disk, SSH access, and a verified official PyTorch template.
Do not create a second pod if the first request's outcome is uncertain; list and
reconcile first. Do not resume TNR provisioning.

Authentication succeeded. At **2026-11-06 00:52:43 UTC**, created exactly one pod,
`a3063hrjda5ggu` (`aster-next-turn-v1`), with an A100-SXM4-80GB in US-MD-1.
The verified quote is $1.79/hour compute, approximately $1.813611/hour including
150 GB volume and 20 GB container storage. The experiment's spend ceiling remains $30. Private account balance is omitted. SSH and the actual
81920 MiB GPU were verified. Official template: `runpod-torch-v280`, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

Durable operational receipts live at
`/media/garret-sutherland/dev-data/cat4b/state/eval/painaxis/next_turn/ops/`:
`CREATE_REQUEST.json`, `CREATE_RESPONSE.json`, and `STATE.json`. Read that state
before any action. Retired SSH connection details are retained only in private operational state. Remote code is `/workspace/aster-next-turn`; Python is
`/workspace/cat4b-data/venv/bin/python`. The registered data path is a symlink to
the persistent `/workspace/cat4b-data`. Both freezes verified unchanged there.

Runtime packages are installed. All 15 CPU/synthetic tests passed on the pod
in 22.191 seconds (Python 3.12.3, torch 2.8.0+cu128, transformers 5.17.0).
The frozen model is downloading for neutral preflight, followed by an unrelated
water-cycle throughput check. Preflight PID 328 and benchmark PID 582 have
separate logs, locks and exit receipts under `/workspace/cat4b-data/setup/`.
No study observation has been made.

**Preflight correction, 2026-11-06 01:04 UTC:** the first GPU preflight loaded
the pinned weights but failed before generation because Transformers 5.17
defaults `apply_chat_template` to a `BatchEncoding`, while the collector assumed
a flat token-ID list. The initial synthetic tokenizer mock did not exercise that
API default. Both tokenized template calls now explicitly request
`return_dict=False`; a CPU regression uses the actual Transformers/tokenizers API
to verify token IDs, boundary location and exact continuation. All 16 local tests
passed in 14.663 seconds. `IMPLEMENTATION.json` records the correction and the
previous receipt's hash/commit. `FREEZE.json` and scientific choices are unchanged.
The first preflight logs and original source archive are preserved under the
remote data volume's `setup/preflight-attempt-1/`. Revised remote validation and
neutral preflight must pass and be committed before releasing collection.

The revised implementation is commit `8ceabb7`, receipt SHA-256
`bec34e9cd9a78c00da5bd840683e481deee86133107ee40132e4c26cf9563708`.
Its 16 remote CPU tests passed in 22.398 seconds. Preflight attempt 2 passed
the generation/hook checks but receipt creation failed because HF Hub's strict
offline snapshot check also required `.gitattributes`, `LICENSE`, and `README.md`.
Those non-model files are being fetched at the same pinned revision; no source or
scientific choice changed. Attempt 2 logs remain in `setup/preflight-attempt-2/`.

**Runtime validated, 2026-11-06 01:11 UTC:** attempt 3 passed, with zero logit
difference and identical sampled tokens between hooked and unhooked generation.
The final remote CPU suite passed all 16 tests in 22.541 seconds. Receipts,
package freeze, failed preflight logs and the unrelated benchmark are copied and
hash-verified under `receipts/runpod-20261106/`. The benchmark generated 256
tokens in 13.310 seconds and a six-token next-turn response in 0.398 seconds.
Maximum-length initial replies throughout could exceed $30; estimate actual
duration from collection counts/timing without changing the registered N.
The runtime receipt must be committed before creating `RUNTIME_RELEASE.json`.
Then launch only the fixed fit stage using the registered `run_stage.sh` through
`/workspace/cat4b-data/setup/run-study-stage.sh fit`. Its global stage lock,
PID, start/finish timestamps and exit code live under the remote data path's
`state/eval/painaxis/next_turn/ops/`; send stdout/stderr to `fit.log` there.
The fitted-model release barrier remains mandatory before any held-out inference.
The existing heartbeat now continues RunPod at 15-minute intervals, with routine
notifications muted. No TNR provisioning is permitted. A separate local budget
guard in `ops/budget_guard.py` stops only this pod's compute by
**2026-11-06 15:45:57 UTC**, reserving $3 for recovery/cleanup. Its PID and log
are in the same directory. The guard never deletes the pod or persistent data;
artifact verification and deletion remain the continuation's responsibility.

On RunPod, map the registered data path to persistent volume storage (for example,
a symlink to `/workspace/cat4b-data`); do not put weights or observations on the
ephemeral container filesystem. Preserve the same model/dtype, source freezes,
CPU tests, neutral GPU preflight, runtime commit, fit commit and test release
barriers below. Time unrelated neutral prompts before study collection to refine
the cost estimate. Track elapsed spend from pod creation, including setup;
reserve at least $3 for artifact transfer and cleanup. If completion would exceed
the limit, save and hash-verify partial artifacts locally, then stop the task pod
before exhausting the budget. Do not change sample counts or choose trials based
on outcomes to fit the budget. Report any incomplete run as incomplete.

**Manual provisioning handoff, 2026-11-05 17:42 MST:** Garret is trying the web
console. The heartbeat is temporarily PAUSED to prevent competing creation
attempts. No TNR instance existed at the handoff. When Garret returns, inspect
TNR status and use the instance he created before considering any new allocation.

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
was configured at 15-minute intervals (currently paused for the manual handoff).
When active, it must check existing instances and saved
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

1. Provision the authorized provider's A100 80 GB or H100 80 GB with enough disk for the roughly 64 GB
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
pipeline at the registered sample size. GPU preflight and actual results remain pending on the allocated RunPod instance.

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

- **2026-11-05 23:48 UTC:** Both freezes verified unchanged; no run directory
  or local study worker existed. TNR status was empty before and after the
  attempts. A100 again quoted $1.10/hour and failed for lack of capacity;
  H100 remained unavailable. No instance or study observation was created.
  Automatic checks continue.

- **2026-11-06 00:03 UTC:** Both freezes verified unchanged; no run directory
  or local study worker existed. Initial A100 request failed while fetching
  GPU specifications with a connection reset, before creation. One retry
  reached creation, quoted $1.10/hour and failed for lack of GPU capacity.
  H100 remained unavailable. TNR status was empty before and after the attempts;
  no instance or study observation was created. Automatic checks continue.

- **2026-11-06 00:18 UTC:** Both freezes verified unchanged; no run directory
  or local study worker existed. TNR status was empty before and after the
  attempts. A100 quoted $1.10/hour and failed for lack of GPU capacity;
  H100 remained unavailable. No instance or study observation was created.
  Automatic checks continue.

- **2026-11-06 00:33 UTC:** Both freezes verified unchanged; no run directory
  or local study worker existed. TNR status was empty before and after the
  attempts. A100 quoted $1.10/hour and failed for lack of GPU capacity;
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
