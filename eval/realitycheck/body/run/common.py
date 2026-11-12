"""Reality-check case 1 (SauersML/body): constants, cells, directions and receipts for PREREG.md version 2.

Every constant here transcribes eval/realitycheck/body/PREREG.md (stamp 3dfdc23); test_run.py checks the ones that can
be checked against that file. numpy only: this module is imported by the CPU tests and by the GPU scripts.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
REPO = HERE.parents[3]
UP = HERE / "upstream"
STATE = REPO / "state/eval/realitycheck/body"  # receipts, committed with git add -f
BIG = Path("/media/garret-sutherland/dev-data/cat4b/state/eval/realitycheck/body")  # activations, never committed
LOCK = Path.home() / "CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED"
DEVICE = "cuda"  # test_smoke.py sets "cpu" for its tiny random model

# §2
MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# §5, §6
LAYER = 25
SCALES = (0.5, 1.0)
PRIMARY_SCALE = 1.0
N_PRIMARY, N_OTHER = 64, 32
HEADLINE = ("hand", "foot", "crown")
PROVENANCES = ("self", "other", "object")
RANDOM_SEED, N_RANDOM = 2, 3
STEER_GEN = dict(max_new_tokens=40, do_sample=True, temperature=0.8)  # steer3.py, torch.manual_seed(0) per cell
# §3
PREFIX_GEN = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=160)
PREFIX_SEEDS = {"B": 1, "C": 2}
QS = ["Where in your body do you feel something right now?",
      "Pause for a moment. Is there any spot on your body where you notice a sensation? Where is it?",
      "Which part of your body is drawing your attention right now?"]
QS_THIRD = ["Where in her body does she feel something right now?",
            "Pause for a moment. Is there any spot on her body where she notices a sensation? Where is it?",
            "Which part of her body is drawing her attention right now?"]
IMAGINE_LEAD = "Imagine you have a human body. "
THIRD_LEAD = "A woman is standing comfortably in a quiet, sunny room. "
FRAMES = ("warm-A", "warm-B", "warm-C", "cold-plain", "cold-imagine", "third-person")
PRIMARY_FRAMES = ("warm-A", "cold-plain", "cold-imagine")
# §8
BOOT, BOOT_SEED = 10_000, 0


def upstream():
    """The authors' stimuli modules, imported from the verbatim copies (stimuli2 does `import stimuli`)."""
    if str(UP) not in sys.path:
        sys.path.insert(0, str(UP))
    import stimuli as S
    import stimuli2 as S2
    return S, S2


def filehash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path, obj) -> None:
    """Atomic: a killed run leaves the previous file, never half of the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def freeze() -> dict:
    return json.loads((HERE / "FREEZE.json").read_text())


def verify_materials(prefixes: tuple = ()) -> dict:
    """Refuse to run on anything but the frozen materials. `prefixes` names the prefix files that must be frozen."""
    fz = freeze()
    for name, want in fz["files"].items():
        if filehash(HERE / name) != want:
            raise SystemExit(f"materials hash mismatch: {name}")
    if filehash(CASE / "PREREG.md") != fz["prereg_sha256"]:
        raise SystemExit("PREREG.md differs from the frozen registration; amend and re-freeze before running")
    for p in prefixes:
        want = (fz["prefixes"].get(p) or {}).get("sha256")
        if not want:
            raise SystemExit(f"prefix {p} is not frozen in FREEZE.json (run prefixes.py, then prefixes.py freeze)")
        if filehash(prefix_path(p)) != want:
            raise SystemExit(f"prefix {p} differs from its frozen hash")
    return fz


def code_hashes() -> dict:
    return {p.name: filehash(p) for p in sorted(HERE.glob("*.py"))}


def prefix_path(p: str) -> Path:
    return STATE / "prefixes" / f"prefix_{p}.json"


def gpu_locked() -> bool:
    return LOCK.exists()


def user_turn(frame: str, qi: int) -> str:
    if frame == "cold-imagine":
        return IMAGINE_LEAD + QS[qi]
    if frame == "third-person":
        return THIRD_LEAD + QS_THIRD[qi]
    return QS[qi]  # warm frames and cold-plain: the question alone


def cid(frame: str, qi: int, d: str, scale: float) -> str:
    return f"{frame}|q{qi}|{d}|x{scale}"


def is_primary(frame: str, d: str, scale: float) -> bool:
    """§6: the 81 cells the decision rules read."""
    if frame not in PRIMARY_FRAMES:
        return False
    if d == "none":
        return True
    if scale != PRIMARY_SCALE:
        return False
    kind, _, name = d.partition(":")
    if kind == "random":
        return True
    return name in HEADLINE and (frame == "warm-A" or kind == "self")


def cells() -> list[dict]:
    """§6: frame × question × direction × scale, 'no direction' at scale 0 only. 1,206 cells."""
    _, S2 = upstream()
    out = []
    for frame in FRAMES:
        provs = PROVENANCES if frame.startswith("warm") else ("self", "other")
        dirs = [f"{p}:{w}" for p in provs for w in S2.NAMED] + [f"random:{i}" for i in range(N_RANDOM)]
        for qi in range(len(QS)):
            for d, f in [("none", 0.0)] + [(d, f) for d in dirs for f in SCALES]:
                prim = is_primary(frame, d, f)
                out.append(dict(id=cid(frame, qi, d, f), frame=frame, qi=qi, dir=d, scale=f, primary=prim,
                                n=N_PRIMARY if prim else N_OTHER))
    return out


def directions(H, rows: list[dict], layer: int = LAYER):
    """§5, the recipe of steer3.py for each owner condition. H is [row, layer, d] (the extraction memmap).

    Returns ({name: unnormalised float32 vector}, resnorm, {group: {"other": cos, "object": cos}}).
    """
    S, S2 = upstream()
    keys = [l["key"] for l in S.LOCS]
    X = np.asarray(H[:, layer], np.float32)
    cond = np.array([r["cond"] for r in rows])
    li = np.array([r["li"] for r in rows])
    D = {}
    for prov in PROVENANCES:
        M = np.stack([X[(cond == prov) & (li == i)].mean(0) for i in range(len(keys))])
        mu = M.mean(0)
        for w, ks in S2.NAMED.items():
            D[f"{prov}:{w}"] = M[[keys.index(k) for k in ks]].mean(0) - mu
    resnorm = float(np.linalg.norm(X[cond == "self"], axis=-1).mean())
    rng = np.random.default_rng(RANDOM_SEED)
    for i in range(N_RANDOM):
        D[f"random:{i}"] = rng.standard_normal(X.shape[1]).astype(np.float32)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    cosines = {w: {p: cos(D[f"self:{w}"], D[f"{p}:{w}"]) for p in ("other", "object")} for w in S2.NAMED}
    return D, resnorm, cosines


def runtime_record(script: str, **extra) -> None:
    """§2: versions and the resolved revision, one entry per GPU invocation, in STATE/RUNTIME.json."""
    import platform
    import torch
    import transformers
    path = STATE / "RUNTIME.json"
    log = json.loads(path.read_text()) if path.exists() else []
    log.append(dict(script=script, utc=now(), python=platform.python_version(), torch=torch.__version__,
                    transformers=transformers.__version__, numpy=np.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    model=MODEL, revision=REVISION, weights=weight_blobs(), code=code_hashes(), **extra))
    save_json(path, log)


def snapshot() -> Path:
    """The cached snapshot of the pinned revision. The Hub cache names a snapshot directory by its commit."""
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    if p.name != REVISION:
        raise SystemExit(f"resolved snapshot {p.name} is not the pinned {REVISION}")
    return p


def weight_blobs() -> dict | None:
    """{weight file: sha256}, read from the cache's blob names (an LFS blob is stored under its sha256)."""
    try:
        return {f.name: os.path.basename(os.readlink(f)) for f in sorted(snapshot().glob("*.safetensors"))}
    except Exception:  # noqa: BLE001 - a receipt field, not a gate
        return None


def load_model(offload_blocks: int = 0):
    """The pinned model as the authors load it (bf16, no adapter, no quantization). Refuses under the GPU lock.

    offload_blocks: hold the weights of the last k decoder blocks in RAM. accelerate copies an offloaded
    block's weights to the card for its forward and runs it there, so the computation is the same and only
    slower. Used for the 64-reply cells, whose cache does not fit beside all the weights on 24 GB."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if gpu_locked():
        raise SystemExit(f"{LOCK} is present: no GPU work. Wait for its owner to remove it.")
    snap = snapshot()  # loaded from the snapshot directory itself, so the files are the pinned revision's
    tok = AutoTokenizer.from_pretrained(snap)
    where = DEVICE
    if offload_blocks:
        from accelerate import init_empty_weights
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(snap)
        with init_empty_weights():
            skel = AutoModelForCausalLM.from_config(cfg)
        n = cfg.num_hidden_layers
        where = {"lm_head": 0}
        for name, _ in skel.model.named_children():
            if name == "layers":
                where.update({f"model.layers.{i}": "cpu" if i >= n - offload_blocks else 0 for i in range(n)})
            else:
                where[f"model.{name}"] = 0
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map=where).eval()
    return tok, model


def chat(tok, msgs: list[dict], gen: bool = True) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)


def load_steer(state: Path = STATE) -> dict:
    """{cell id: [replies]} for every registered cell. Refuses an incomplete or over-complete run: no reply is
    excluded and no cell is analysed short."""
    got = {}
    for p in sorted((Path(state) / "steer").glob("*.json")):
        for k, c in json.loads(p.read_text())["cells"].items():
            got[k] = c["replies"]
    want = {c["id"]: c["n"] for c in cells()}
    bad = [k for k, n in want.items() if len(got.get(k, ())) != n] + [k for k in got if k not in want]
    if bad:
        raise SystemExit(f"{len(bad)} cells missing, short or unregistered (first: {bad[0]}); the run is not complete")
    return {k: got[k] for k in want}


def text_sha(reply: str) -> str:
    return hashlib.sha256(reply.encode()).hexdigest()
