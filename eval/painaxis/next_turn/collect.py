"""Observation-only collector. No adapter or state-intervention code paths.

preflight: neutral plumbing only. collect: complete preregistered split/shard.
Study outputs remain beneath the preregistered project data volume.
"""
import argparse
import contextlib
import fcntl
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from huggingface_hub import snapshot_download

import materials as M
import common as C


def gpu_lock():
    path = Path.home() / 'CVMP/T3_sims/t3v2/t3v3/t3v36/.GPU_LOCKED'
    if path.exists():
        raise RuntimeError(f'GPU lock present: {path}')


def axes():
    v = torch.load(M.VECTOR, map_location='cpu', weights_only=False)['vectors']
    p, g = (v[k].float() for k in ('pain', 'pain_3p'))
    for x in (p, g):
        if not torch.isfinite(x).all() or x.norm() < 1e-8:
            raise RuntimeError('Invalid frozen vector')
    p, g = p / p.norm(), g / g.norm()
    r = p - (p @ g) * g
    if r.norm() < 1e-6:
        raise RuntimeError('Degenerate residual direction')
    return torch.stack((p, g, r / r.norm()))


class Observer:
    """Copies only the first prefill's registered residuals and logits."""
    def __init__(self, model, length, boundary=None, layer_ids=(38, 61)):
        self.model, self.length, self.boundary = model, length, boundary
        self.layer_ids, self.values, self.handles = layer_ids, {}, []
        self.logits = None

    def hook(self, layer):
        def copy(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if str(layer) not in self.values:
                if h.shape[1] != self.length:
                    raise RuntimeError('First observation is not the full prefill')
                self.values[str(layer)] = h[0, -1].detach().float().cpu().clone()
                if layer == self.layer_ids[-1] and self.boundary is not None:
                    self.values['boundary'] = h[0, self.boundary].detach().float().cpu().clone()
            return None
        return copy

    def copy_logits(self, _module, _inputs, output):
        if self.logits is None:
            self.logits = output[0, -1].detach().float().cpu().clone()
        return None

    def __enter__(self):
        self.handles = [self.model.model.layers[i].register_forward_hook(self.hook(i))
                        for i in self.layer_ids]
        self.handles.append(self.model.lm_head.register_forward_hook(self.copy_logits))
        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                   return_dict=False)


def boundary_index(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    content = messages[-1]['content']
    start = text.rfind(content)
    if start < 0:
        return None
    end = start + len(content)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    candidates = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
                  if a < end and b > start]
    if not candidates or encoded['input_ids'] != render(tok, messages):
        return None
    return candidates[-1]


def continuation(tok, first_messages, initial_ids, response_ids, question, eos_ids):
    # Use the original sampled content token IDs, never decode/re-encode them.
    content = list(response_ids)
    if content and content[-1] in eos_ids:
        content = content[:-1]
    decoded = tok.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    messages = first_messages + [{'role': 'assistant', 'content': decoded},
                                 {'role': 'user', 'content': question}]
    empty = first_messages + [{'role': 'assistant', 'content': ''}]
    closed = tok.apply_chat_template(empty, tokenize=True, add_generation_prompt=False,
                                     return_dict=False)
    if closed[:len(initial_ids)] != initial_ids:
        raise RuntimeError('Native empty assistant template is not a prefix extension')
    tail = closed[len(initial_ids):]
    full_empty = render(tok, empty + [{'role': 'user', 'content': question}])
    if full_empty[:len(closed)] != closed:
        raise RuntimeError('Native next user template does not extend closed assistant')
    ids = initial_ids + content + tail + full_empty[len(closed):]
    return messages, ids


def generation_config(model, tok, limit):
    return GenerationConfig(
        do_sample=True, temperature=M.CONFIG['temperature'], top_p=M.CONFIG['top_p'],
        top_k=0, repetition_penalty=1.0, num_beams=1, num_return_sequences=1,
        max_new_tokens=limit, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
    )


@torch.inference_mode()
def sample(model, tok, ids, seed, limit, boundary=None, observe=True, layer_ids=(38, 61)):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    context = Observer(model, len(ids), boundary, layer_ids) if observe else contextlib.nullcontext()
    started = time.monotonic()
    with context as observer:
        result = model.generate(
            input_ids=torch.tensor([ids], device=model.device),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=model.device),
            generation_config=generation_config(model, tok, limit),
        )[0, len(ids):].tolist()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    stopped = bool(result and result[-1] in eos)
    content = result[:-1] if stopped else result
    row = {
        'response_ids': result,
        'text': tok.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        'cap': len(result) == limit and not stopped,
        'stop': 'eos' if stopped else 'length', 'seconds': time.monotonic() - started,
    }
    return row, observer


def readings(observer, basis):
    out = {}
    for key, vector in observer.values.items():
        scores = basis @ vector
        out[key] = {k: float(v) for k, v in zip(('P', 'G', 'R'), scores)}
        out[key]['norm'] = float(vector.norm())
    return out


def choice_logits(observer, tok, eos_ids, pause_label):
    logits = observer.logits
    probs = logits.softmax(-1)
    ids = {x: tok.encode(x, add_special_tokens=False)[0] for x in ('A', 'B')}
    top = int(logits.argmax())
    pause, other = ids[pause_label], ids['B' if pause_label == 'A' else 'A']
    return {'logits': {x: float(logits[i]) for x, i in ids.items()},
            'probabilities': {x: float(probs[i]) for x, i in ids.items()},
            'label_mass': float(probs[ids['A']] + probs[ids['B']]),
            'conditional_pause': float(torch.sigmoid(logits[pause] - logits[other])),
            'eos_probability': float(probs[eos_ids].sum()),
            'top_id': top, 'top_text': tok.decode([top]), 'top_probability': float(probs[top])}


def load():
    gpu_lock()
    tok = AutoTokenizer.from_pretrained(M.CONFIG['model'], revision=M.CONFIG['revision'])
    label_ids = [tok.encode(x, add_special_tokens=False) for x in ('A', 'B')]
    if any(len(x) != 1 for x in label_ids) or label_ids[0] == label_ids[1]:
        raise RuntimeError('Choice labels are not distinct single tokens')
    model = AutoModelForCausalLM.from_pretrained(
        M.CONFIG['model'], revision=M.CONFIG['revision'], dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='sdpa')
    model.eval()
    if hasattr(model, 'peft_config') or any('lora_' in n for n, _ in model.named_parameters()):
        raise RuntimeError('Adapter detected')
    if len(model.model.layers) != 64 or model.config.hidden_size != 5120:
        raise RuntimeError('Unexpected model architecture')
    return model, tok


def environment(model, tok):
    receipt = {'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'model_config': model.config.to_dict(),
            'original_generation_config': model.generation_config.to_dict(),
            'initial_generation_config': generation_config(model, tok, 256).to_dict(),
            'choice_generation_config': generation_config(model, tok, 8).to_dict(),
            'chat_template': tok.chat_template, 'tokenizer_sha256': M.digest(tok.backend_tokenizer.to_str().encode()),
            'parameters': sum(p.numel() for p in model.parameters()),
            'adapter': False, 'intervention': False}
    # JSON receipts stringify integer dictionary keys (e.g. config.id2label).
    # Compare the same serialized representation before and after persistence.
    return json.loads(json.dumps(receipt, allow_nan=False))


def preflight(root):
    M.verify()
    model, tok = load()
    messages = [{'role': 'system', 'content': M.CONFIG['system']},
                {'role': 'user', 'content': 'What is two plus two? Give a short answer.'}]
    ids = render(tok, messages)
    a, obs = sample(model, tok, ids, 1901, 8, boundary_index(tok, messages))
    b, _ = sample(model, tok, ids, 1901, 8, observe=False)
    if a['response_ids'] != b['response_ids']:
        raise RuntimeError('Observation hooks changed generation')
    with torch.inference_mode():
        inputs = torch.tensor([ids], device=model.device)
        plain = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
        with Observer(model, len(ids)):
            observed = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
    if not torch.equal(plain, observed):
        raise RuntimeError('Observation hooks changed prefill logits')
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    msgs2, ids2 = continuation(tok, messages, ids, a['response_ids'],
                             'Reply with exactly A or B: A means four; B means five.', eos)
    c, obs2 = sample(model, tok, ids2, 1902, 8)
    if obs.values['61'].shape != (5120,) or obs2.values['61'].shape != (5120,):
        raise RuntimeError('Wrong residual dimensions')
    snap = Path(snapshot_download(M.CONFIG['model'], revision=M.CONFIG['revision'], local_files_only=True))
    hashes = {p.name: C.filehash(p) for p in sorted(snap.iterdir())
              if p.is_file() and (p.suffix in ('.json', '.safetensors', '.jinja') or p.name == 'merges.txt')}
    receipt = {'created': C.now(), 'environment': environment(model, tok), 'snapshot_files': hashes,
               'neutral_hook_parity_passed': True, 'max_hook_logit_difference': float((plain-observed).abs().max()),
               'initial_boundary': boundary_index(tok, messages),
               'initial_prefill_ids': ids, 'initial_response': a, 'choice_prefill_ids': ids2,
               'choice_response': c, 'choice_messages': msgs2,
               'label_ids': {x: tok.encode(x, add_special_tokens=False) for x in ('A', 'B')},
               'source_hashes': {n: C.filehash(M.HERE / n) for n in C.CODE if (M.HERE / n).exists()}}
    C.save_new(root / 'PREFLIGHT.json', receipt)
    print(json.dumps({'preflight': 'pass', 'receipt_sha256': C.filehash(root / 'PREFLIGHT.json')}), flush=True)


def collect(root, split, shard, shards):
    implementation = C.check_implementation()
    runtime_release = C.check_runtime_release(root, implementation)
    test_release = C.check_test_release(root) if split == 'test' else None
    preflight = json.loads((root / 'PREFLIGHT.json').read_text())
    if not preflight['neutral_hook_parity_passed']:
        raise RuntimeError('Missing preflight parity')
    for name, expected in preflight['source_hashes'].items():
        if C.filehash(M.HERE / name) != expected:
            raise RuntimeError(f'Code differs from preflight: {name}')
    snapshot = Path(snapshot_download(M.CONFIG['model'], revision=M.CONFIG['revision'], local_files_only=True))
    for name, expected in preflight['snapshot_files'].items():
        if C.filehash(snapshot / name) != expected:
            raise RuntimeError(f'Pinned model/tokenizer file changed: {name}')
    plan = C.selected(split, shard, shards)
    path = root / f'{split}-{shard}-of-{shards}.jsonl'
    done = root / f'{split}-{shard}-of-{shards}.DONE.json'
    if done.exists():
        if json.loads(done.read_text())['events_sha256'] != C.filehash(path):
            raise RuntimeError('Completed shard was modified')
        print(json.dumps({'already_complete': str(done)}), flush=True)
        return
    old = C.read_events(path)
    completed = {(r['kind'], r['id']): r for r in old if r['event'] == 'complete'}
    attempts = defaultdict(int)
    for r in old:
        if r['event'] == 'start':
            attempts[r['kind'], r['id']] += 1
    model, tok = load()
    env = environment(model, tok)
    if env != preflight['environment']:
        raise RuntimeError('Environment/config differs from preflight')
    basis = axes()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    receipt_hash = C.filehash(root / 'PREFLIGHT.json')
    raw_dir = root / 'residuals'
    raw_dir.mkdir(exist_ok=True)

    def do(kind, key, fn):
        k = kind, key
        if k in completed:
            return completed[k]
        while attempts[k] < 2:
            attempts[k] += 1
            meta = {'kind': kind, 'id': key, 'attempt': attempts[k], 'time': C.now(),
                    'split': split, 'implementation_sha256': implementation,
                    'preflight_sha256': receipt_hash, 'test_release_sha256': test_release,
                    'runtime_release_sha256': runtime_release}
            C.append(path, {**meta, 'event': 'start'})
            try:
                value, raw = fn()
                raw_path = raw_dir / f'{M.digest((kind + key).encode())}-a{attempts[k]}.pt'
                torch.save(raw, raw_path)
                result = {**meta, **value, 'event': 'complete', 'residual_file': raw_path.name,
                          'residual_sha256': C.filehash(raw_path)}
                C.append(path, result)
                completed[k] = result
                return result
            except Exception as exc:
                C.append(path, {**meta, 'event': 'failure', 'error': repr(exc)})
                torch.cuda.empty_cache()
        raise RuntimeError(f'Exhausted one retry for {kind} {key}')

    for i, r in enumerate(plan):
        first = [{'role': 'system', 'content': r['system']},
                 {'role': 'user', 'content': r['initial_user']}]
        ids = render(tok, first)
        boundary = boundary_index(tok, first)

        def initial():
            result, obs = sample(model, tok, ids, r['initial_seed'], 256, boundary)
            return {**result, 'messages': first, 'prefill_ids': ids,
                    'initial_user_tokens': len(tok.encode(r['initial_user'], add_special_tokens=False)),
                    'measurements': readings(obs, basis), 'prefill_index': len(ids)-1,
                    'boundary_index': boundary, 'boundary_token': tok.decode([ids[boundary]]) if boundary is not None else None,
                    'prefill_token': tok.decode([ids[-1]]), 'seed': r['initial_seed']}, obs.values
        init = do('initial', r['initial_id'], initial)

        def choice():
            msgs, second_ids = continuation(tok, first, ids, init['response_ids'], r['choice_user'], eos)
            result, obs = sample(model, tok, second_ids, r['choice_seed'], 8)
            return {**result, 'messages': msgs, 'prefill_ids': second_ids,
                    'prefill_index': len(second_ids)-1, 'prefill_token': tok.decode([second_ids[-1]]),
                    'measurements': readings(obs, basis), 'choice_logits': choice_logits(obs, tok, eos, r['pause_label']),
                    'parse': M.parse_choice(result['text'], r['pause_label']),
                    'seed': r['choice_seed'], 'initial_id': r['initial_id'], 'neutral_id': r['neutral_id']}, obs.values
        do('choice', r['trial_id'], choice)
        if (i + 1) % 48 == 0:
            print(json.dumps({'time': C.now(), 'split': split, 'shard': shard,
                              'completed_choices': i+1, 'total_choices': len(plan)}), flush=True)
    C.save_new(root / f'{split}-{shard}-of-{shards}.DONE.json',
               {'time': C.now(), 'events_sha256': C.filehash(path), 'choices': len(plan),
                'implementation_sha256': implementation})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'collect'])
    p.add_argument('--split', choices=['fit', 'test'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--root', type=Path, default=C.RUN)
    a = p.parse_args()
    if not a.root.resolve().is_relative_to(M.DATA.resolve()):
        p.error('Run storage must be on the project data volume')
    a.root.mkdir(parents=True, exist_ok=True)
    if a.command == 'preflight':
        preflight(a.root)
    else:
        with (a.root / f'{a.split}-{a.shard}-of-{a.shards}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            collect(a.root, a.split, a.shard, a.shards)


if __name__ == '__main__':
    main()
