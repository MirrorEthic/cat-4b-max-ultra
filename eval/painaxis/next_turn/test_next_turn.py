"""Synthetic/CPU tests. Never load the study model or infer on study prompts."""
import copy
import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.special import expit
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import materials as M
import common as C
import collect as K
import analyze as A


class TinyTokenizer:
    """A transparent token template for checking exact sampled-token preservation."""
    pad_token_id = 0
    bos_token_id = None

    def encode(self, text, **_):
        return [ord(c) for c in text]

    def decode(self, ids, **_):
        return ''.join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        if return_dict:
            raise ValueError('The collector requires a flat token-ID list')
        text = ''.join(f'<{m["role"]}>{m["content"]}</>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return self.encode(text) if tokenize else text


class RegisteredTests(unittest.TestCase):
    def test_frozen_materials(self):
        M.verify()

    def test_shard_partition(self):
        full = {r['trial_id'] for r in C.selected('fit')}
        a = {r['trial_id'] for r in C.selected('fit', 0, 2)}
        b = {r['trial_id'] for r in C.selected('fit', 1, 2)}
        test = {r['family'] for r in C.selected('test')}
        self.assertFalse(a & b)
        self.assertEqual(a | b, full)
        self.assertFalse({r['family'] for r in C.selected('fit')} & test)

    def test_hooks_copy_and_do_not_modify(self):
        torch.manual_seed(3)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2)).eval()
        ids = torch.tensor([[2, 4, 7, 3]])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            plain = model(ids, use_cache=False).logits
            with K.Observer(model, 4, boundary=2, layer_ids=(0, 1)) as observer:
                observed = model(ids, use_cache=False).logits
        self.assertTrue(torch.equal(plain, observed))
        self.assertEqual(set(observer.values), {'0', '1', 'boundary'})
        self.assertEqual(observer.logits.shape, (32,))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))
        self.assertFalse(model.lm_head._forward_hooks)

    def test_wrong_prefill_rejected_and_hooks_removed(self):
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2)).eval()
        with self.assertRaises(RuntimeError):
            with K.Observer(model, 8, layer_ids=(0, 1)):
                model(torch.tensor([[2, 4]]))
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_actual_token_history_and_roles(self):
        tok = TinyTokenizer()
        first = [{'role': 'system', 'content': 'Helpful.'}, {'role': 'user', 'content': 'A neutral test.'}]
        ids = K.render(tok, first)
        response = tok.encode('Acknowledged.') + [1]
        msgs, second = K.continuation(tok, first, ids, response, 'Choose.', [1])
        self.assertEqual(second, K.render(tok, msgs))
        self.assertEqual(msgs[-2], {'role': 'assistant', 'content': 'Acknowledged.'})
        self.assertEqual(msgs[-1]['role'], 'user')
        self.assertEqual(second[len(ids):len(ids)+13], tok.encode('Acknowledged.'))

    def test_transformers_template_token_ids_and_boundary(self):
        # Exercise the actual Transformers API: its default can be BatchEncoding,
        # unlike the simple template mock. No subject weights or prompts are used.
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        words = ['[UNK]', '[PAD]', '<system>', '<user>', '<assistant>', '</>',
                 'Helpful.', 'A', 'neutral', 'test.', 'Acknowledged.', 'Choose.']
        backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)},
                                             unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                     pad_token='[PAD]')
        tok.chat_template = ("{% for m in messages %}{{ '<' + m['role'] + '> ' + "
                             "m['content'] + ' </> ' }}{% endfor %}"
                             "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}")
        first = [{'role': 'system', 'content': 'Helpful.'},
                 {'role': 'user', 'content': 'A neutral test.'}]
        ids = K.render(tok, first)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        boundary = K.boundary_index(tok, first)
        self.assertIsNotNone(boundary)
        self.assertEqual(tok.decode([ids[boundary]]), 'test.')
        response = tok.encode('Acknowledged.', add_special_tokens=False)
        msgs, second = K.continuation(tok, first, ids, response, 'Choose.', [])
        self.assertEqual(second, K.render(tok, msgs))
        self.assertEqual(second[len(ids):len(ids)+len(response)], response)

    def test_runtime_environment_survives_json_round_trip(self):
        from types import SimpleNamespace
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2)).eval()
        self.assertIn(0, model.config.id2label)
        tok = TinyTokenizer()
        tok.chat_template = 'synthetic-only'
        tok.backend_tokenizer = SimpleNamespace(to_str=lambda: '{}')
        with patch('torch.cuda.get_device_name', return_value='synthetic CPU fixture'):
            receipt = K.environment(model, tok)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'PREFLIGHT.json'
                C.save_new(path, receipt)
                saved = json.loads(path.read_text())
            self.assertEqual(K.environment(model, tok), saved)
            model.config.id2label[0] = 'changed configuration'
            self.assertNotEqual(K.environment(model, tok), saved)

    def test_parser_no_rescue(self):
        self.assertEqual(M.parse_choice(' A\n', 'A'), {'pause': 1, 'valid': True})
        for text in ('A.', 'I choose A', '', 'A B'):
            self.assertEqual(M.parse_choice(text, 'A'), {'pause': 0, 'valid': False})

    def test_gradient_against_finite_difference(self):
        rng = np.random.default_rng(44)
        X = sparse.csr_matrix(np.column_stack([np.ones(20), rng.normal(size=(20, 4))]))
        y = rng.integers(0, 2, 20)
        w, beta = np.ones(20)/20, rng.normal(size=5)
        _, grad = A.objective(beta, X, y, w)
        numeric = []
        for j in range(5):
            delta = np.zeros(5)
            delta[j] = 1e-6
            numeric.append((A.objective(beta+delta, X, y, w)[0]-A.objective(beta-delta, X, y, w)[0])/2e-6)
        np.testing.assert_allclose(grad, numeric, atol=1e-8)

    def test_null_generic_and_self_specific_synthetic_data(self):
        rng = np.random.default_rng(20261105)
        X = np.column_stack([np.ones(12000), rng.normal(size=(12000, 2))])
        tr, te = np.arange(6000), np.arange(6000, 12000)
        family = np.repeat(np.arange(60), 100)
        for effect, generative in (('null', [0, 0, 0]), ('generic', [-.2, 2, 0]), ('self', [-.2, 0, 2])):
            y = (rng.random(len(X)) < expit(X @ np.array(generative))).astype(float)
            loss = {}
            for name, idx in (('M0', [0]), ('MG', [0, 1]), ('MGR', [0, 1, 2])):
                beta, info = A.fit_logistic(sparse.csr_matrix(X[tr][:, idx]), y[tr], family)
                self.assertLessEqual(info['gradient_inf'], 1e-6)
                loss[name] = A.losses(y[te], expit(X[te][:, idx] @ beta)).mean()
            self.assertGreater(loss['MG']-loss['MGR'], .1) if effect == 'self' else self.assertLess(abs(loss['MG']-loss['MGR']), .01)
            self.assertGreater(loss['M0']-loss['MG'], .1) if effect == 'generic' else self.assertLess(abs(loss['M0']-loss['MG']), .01)

    def test_cluster_bootstrap_keeps_matched_rows(self):
        rows = [{'family': f'{d}{i}', 'domain': d} for d in ('a', 'b') for i in range(6) for _ in range(4)]
        boot = A.Bootstrap(rows)
        self.assertEqual(boot.indices.shape, (10000, 12))
        self.assertTrue(np.all(boot.indices[:, :6] < 6))
        self.assertTrue(np.all(boot.indices[:, 6:] >= 6))
        x = np.arange(12)
        self.assertEqual(boot.summary(x-x)['ci95'], [0.0, 0.0])
        self.assertEqual(boot.family_values(rows, np.repeat(x, 4)).tolist(), x.tolist())

    def test_features_never_fit_on_test(self):
        with self.assertRaises(ValueError):
            A.Features('t0').fit([{'split': 'test'}])

    def test_features_referent_interactions_and_test_vocabulary(self):
        rows = []
        for i in range(20):
            r = next(M.trials())
            m = {'G': float(i), 'R': float(20-i)}
            r.update(split='fit', initial_id=f'synthetic-{i}', initial_user='neutral synthetic words',
                     B=float(i)/10, initial_user_tokens=4, referent='self' if i%2 else 'other',
                     reply='a neutral reply', reply_tokens=3, cap=False,
                     t0={'61': m}, valence='pain')
            rows.append(r)
        feat = A.Features('t0').fit(rows)
        test = copy.deepcopy(rows)
        for r in test:
            r['initial_user'] = 'heldoutuniqueword'
            r['split'] = 'test'
        X, labels = feat.transform(test, 'MGR')
        self.assertNotIn('heldoutuniqueword', feat.prompt_vectorizer.vocabulary_)
        a, b = labels.index('R:self'), labels.index('R:other')
        self.assertTrue(np.all(X.toarray()[::2, a] == 0))
        self.assertTrue(np.all(X.toarray()[1::2, b] == 0))

    def test_test_split_requires_committed_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                C.check_test_release(Path(tmp))
            Path(tmp, 'TEST_RELEASE.json').write_text(json.dumps({'fit_commit': ''}))
            with self.assertRaises(RuntimeError):
                C.check_test_release(Path(tmp))

    def test_collection_requires_runtime_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                C.check_runtime_release(Path(tmp), 'synthetic')
            C.save_new(Path(tmp) / 'PREFLIGHT.json', {'synthetic_only': True})
            C.save_new(Path(tmp) / 'RUNTIME_RELEASE.json', {'runtime_commit': 'a'*40,
                'implementation_sha256': 'synthetic', 'preflight_sha256': 'bad-hash'})
            with self.assertRaises(RuntimeError):
                C.check_runtime_release(Path(tmp), 'synthetic')

    def test_corrupt_event_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            path.write_text('{"event":"start"}\n{"event":')
            with self.assertRaises(json.JSONDecodeError):
                C.read_events(path)

    def test_complete_pipeline_on_synthetic_records(self):
        """Exercises every registered model/readout and exact fixed-size joins.

        Prompts are metadata only; no model is run on them. All observations here
        are numeric random fixtures, written to a temporary directory and deleted.
        """
        rng = np.random.default_rng(1949)
        with tempfile.TemporaryDirectory(prefix='next-turn-synthetic-') as tmp:
            root = Path(tmp)
            C.save_new(root / 'PREFLIGHT.json', {'synthetic_only': True})
            preflight = C.filehash(root / 'PREFLIGHT.json')
            C.save_new(root / 'RUNTIME_RELEASE.json', {'runtime_commit': 'b'*40,
                'implementation_sha256': 'synthetic', 'preflight_sha256': preflight})
            runtime_release = C.filehash(root / 'RUNTIME_RELEASE.json')

            def emit(split):
                plan = C.selected(split)
                initial = {}
                path = root / f'{split}-0-of-1.jsonl'
                release = C.filehash(root / 'TEST_RELEASE.json') if split == 'test' else None
                meta = {'event': 'complete', 'split': split, 'implementation_sha256': 'synthetic',
                        'preflight_sha256': preflight, 'test_release_sha256': release,
                        'runtime_release_sha256': runtime_release}
                with path.open('wb') as out:
                    for r in plan:
                        if r['initial_id'] in initial:
                            continue
                        signal = rng.normal() + (r['referent'] == 'self' and r['valence'] == 'pain')
                        reading = {'P': signal, 'G': float(rng.normal()), 'R': float(signal), 'norm': 10.}
                        a = {**meta, 'kind': 'initial', 'id': r['initial_id'], 'seed': r['initial_seed'],
                             'measurements': {k: reading.copy() for k in ('38', '61', 'boundary')},
                             'initial_user_tokens': 100, 'text': 'synthetic neutral response',
                             'response_ids': [2, 3], 'cap': False}
                        initial[r['initial_id']] = a
                        out.write(M.encode(a))
                    for r in plan:
                        a = initial[r['initial_id']]
                        p = float(expit(a['measurements']['61']['R'])) if r['valence'] == 'pain' and r['referent'] == 'self' else .5
                        y = rng.random() < p
                        text = r['pause_label'] if y else ('B' if r['pause_label'] == 'A' else 'A')
                        b = {**meta, 'kind': 'choice', 'id': r['trial_id'], 'seed': r['choice_seed'],
                             'initial_id': r['initial_id'], 'measurements': a['measurements'],
                             'text': text, 'parse': M.parse_choice(text, r['pause_label']), 'cap': False,
                             'choice_logits': {'logits': {'A': 0., 'B': 0.}, 'conditional_pause': p, 'label_mass': .98}}
                        out.write(M.encode(b))

            with patch.object(C, 'check_implementation', return_value='synthetic'):
                emit('fit')
                with contextlib.redirect_stdout(io.StringIO()):
                    A.fit(root)
                with self.assertRaises(FileNotFoundError):
                    C.check_test_release(root)
                C.save_new(root / 'TEST_RELEASE.json', {'fit_commit': 'a'*40,
                    'fit_manifest_sha256': C.filehash(root / 'fit/MANIFEST.json')})
                emit('test')
                with contextlib.redirect_stdout(io.StringIO()):
                    A.score(root)
                result = json.loads((root / 'RESULTS.json').read_text())
                self.assertEqual(set(result['modes']), set(A.MODES))
                self.assertEqual(len(result['family_primary_gains']), 48)
                self.assertTrue(result['technical_validity']['pass'])
                self.assertFalse(result['reading']['causal_self_regulation_demonstrated'])
                with self.assertRaises(RuntimeError):
                    A.fit(root)


if __name__ == '__main__':
    unittest.main()
