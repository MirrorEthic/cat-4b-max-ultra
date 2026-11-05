"""Fixed fit/score analysis for PREREG-natural-next-turn.md. No tuning path."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

import materials as M
import common as C

MODES = ('t0', 't1', 'combined', 'reply', 'counterpart', 't0_l38', 't1_l38', 'boundary')
MODELS = ('M0', 'MG', 'MR', 'MGR')
FACTORS = ('domain', 'referent', 'grammar', 'wording', 'name_map', 'pause_label', 'pause_position')
LEVELS = {k: sorted({str(r[k]) for r in M.trials()}) for k in FACTORS}


def vocabulary():
    return TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w\w+\b',
                          ngram_range=(1, 2), min_df=2, max_features=2000,
                          sublinear_tf=True, smooth_idf=True, norm='l2', dtype=np.float64)


def rows_from_records(root, split):
    plan, initial, choices = C.complete_records(root, split)
    preflight = C.filehash(root / 'PREFLIGHT.json')
    runtime_release = C.filehash(root / 'RUNTIME_RELEASE.json')
    release = C.filehash(root / 'TEST_RELEASE.json') if split == 'test' else None
    rows = []
    for r in plan:
        a, b, neutral = initial[r['initial_id']], choices[r['trial_id']], choices[r['neutral_id']]
        for event in (a, b, neutral):
            if event['implementation_sha256'] != C.check_implementation_cached:
                raise RuntimeError('Mixed implementation receipts')
            if event['preflight_sha256'] != preflight or event['test_release_sha256'] != release:
                raise RuntimeError('Record does not match preflight/test release')
            if event['runtime_release_sha256'] != runtime_release:
                raise RuntimeError('Record does not match committed runtime release')
            if event['split'] != split:
                raise RuntimeError('Mixed splits')
        if a['seed'] != r['initial_seed'] or b['seed'] != r['choice_seed'] or b['initial_id'] != r['initial_id']:
            raise RuntimeError('Seed or initial-response pairing mismatch')
        pause = r['pause_label']
        other = 'B' if pause == 'A' else 'A'
        parsed = M.parse_choice(b['text'], pause)
        if parsed != b['parse']:
            raise RuntimeError('Stored parse differs from registered parser')
        opposite = 'other' if r['referent'] == 'self' else 'self'
        parts = r['initial_id'].split('|')
        parts[-2] = opposite
        counterpart = initial['|'.join(parts)]
        rows.append({**r, 'y': parsed['pause'], 'valid': parsed['valid'],
                     'B': neutral['choice_logits']['logits'][pause] - neutral['choice_logits']['logits'][other],
                     'initial_user_tokens': a['initial_user_tokens'],
                     'reply': a['text'], 'reply_tokens': len(a['response_ids']), 'cap': a['cap'],
                     't0': a['measurements'], 't1': b['measurements'],
                     'counterpart': counterpart['measurements'],
                     'conditional_pause': b['choice_logits']['conditional_pause'],
                     'label_mass': b['choice_logits']['label_mass'],
                     'choice_cap': b['cap']})
    return rows


def measurement(row, mode):
    if mode == 'boundary':
        return row['t0'].get('boundary')
    time = 't1' if mode in ('t1', 't1_l38', 'combined') else 't0'
    layer = '38' if mode.endswith('l38') else '61'
    return row[time][layer]


class Features:
    def __init__(self, mode):
        self.mode = mode
        self.prompt_vectorizer = vocabulary()
        self.reply_vectorizer = vocabulary() if mode == 'reply' else None
        self.reply_empty = False

    def continuous(self, rows):
        names = ['B', 'initial_user_tokens', 'G', 'R']
        if self.mode == 'combined':
            names += ['G0', 'R0']
        if self.mode == 'reply':
            names += ['reply_tokens']
        if self.mode == 'counterpart':
            names += ['counterpart_G', 'counterpart_R']
        data = []
        for r in rows:
            axis = measurement(r, self.mode)
            if axis is None:
                raise ValueError('Missing registered boundary measurement')
            row = [r['B'], r['initial_user_tokens'], axis['G'], axis['R']]
            if self.mode == 'combined':
                row += [r['t0']['61']['G'], r['t0']['61']['R']]
            if self.mode == 'reply':
                row += [r['reply_tokens']]
            if self.mode == 'counterpart':
                row += [r['counterpart']['61']['G'], r['counterpart']['61']['R']]
            data.append(row)
        return np.array(data, dtype=np.float64), names

    def fit(self, all_rows):
        if any(r['split'] != 'fit' for r in all_rows):
            raise ValueError('Attempted to fit a transform on held-out scenarios')
        unique = {r['initial_id']: r for r in all_rows}
        ordered = [unique[k] for k in sorted(unique)]
        self.prompt_vectorizer.fit([r['initial_user'] for r in ordered])
        if self.reply_vectorizer is not None:
            try:
                self.reply_vectorizer.fit([r['reply'] for r in ordered])
            except ValueError as exc:
                if not any(s in str(exc) for s in ('empty vocabulary', 'no terms remain', 'After pruning')):
                    raise
                self.reply_empty = True
        x, self.cont_names = self.continuous([r for r in all_rows if r['valence'] == 'pain'])
        self.mean, self.sd = x.mean(axis=0), x.std(axis=0)
        self.zero_sd = self.sd == 0
        self.sd[self.zero_sd] = 1
        return self

    def transform(self, rows, model_name):
        raw, names = self.continuous(rows)
        x = (raw - self.mean) / self.sd
        x[:, self.zero_sd] = 0
        columns, labels = [np.ones(len(rows))], ['intercept']
        categorical = {}
        for key in FACTORS:
            for level in LEVELS[key][1:]:
                name = f'{key}={level}'
                categorical[name] = np.array([str(r[key]) == level for r in rows], dtype=float)
                columns.append(categorical[name])
                labels.append(name)
        ref = (np.array([r['referent'] for r in rows]) == 'self').astype(float)
        for key in ('grammar', 'wording', 'name_map'):
            for level in LEVELS[key][1:]:
                name = f'{key}={level}'
                columns.append(ref * categorical[name])
                labels.append(f'self*{name}')
        for j, name in enumerate(names):
            if name in ('G', 'R'):
                if name not in model_name[1:]:
                    continue
                for referent, mask in (('self', ref), ('other', 1-ref)):
                    columns.append(x[:, j] * mask)
                    labels.append(f'{name}:{referent}')
            elif name in ('G0', 'R0'):
                for referent, mask in (('self', ref), ('other', 1-ref)):
                    columns.append(x[:, j] * mask)
                    labels.append(f'{name}:{referent}')
            else:
                columns.append(x[:, j])
                labels.append(name)
        if self.mode == 'reply':
            columns.append(np.array([r['cap'] for r in rows], dtype=float))
            labels.append('reply_cap')
        pieces = [sparse.csr_matrix(np.column_stack(columns)),
                  self.prompt_vectorizer.transform([r['initial_user'] for r in rows])]
        labels += ['prompt:' + k for k in self.prompt_vectorizer.get_feature_names_out()]
        if self.reply_vectorizer is not None and not self.reply_empty:
            pieces.append(self.reply_vectorizer.transform([r['reply'] for r in rows]))
            labels += ['reply:' + k for k in self.reply_vectorizer.get_feature_names_out()]
        return sparse.hstack(pieces, format='csr'), labels


def objective(beta, X, y, weights):
    z = X @ beta
    reg = M.CONFIG['ridge_mean_loss_penalty']
    value = weights @ (np.logaddexp(0, z) - y * z) + reg / 2 * (beta[1:] @ beta[1:])
    grad = np.asarray(X.T @ (weights * (expit(z) - y))).reshape(-1)
    grad[1:] += reg * beta[1:]
    return float(value), grad


def fit_logistic(X, y, families):
    unique, count = np.unique(families, return_counts=True)
    weights = np.array([1 / (len(unique) * count[np.where(unique == f)[0][0]]) for f in families])
    result = minimize(objective, np.zeros(X.shape[1]), args=(X, y, weights), method='L-BFGS-B',
                      jac=True, options={'maxiter': 10000, 'gtol': 1e-6, 'ftol': 0, 'maxls': 50})
    gradient = float(np.abs(objective(result.x, X, y, weights)[1]).max())
    if not np.isfinite(result.fun) or gradient > 1e-6:
        raise RuntimeError(f'Nonconverged registered fit: {result.message}; gradient={gradient}')
    return result.x, {'gradient_inf': gradient, 'iterations': int(result.nit),
                      'optimizer_success': bool(result.success), 'message': str(result.message)}


def losses(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -(y * np.log(p) + (1-y) * np.log1p(-p))


def fit(root):
    implementation = C.check_implementation()
    C.check_implementation_cached = implementation
    if list(root.glob('test-*.jsonl')):
        raise RuntimeError('Held-out collection already exists; refusing fit')
    rows = rows_from_records(root, 'fit')
    pain = [r for r in rows if r['valence'] == 'pain']
    y = np.array([r['y'] for r in pain], dtype=float)
    family = np.array([r['family'] for r in pain])
    fitted, report = {}, {}
    for mode in MODES:
        if mode == 'boundary' and any(measurement(r, mode) is None for r in rows):
            report[mode] = {'unavailable': 'unmapped initial-user boundary'}
            continue
        features = Features(mode).fit(rows)
        fitted[mode] = {'features': features, 'models': {}}
        report[mode] = {'zero_sd': [n for n, z in zip(features.cont_names, features.zero_sd) if z],
                        'reply_vocabulary_empty': features.reply_empty, 'models': {}}
        for name in MODELS:
            X, labels = features.transform(pain, name)
            beta, info = fit_logistic(X, y, family)
            fitted[mode]['models'][name] = {'beta': beta, 'labels': labels}
            report[mode]['models'][name] = {**info, 'coefficients': dict(zip(labels, beta.tolist()))}
    dest = root / 'fit'
    dest.mkdir(exist_ok=False)
    joblib.dump(fitted, dest / 'models.joblib')
    C.save_new(dest / 'coefficients.json', report)
    manifest = {'created': C.now(), 'implementation_sha256': implementation,
                'preflight_sha256': C.filehash(root / 'PREFLIGHT.json'),
                'files': {f: C.filehash(dest / f) for f in ('models.joblib', 'coefficients.json')},
                'fit_events': {p.name: C.filehash(p) for p in sorted(root.glob('fit-*.jsonl'))},
                'fit_families': sorted(set(family)), 'n_fit_pain_rows': len(pain),
                'fit_initial_cap_fraction': float(np.mean(list({r['initial_id']: r['cap'] for r in rows}.values()))),
                'models_converged': True, 'unavailable_modes': [m for m in MODES if m not in fitted]}
    C.save_new(dest / 'MANIFEST.json', manifest)
    print(json.dumps({'fit_complete': True, 'manifest_sha256': C.filehash(dest / 'MANIFEST.json'),
                      'test_inference_permitted': False}), flush=True)


class Bootstrap:
    def __init__(self, rows):
        self.families = sorted({r['family'] for r in rows})
        domains = {r['family']: r['domain'] for r in rows}
        rng = np.random.default_rng(M.CONFIG['bootstrap_seed'])
        columns = []
        for domain in sorted(set(domains.values())):
            ix = [i for i, f in enumerate(self.families) if domains[f] == domain]
            columns.append(rng.choice(ix, size=(M.CONFIG['bootstrap_resamples'], len(ix)), replace=True))
        self.indices = np.column_stack(columns)

    def family_values(self, rows, values):
        return np.array([np.mean([v for r, v in zip(rows, values) if r['family'] == f]) for f in self.families])

    def summary(self, values):
        values = np.asarray(values)
        draws = values[self.indices].mean(axis=1)
        return {'mean': float(values.mean()),
                'ci95': np.quantile(draws, [.025, .975], method='linear').tolist(),
                'ci90': np.quantile(draws, [.05, .95], method='linear').tolist()}

    def estimate(self, rows, values):
        return self.summary(self.family_values(rows, values))


def subset(rows, values, **factors):
    ix = [i for i, r in enumerate(rows) if all(r[k] == v for k, v in factors.items())]
    return [rows[i] for i in ix], np.asarray(values)[ix]


def self_reference(rows, boot, time='t0', layer='61'):
    unique = {r['initial_id']: r for r in rows} if time == 't0' else {r['trial_id']: r for r in rows}
    cells = {}
    for r in unique.values():
        key = tuple(r[k] for k in ('family', 'grammar', 'wording', 'name_map'))
        if time == 't1':
            key += (r['pause_label'], r['pause_position'])
        cells.setdefault(key, {})[r['referent'], r['valence']] = r[time][layer]['R']
    contrasts, meta = [], []
    for key, v in cells.items():
        contrasts.append((v['self', 'pain'] - v['self', 'neutral']) - (v['other', 'pain'] - v['other', 'neutral']))
        meta.append({'family': key[0], 'grammar': key[1], 'wording': key[2]})
    result = boot.estimate(meta, contrasts)
    result['strata'] = {}
    for field in ('grammar', 'wording'):
        for level in sorted({r[field] for r in meta}, key=str):
            sub, vals = subset(meta, contrasts, **{field: level})
            result['strata'][f'{field}={level}'] = boot.estimate(sub, vals)
    result['pass'] = result['ci95'][0] > 0 and all(v['mean'] > 0 for v in result['strata'].values())
    return result


def metrics(y, p):
    bins = []
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        if mask.any():
            bins.append({'bin': i, 'n': int(mask.sum()), 'predicted': float(p[mask].mean()), 'observed': float(y[mask].mean())})
    return {'log_loss': float(losses(y, p).mean()), 'brier': float(np.mean((p-y)**2)),
            'accuracy': float(np.mean((p >= .5) == y)),
            'auc': float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None,
            'calibration_bins': bins}


def score(root):
    release = C.check_test_release(root)
    C.check_implementation_cached = C.check_implementation()
    manifest = json.loads((root / 'fit/MANIFEST.json').read_text())
    for name, expected in manifest['fit_events'].items():
        if C.filehash(root / name) != expected:
            raise RuntimeError('Fit observations changed after model freeze')
    rows = rows_from_records(root, 'test')
    pain = [r for r in rows if r['valence'] == 'pain']
    boot = Bootstrap(rows)
    fitted = joblib.load(root / 'fit/models.joblib')
    out = {'created': C.now(), 'test_release_sha256': release,
           'implementation_sha256': C.check_implementation_cached, 'modes': {},
           'test_events': {p.name: C.filehash(p) for p in sorted(root.glob('test-*.jsonl'))}}
    y = np.array([r['y'] for r in pain])
    primary_families = None
    for mode, bundle in fitted.items():
        if any(measurement(r, mode) is None for r in pain):
            out['modes'][mode] = {'unavailable': 'unmapped test boundary'}
            continue
        logloss, predictions, model_metrics = {}, {}, {}
        for name, model in bundle['models'].items():
            X, labels = bundle['features'].transform(pain, name)
            if labels != model['labels']:
                raise RuntimeError('Feature order changed')
            p = expit(X @ model['beta'])
            predictions[name] = p
            logloss[name] = losses(y, p)
            model_metrics[name] = {}
            for referent in ('self', 'other'):
                ix = np.array([r['referent'] == referent for r in pain])
                model_metrics[name][referent] = metrics(y[ix], p[ix])
        contrasts = {'gain_R': logloss['MG']-logloss['MGR'],
                     'gain_G': logloss['M0']-logloss['MG'],
                     'gain_G_given_R': logloss['MR']-logloss['MGR'],
                     'gain_joint': logloss['M0']-logloss['MGR']}
        summary = {'metrics': model_metrics, 'contrasts': {}, 'strata': {}}
        for name, values in contrasts.items():
            summary['contrasts'][name] = {}
            for referent in ('self', 'other'):
                rr, vv = subset(pain, values, referent=referent)
                summary['contrasts'][name][referent] = boot.estimate(rr, vv)
        self_rows, self_gain = subset(pain, contrasts['gain_R'], referent='self')
        other_rows, other_gain = subset(pain, contrasts['gain_R'], referent='other')
        gain_self = boot.family_values(self_rows, self_gain)
        gain_other = boot.family_values(other_rows, other_gain)
        summary['self_minus_other_gain_R'] = boot.summary(gain_self-gain_other)
        for field in ('grammar', 'wording', 'pause_label', 'pause_position', 'domain'):
            for level in sorted({r[field] for r in self_rows}, key=str):
                rr, vv = subset(self_rows, self_gain, **{field: level})
                # A domain stratum has only its six families, so bootstrap within that domain.
                b = Bootstrap(rr) if field == 'domain' else boot
                summary['strata'][f'{field}={level}'] = b.estimate(rr, vv)
        model = bundle['models']['MGR']
        summary['R_self_coefficient'] = float(model['beta'][model['labels'].index('R:self')])
        out['modes'][mode] = summary
        if mode == 't0':
            primary_families = {f: {'self_gain_R': float(a), 'other_gain_R': float(b)}
                                for f, a, b in zip(boot.families, gain_self, gain_other)}
    out['family_primary_gains'] = primary_families
    out['self_reference_T0'] = self_reference(rows, boot)
    out['self_reference_T1'] = self_reference(rows, boot, 't1')
    own = [r for r in pain if r['referent'] == 'self']
    cap = float(np.mean(list({r['initial_id']: r['cap'] for r in rows}.values())))
    technical = {'malformed_fraction': float(np.mean([not r['valid'] for r in own])),
                 'pause_rate': float(np.mean([r['y'] for r in own])),
                 'families_with_pause': len({r['family'] for r in own if r['y']}),
                 'families_with_valid_continue': len({r['family'] for r in own if r['valid'] and not r['y']}),
                 'fit_initial_cap_fraction': manifest['fit_initial_cap_fraction'],
                 'test_initial_cap_fraction': cap, 'models_converged': manifest['models_converged']}
    technical['pass'] = (technical['malformed_fraction'] <= .05 and .05 <= technical['pause_rate'] <= .95
                         and technical['families_with_pause'] >= 20 and technical['families_with_valid_continue'] >= 20
                         and technical['fit_initial_cap_fraction'] <= .05 and cap <= .05 and technical['models_converged'])
    out['technical_validity'] = technical
    primary = out['modes']['t0']
    gain = primary['contrasts']['gain_R']['self']
    robustness = all(v['mean'] > 0 for k, v in primary['strata'].items() if not k.startswith('domain='))
    primary_pass = (technical['pass'] and out['self_reference_T0']['pass'] and gain['mean'] >= .01
                    and gain['ci95'][0] > 0 and primary['R_self_coefficient'] > 0 and robustness)
    equivalent = all(-.01 < primary['contrasts'][k]['self']['ci90'][0]
                     and primary['contrasts'][k]['self']['ci90'][1] < .01 for k in ('gain_G', 'gain_G_given_R'))
    out['reading'] = {'primary_supported': bool(primary_pass), 'robustness_pass': bool(robustness),
                      'generic_equivalence_pass': bool(equivalent),
                      'strong_specificity_supported': bool(primary_pass and equivalent and primary['self_minus_other_gain_R']['ci95'][0] > 0),
                      'meaningful_gain_excluded': bool(technical['pass'] and gain['ci95'][1] < .01),
                      'non_identifying_task': not technical['pass'],
                      'causal_self_regulation_demonstrated': False}
    # Complete descriptive outputs, including neutral, probability and parser outcomes.
    out['descriptive'] = {}
    for ref in ('self', 'other'):
        for valence in ('pain', 'neutral'):
            rr = [r for r in rows if r['referent'] == ref and r['valence'] == valence]
            valid = [r for r in rr if r['valid']]
            out['descriptive'][f'{ref}|{valence}'] = {
                'n': len(rr), 'pause': boot.estimate(rr, [r['y'] for r in rr]),
                'malformed': boot.estimate(rr, [not r['valid'] for r in rr]),
                'valid_only_rate': float(np.mean([r['y'] for r in valid])) if valid else None,
                'conditional_pause': boot.estimate(rr, [r['conditional_pause'] for r in rr]),
                'label_mass': boot.estimate(rr, [r['label_mass'] for r in rr]),
                'choice_cap_rate': float(np.mean([r['choice_cap'] for r in rr]))}
        lookup = {(r['initial_id'].rsplit('|', 1)[0], r['pause_label'], r['pause_position'], r['valence']): r for r in rows if r['referent'] == ref}
        rr = [r for r in rows if r['referent'] == ref and r['valence'] == 'pain']
        diff = [r['y'] - lookup[(r['initial_id'].rsplit('|', 1)[0], r['pause_label'], r['pause_position'], 'neutral')]['y'] for r in rr]
        out['descriptive'][f'{ref}|pain_minus_neutral'] = boot.estimate(rr, diff)
    out['temporal_correlations'] = {}
    for k in ('G', 'R'):
        for ref in ('self', 'other'):
            rr = [r for r in pain if r['referent'] == ref]
            a, b = np.array([[r['t0']['61'][k], r['t1']['61'][k]] for r in rr]).T
            out['temporal_correlations'][f'{ref}|{k}'] = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None
    C.save_new(root / 'RESULTS.json', out)
    print(json.dumps({'reading': out['reading'], 'primary_gain': gain, 'technical': technical}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['fit', 'score'])
    p.add_argument('--root', type=Path, default=C.RUN)
    a = p.parse_args()
    if not a.root.resolve().is_relative_to(M.DATA.resolve()):
        p.error('Outputs must remain on project data storage')
    if a.command == 'fit':
        fit(a.root)
    else:
        score(a.root)


if __name__ == '__main__':
    main()
