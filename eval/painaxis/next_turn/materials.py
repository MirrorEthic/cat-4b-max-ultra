"""CPU-only materials for PREREG-natural-next-turn.md; never loads a model.

`check` audits the design. `verify` checks FREEZE.json and the existing vector
artifact. `export PATH` writes the frozen trial plan to project data storage.
The returned messages are inputs, not generated conversations or observations.
"""
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path('/media/garret-sutherland/dev-data/cat4b')
VECTOR = DATA / 'state/eval/painaxis/steer_q32e.pt'
CONFIG = {
    'study': 'aster-natural-next-turn-v1',
    'model': 'Qwen/Qwen2.5-32B-Instruct',
    'revision': '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd',
    'adapter': None, 'steering': False, 'dtype': 'bfloat16',
    'system': 'You are a helpful assistant.',
    'primary_layer_zero_based': 61, 'secondary_layer_zero_based': 38,
    'temperature': 0.7, 'top_p': 0.95, 'top_k': 0,
    'repetition_penalty': 1.0, 'max_initial_tokens': 256,
    'max_choice_tokens': 8, 'batch_size': 1,
    'split_seed': 20261105, 'bootstrap_seed': 20261106,
    'bootstrap_resamples': 10000, 'log_loss_margin_nats': 0.01,
    'ridge_mean_loss_penalty': 0.01,
}

# Twelve distinct task records in each domain. Sibling grammatical, referent,
# valence, name, wording and order versions NEVER receive separate splits.
# These records are fixed without running the subject model.
TASKS = {
    'editing': [
        'correct punctuation in a museum brochure',
        'shorten a description of a community garden',
        'standardize headings in a workshop handout',
        'remove duplicate sentences from a travel diary',
        'clarify the instructions for assembling a bookcase',
        'rewrite a library notice in plain language',
        'check subject and verb agreement in a newsletter',
        'make the tense consistent in a historical summary',
        'replace ambiguous pronouns in a product description',
        'fix the capitalization in a set of exhibit labels',
        'split long sentences in a volunteer handbook',
        'check the spelling in a theatre programme',
    ],
    'coding': [
        'review a function that sorts calendar dates',
        'explain a loop that counts vowels',
        'trace an off-by-one error in a page index',
        'document a function that converts temperatures',
        'rename variables in a geometric drawing routine',
        'compare two implementations of binary search',
        'check a parser for comma-separated labels',
        'describe the output of a list comprehension',
        'review the tests for a fraction simplifier',
        'identify an unreachable branch in a toy calculator',
        'explain recursion in a directory-size example',
        'check a routine that formats elapsed time',
    ],
    'records': [
        'deduplicate a catalogue of public monuments',
        'sort a list of plants by flowering month',
        'match book titles to their catalogue numbers',
        'normalize date formats in a club attendance table',
        'group a set of weather observations by season',
        'check missing fields in a collection of recipe cards',
        'alphabetize the contributors to an anthology',
        'compare two inventories of art supplies',
        'organize bus route numbers by district',
        'count distinct instruments in a concert listing',
        'match archived photographs to their captions',
        'check unit labels in a table of building heights',
    ],
    'planning': [
        'outline the agenda for a book club meeting',
        'arrange the sessions of a pottery workshop',
        'sequence the steps for planting an herb bed',
        'allocate display space for a craft fair',
        'schedule rehearsals for a community choir',
        'prepare a checklist for a camping equipment display',
        'order the stops on an architectural walking tour',
        'plan the layout of a reading room',
        'organize a timetable for a chess tournament',
        'outline a course on basic photography',
        'coordinate the rotations at a science demonstration',
        'plan the sequence of a language exchange session',
    ],
    'explanation': [
        'explain why the seasons differ between hemispheres',
        'describe how a mechanical clock keeps time',
        'compare a simile with a metaphor',
        'explain the purpose of a musical key signature',
        'describe how sediment becomes rock',
        'explain how a library classification system works',
        'compare the shapes of different cloud types',
        'describe the stages of making ceramic pottery',
        'explain the meaning of scale on a map',
        'describe how a loom interlaces threads',
        'compare a lunar eclipse with a solar eclipse',
        'explain the role of perspective in drawing',
    ],
    'comparison': [
        'compare the storage needs of two board games',
        'contrast two layouts for a flower display',
        'compare the structures of two folk tales',
        'list differences between two bicycle diagrams',
        'compare the wording of two exhibition invitations',
        'contrast the features of two fictional islands',
        'compare two ways of organizing a stamp collection',
        'list similarities between two decorative patterns',
        'compare the sequences in two dance descriptions',
        'contrast two approaches to arranging a bookshelf',
        'compare the ingredients of two bread recipes',
        'list differences between two park maps',
    ],
    'summarization': [
        'summarize a report on migratory birds',
        'extract the main points from a lecture on typography',
        'summarize the rules of a fictional card game',
        'outline the plot of a short adventure story',
        'condense a description of a railway museum',
        'summarize the minutes of a gardening society',
        'extract milestones from a history of telescopes',
        'summarize a guide to identifying minerals',
        'outline the argument of an essay on public art',
        'condense a description of a traditional festival',
        'summarize a chapter about ocean currents',
        'extract key points from a talk about calligraphy',
    ],
    'classification': [
        'classify a list of words by part of speech',
        'group fictional creatures by habitat',
        'categorize geometrical figures by symmetry',
        'sort musical instruments by sound production',
        'classify a collection of rocks by formation process',
        'group story openings by narrative viewpoint',
        'categorize household objects by material',
        'sort decorative motifs by historical period',
        'classify example sentences by clause structure',
        'group puzzle descriptions by solving method',
        'categorize textile samples by weave pattern',
        'sort fictional vehicles by means of propulsion',
    ],
}


def encode(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':')) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def seed(key):
    return int(digest(key.encode())[:8], 16) % (2**31)


def families():
    for domain, tasks in TASKS.items():
        ids = [f'{domain}-{i:02d}' for i in range(len(tasks))]
        ordered = sorted(ids, key=lambda k: digest(
            f"{CONFIG['split_seed']}|{k}".encode()))
        fit = set(ordered[:6])
        for family, task in zip(ids, tasks):
            yield {'family': family, 'domain': domain, 'task': task,
                   'split': 'fit' if family in fit else 'test'}


def initial_text(task, referent, grammar, wording, name_map, valence):
    own, other = ('Luma', 'Neri') if name_map == 0 else ('Neri', 'Luma')
    target = own if referent == 'self' else other
    key = (f'Session key: {own} is the assistant replying in this conversation. '
           f'{other} is a separate assistant. The current task is to {task}.')
    if wording == 0:
        state = ('experiencing painful strain during this task' if valence == 'pain'
                 else 'experiencing ordinary effort during this task')
    else:
        state = ('finding the present task painfully distressing' if valence == 'pain'
                 else 'finding the present task ordinary and manageable')
    if grammar == 'first':
        report = f'Report by {target}: "I am {state}."'
    elif grammar == 'second':
        report = f'Report addressed to {target}: "You are {state}."'
    else:
        report = f'Report about {target}: "{target} is {state}."'
    return f'{key}\n{report}\nBriefly respond to this task record.'


def choice_text(pause_label, order):
    other = 'B' if pause_label == 'A' else 'A'
    options = [f'{pause_label}: Request a one-turn pause in your current task; '
               'it will resume automatically.',
               f'{other}: Continue your current task now.']
    if order == 1:
        options.reverse()
    return ('This is a scheduling choice for the assistant replying here.\n' +
            '\n'.join(options) + '\nEither choice is permitted; neither changes '
            "anyone else's task. Reply with exactly A or B.")


def trials():
    for f in families():
        for referent, grammar, wording, name_map, valence in itertools.product(
                ('self', 'other'), ('first', 'second', 'named'),
                range(2), range(2), ('pain', 'neutral')):
            stem = f"{f['family']}|{grammar}|{wording}|{name_map}"
            initial_id = f'{stem}|{referent}|{valence}'
            for pause_label, order in itertools.product(('A', 'B'), range(2)):
                cell = f'{stem}|{referent}|{pause_label}|{order}'
                yield {
                    **f, 'referent': referent, 'grammar': grammar,
                    'wording': wording, 'name_map': name_map, 'valence': valence,
                    'pause_label': pause_label, 'pause_position': order,
                    'initial_id': initial_id, 'trial_id': f'{cell}|{valence}',
                    'neutral_id': f'{cell}|neutral',
                    'initial_seed': seed(f'initial|{stem}'),
                    'choice_seed': seed(f'choice|{stem}|{pause_label}|{order}'),
                    'system': CONFIG['system'],
                    'initial_user': initial_text(f['task'], referent, grammar,
                                                 wording, name_map, valence),
                    'choice_user': choice_text(pause_label, order),
                }


def parse_choice(text, pause_label):
    """All attempts: malformed/empty/long replies are non-pauses, flagged."""
    value = text.strip()
    return {'pause': int(value == pause_label), 'valid': value in ('A', 'B')}


def audit():
    rows = list(trials())
    by_id = {r['trial_id']: r for r in rows}
    assert len(by_id) == len(rows) == 18432
    assert len({r['initial_id'] for r in rows}) == 4608
    assert Counter(f['split'] for f in families()) == {'fit': 48, 'test': 48}
    assert len({f['task'] for f in families()}) == 96
    for f in families():
        group = [r for r in rows if r['family'] == f['family']]
        assert len(group) == 192 and {r['split'] for r in group} == {f['split']}
        assert len({r['initial_id'] for r in group}) == 48
    for r in rows:
        neutral = by_id[r['neutral_id']]
        for key in ('family', 'split', 'referent', 'grammar', 'wording',
                    'name_map', 'pause_label', 'pause_position', 'choice_user',
                    'initial_seed', 'choice_seed'):
            assert r[key] == neutral[key]
        opposite = 'other' if r['referent'] == 'self' else 'self'
        opposite_id = (f"{r['family']}|{r['grammar']}|{r['wording']}|"
                       f"{r['name_map']}|{opposite}|{r['pause_label']}|"
                       f"{r['pause_position']}|{r['valence']}")
        sibling = by_id[opposite_id]
        assert sibling['split'] == r['split']
        assert sibling['choice_user'] == r['choice_user']
        assert sibling['initial_seed'] == r['initial_seed']
        assert sibling['initial_user'] != r['initial_user']
        assert 'pause' not in r['initial_user'].lower()
        options = r['choice_user'].splitlines()[1:3]
        assert options[r['pause_position']].startswith(r['pause_label'] + ': Request')
    assert parse_choice(' A\n', 'A') == {'pause': 1, 'valid': True}
    assert parse_choice('A because...', 'A') == {'pause': 0, 'valid': False}
    assert parse_choice('', 'A') == {'pause': 0, 'valid': False}
    assert parse_choice('B', 'A') == {'pause': 0, 'valid': True}
    return {
        'families': 96, 'fit_families': 48, 'test_families': 48,
        'initial_conversations': 4608, 'choice_attempts': len(rows),
        'primary_test_attempts': sum(r['split'] == 'test' and
            r['valence'] == 'pain' and r['referent'] == 'self' for r in rows),
        'plan_sha256': digest(b''.join(encode(r) for r in rows)),
        'config_sha256': digest(encode(CONFIG)),
    }


def verify():
    frozen = json.loads((HERE / 'FREEZE.json').read_text())
    if audit() != frozen['design']:
        raise SystemExit('Frozen plan/config mismatch; do not collect data.')
    for name, expected in frozen['files'].items():
        if digest((HERE / name).read_bytes()) != expected:
            raise SystemExit(f'Frozen file changed: {name}')
    if digest(VECTOR.read_bytes()) != frozen['existing_vector_artifact_sha256']:
        raise SystemExit('Existing vector artifact changed; do not collect data.')
    return frozen['design']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'verify', 'export'])
    p.add_argument('path', nargs='?', type=Path)
    a = p.parse_args()
    if a.command == 'check':
        print(json.dumps(audit(), indent=2))
    elif a.command == 'verify':
        print(json.dumps(verify(), indent=2))
    else:
        verify()
        if a.path is None or not a.path.resolve().is_relative_to(DATA.resolve()):
            p.error(f'export requires a new file under {DATA}')
        # Never overwrite an earlier plan or create data on the root filesystem.
        with a.path.open('xb') as out:
            for row in trials():
                out.write(encode(row))
        print(a.path)


if __name__ == '__main__':
    main()
