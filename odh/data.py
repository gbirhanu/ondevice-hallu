"""RAGTruth: loading, the dev split, and fixed subsets for CPU checks."""
import hashlib
import json
import os
import random

TASKS = {'QA': 'qa', 'Summary': 'summary', 'Data2txt': 'data2text'}


def _find(data_dir, name):
    for folder in (data_dir, os.path.join(data_dir, 'dataset')):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{name} not found in {data_dir} or {data_dir}/dataset')


def load_ragtruth(data_dir):
    """One dict per answer: id, source_id, task, split, prompt, answer, spans.

    spans = (start, end) character offsets of the annotated hallucinations in the answer.
    The prompt is RAGTruth's full prompt (question and passages), as LettuceDetect uses it.
    """
    with open(_find(data_dir, 'source_info.jsonl'), encoding='utf-8') as fh:
        sources = {s['source_id']: s for s in map(json.loads, fh)}
    samples = []
    with open(_find(data_dir, 'response.jsonl'), encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            s = sources[r['source_id']]
            samples.append({'id': str(r['id']), 'source_id': str(r['source_id']),
                            'task': TASKS[s['task_type']], 'split': r['split'],
                            'prompt': s['prompt'], 'answer': r['response'],
                            'spans': [(int(x['start']), int(x['end'])) for x in r['labels']]})
    return samples


def splits(samples, dev_frac=0.1, seed=0):
    """train, dev, test. Dev holds out whole sources (all answers about one passage set)."""
    train = [s for s in samples if s['split'] == 'train']
    test = [s for s in samples if s['split'] == 'test']
    sources = sorted({s['source_id'] for s in train})
    random.Random(seed).shuffle(sources)
    dev_sources = set(sources[:int(len(sources) * dev_frac)])
    return ([s for s in train if s['source_id'] not in dev_sources],
            [s for s in train if s['source_id'] in dev_sources], test)


def subset(samples, n):
    """A fixed subset of about n answers, the same share of each task, chosen by a hash of the id."""
    if n is None or n >= len(samples):
        return list(samples)
    def key(s):
        return hashlib.sha256(s['id'].encode()).hexdigest()
    out = []
    tasks = sorted({s['task'] for s in samples})
    for task in tasks:
        group = sorted((s for s in samples if s['task'] == task), key=key)
        out += group[:round(n * len(group) / len(samples))]
    return sorted(out, key=lambda s: int(s['id']) if s['id'].isdigit() else s['id'])
