"""On-device hallucination detection experiments on RAGTruth.

    python run.py models                          # find the published small detectors
    python run.py score  --name tiny68m           # span F1 on the full test set (GPU if present)
    python run.py export --name tiny68m           # ONNX full precision + 8-bit (--recipe w8, mix or dyn)
    python run.py check8bit --name tiny68m        # what 8-bit costs in F1, on a fixed test subset
    python run.py bench  --name tiny68m --threads 1 2   # time per answer on this CPU
    python run.py train  --name mine68m --base jhu-clsp/ettin-encoder-68m
    python run.py table                           # results/table.md and results/ladder.png
    python run.py signals --generator llama-2-7b-chat   # the answering model's signals (GPU)
    python run.py fusion                          # do those signals add to text checkers?
    python run.py teach  --name ld_large          # teacher's word probabilities (training answers)
    python run.py generate --generator qwen2.5-1.5b    # new answers from a small model (GPU)
    python run.py train  --name kd17m --base jhu-clsp/ettin-encoder-17m --teacher ld_large --extra qwen2.5-1.5b
    python run.py distill                         # table of the teacher-student test
    python run.py generate --generator qwen2.5-7b  # answers from a large model (4-bit), for comparison
    python run.py llmlabel --name gpt4o --llm-model gpt-4o   # a large LLM labels the new answers (API)
    python run.py psilo                           # score checkers on PsiloQA, by answer-model size
    python run.py compare --pairs kd17m:kdlarge17m kd17m:llm17m   # differences with 95% intervals

Every command skips work whose result file already exists (add --redo to repeat it).
Results are small JSON files in --work-dir/results; nothing else is ever overwritten.
"""
import argparse
import gzip
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from odh import data  # noqa: E402
from odh.metrics import bootstrap, score  # noqa: E402
from odh.spans import all_spans  # noqa: E402

# Published detectors, smallest first: (run name, label, Hugging Face id patterns in order of preference)
LADDER = [
    ('tiny17m', 'TinyLettuce 17M', [r'KRLabsOrg/tinylettuce-ettin-17m-en-v1', r'KRLabsOrg/tinylettuce-ettin-17m-en']),
    ('tiny32m', 'TinyLettuce 32M', [r'KRLabsOrg/tinylettuce-ettin-32m-en-v1', r'KRLabsOrg/tinylettuce-ettin-32m-en']),
    ('tiny68m', 'TinyLettuce 68M', [r'KRLabsOrg/tinylettuce-ettin-68m-en-v1', r'KRLabsOrg/tinylettuce-ettin-68m-en']),
    ('ld_base', 'LettuceDetect-base', [r'KRLabsOrg/lettucedect-base-modernbert-en-v1']),
    ('ld_large', 'LettuceDetect-large', [r'KRLabsOrg/lettucedect-large-modernbert-en-v1']),
]


def paths(work):
    p = {k: os.path.join(work, k) for k in ('results', 'preds', 'onnx', 'models', 'signals', 'extra', 'llm')}
    for d in p.values():
        os.makedirs(d, exist_ok=True)
    return p


def write_json(path, value):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(value, fh, indent=2)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def done(path, args, recipe=None):
    """True when the result exists (and, for 8-bit results, was made with the same recipe)."""
    if os.path.exists(path) and not args.redo:
        if recipe and read_json(path).get('recipe', 'dyn') != recipe:
            print(f'{path} was made with another 8-bit recipe; redoing it with {recipe}.')
            return False
        print(f'Already done: {path} (add --redo to repeat).')
        return True
    return False


def model_of(p, name, given=None):
    """The model behind a run name: --model, else the resolved published id, else a trained run."""
    if given:
        return given
    scored = os.path.join(p['results'], f'{name}_score.json')
    if os.path.exists(scored):
        return read_json(scored)['model']
    resolved = os.path.join(p['results'], 'models.json')
    if os.path.exists(resolved) and name in read_json(resolved):
        return read_json(resolved)[name]['id']
    trained = os.path.join(p['models'], name)
    if os.path.isdir(trained):
        return trained
    raise ValueError(f'No model known for {name}: run "models" first or pass --model.')


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def cmd_models(args, p):
    from huggingface_hub import HfApi
    api, out = HfApi(), {}
    available = {m.id for m in api.list_models(author='KRLabsOrg', limit=500)}
    for name, label, patterns in LADDER:
        found = next((pat for pat in patterns if pat in available), None)
        if found is None:      # fall back to any id that matches the size
            size = re.search(r'(\d+m)', patterns[0])
            found = next((m for m in sorted(available) if size and re.search(
                rf'tinylettuce-ettin-{size.group(1)}-en(-v\d+)?$', m)), None)
        if found:
            out[name] = {'label': label, 'id': found}
            print(f'{name:9s} {label:22s} {found}')
        else:
            print(f'{name:9s} {label:22s} NOT FOUND on Hugging Face; skipped')
    write_json(os.path.join(p['results'], 'models.json'), out)


def save_preds(path, ids, tokens):
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump({'ids': ids, 'tokens': tokens}, fh)


def load_preds(path, samples):
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        record = json.load(fh)
    by_id = dict(zip(record['ids'], record['tokens']))
    return [by_id[s['id']] for s in samples]


def _scores(spans, samples):
    s = score(spans, samples)
    return {'span_f1': s['all'][2], 'precision': s['all'][0], 'recall': s['all'][1],
            'f1_ci95': bootstrap(spans, samples),
            'by_task': {t: {'precision': v[0], 'recall': v[1], 'span_f1': v[2]}
                        for t, v in s.items() if t != 'all'}}


def cmd_score(args, p):
    out = os.path.join(p['results'], f'{args.name}_score.json')
    if done(out, args):
        return
    from odh.model import load, predict
    _, _, test = data.splits(data.load_ragtruth(args.data_dir))
    test = data.subset(test, args.limit)
    name = model_of(p, args.name, args.model)
    print(f'Scoring {name} on {len(test)} test answers on {device()}.')
    t0 = time.time()
    tok, model = load(name, device())
    tokens = predict(model, tok, test, args.max_length, log=print)
    save_preds(os.path.join(p['preds'], f'{args.name}_test.json.gz'), [s['id'] for s in test], tokens)
    result = {'name': args.name, 'model': name, 'answers': len(test), 'cutoff': 0.5,
              'params': sum(x.numel() for x in model.parameters()), 'max_length': args.max_length,
              'seconds': round(time.time() - t0), **_scores(all_spans(tokens), test)}
    write_json(out, result)
    print(f'{args.name}: span F1 {result["span_f1"]} {result["f1_ci95"]} '
          f'(QA {result["by_task"].get("qa", {}).get("span_f1")}, '
          f'summary {result["by_task"].get("summary", {}).get("span_f1")})')


def encoded(tok, samples, max_length):
    from odh.model import encode
    return [encode(tok, s['prompt'], s['answer'], max_length)[0] for s in samples]


def make_onnx(args, p, name):
    """ONNX files of a run (full precision and 8-bit), made only when missing or when the
    8-bit file was made with another recipe."""
    from transformers import AutoTokenizer
    from odh import cpu
    folder = os.path.join(p['onnx'], args.name)
    fp32, int8 = os.path.join(folder, 'model_fp32.onnx'), os.path.join(folder, 'model_int8.onnx')
    recipe_file = os.path.join(folder, 'int8_recipe.txt')
    params, made = None, None
    if not os.path.exists(fp32):
        print(f'Exporting {name} to ONNX in {folder}.')
        params = cpu.export(name, folder)
    old = open(recipe_file).read().strip() if os.path.exists(recipe_file) else None
    if not os.path.exists(int8) or old != args.recipe:
        print(f'Making the 8-bit file of {args.name} (recipe {args.recipe}).')
        calib = None
        if args.recipe == 'mix':      # a few training answers, only to find layers with extreme inputs
            train, _, _ = data.splits(data.load_ragtruth(args.data_dir))
            calib = encoded(AutoTokenizer.from_pretrained(folder), data.subset(train, 8), args.max_length)
        made = cpu.quantize(fp32, int8, args.recipe, calib=calib)
        with open(recipe_file, 'w') as fh:
            fh.write(args.recipe)
    return folder, fp32, int8, params, made


def cmd_export(args, p):
    out = os.path.join(p['results'], f'{args.name}_export.json')
    if done(out, args, args.recipe):
        make_onnx(args, p, model_of(p, args.name, args.model))
        return
    from transformers import AutoTokenizer
    from odh import cpu
    name = model_of(p, args.name, args.model)
    folder, fp32, int8, params, made = make_onnx(args, p, name)
    if params is None:
        from odh.model import load
        params = sum(x.numel() for x in load(name)[1].parameters())
    train, _, test = data.splits(data.load_ragtruth(args.data_dir))
    check = data.subset(test, 12)
    worst = cpu.parity(name, fp32, check, args.max_length)
    tok = AutoTokenizer.from_pretrained(folder)
    close = cpu.agreement(fp32, int8, encoded(tok, check, args.max_length))
    # How coarse one-scale 8-bit rounding of each layer's input would be (why "dyn" failed)
    ranges = cpu.input_ranges(fp32, encoded(tok, data.subset(train, 4), args.max_length))
    extreme = sorted((v, k) for k, v in ranges.items() if v > cpu.EXTREME)
    result = {'name': args.name, 'model': name, 'params': params, 'recipe': args.recipe,
              'fp32_mb': cpu.size_mb(fp32), 'int8_mb': cpu.size_mb(int8),
              'onnx_vs_pytorch_max_prob_diff': worst, 'int8_vs_fp32': close,
              'layers_with_weights': len(ranges), 'layers_with_extreme_inputs': len(extreme),
              'worst_input_step_over_typical': max(ranges.values()) if ranges else None,
              'extreme_layers': [k for _, k in reversed(extreme)][:40], 'quantize': made}
    write_json(out, result)
    print(f'{args.name}: {params / 1e6:.0f}M parameters, {result["fp32_mb"]} MB -> {result["int8_mb"]} MB '
          f'in 8-bit ({args.recipe}); ONNX matches PyTorch within {worst:.1e}; 8-bit against float: '
          f'largest probability change {close["max_prob_diff"]}, decisions flipped {100 * close["flipped_share"]:.2f}%; '
          f'{len(extreme)} of {len(ranges)} layers have extreme inputs.')
    if worst > 1e-2:
        print('WARNING: the ONNX file does not match PyTorch; do not trust its numbers.')
    if close['flipped_share'] > 0.02:
        print('WARNING: the 8-bit file changes more than 2% of word decisions.')


def cmd_check8bit(args, p):
    out = os.path.join(p['results'], f'{args.name}_int8.json')
    if done(out, args, args.recipe):
        return
    from transformers import AutoTokenizer
    from odh import cpu
    _, _, test = data.splits(data.load_ragtruth(args.data_dir))
    sub = data.subset(test, args.n)
    folder = make_onnx(args, p, model_of(p, args.name, args.model))[0]
    tok = AutoTokenizer.from_pretrained(folder)
    full = all_spans(load_preds(os.path.join(p['preds'], f'{args.name}_test.json.gz'), sub))
    print(f'Running the 8-bit model of {args.name} on {len(sub)} test answers on the CPU.')
    t0 = time.time()
    int8_tokens = cpu.predict(cpu.session(os.path.join(folder, 'model_int8.onnx')), tok, sub, args.max_length)
    int8 = all_spans(int8_tokens)
    a, b = score(full, sub)['all'][2], score(int8, sub)['all'][2]
    result = {'name': args.name, 'recipe': args.recipe, 'answers': len(sub),
              'span_f1_full_precision': a, 'span_f1_int8': b,
              'change': round(b - a, 2), 'change_ci95': bootstrap(int8, sub, base=full),
              'seconds': round(time.time() - t0)}
    write_json(out, result)
    print(f'{args.name}: span F1 on {len(sub)} answers {a} -> {b} in 8-bit '
          f'(change {result["change"]:+.2f}, 95% interval {result["change_ci95"]}).')


def cmd_bench(args, p):
    out = os.path.join(p['results'], f'{args.name}_bench_{args.where}.json')
    if done(out, args, args.recipe):
        return
    from transformers import AutoTokenizer
    from odh import cpu
    _, _, test = data.splits(data.load_ragtruth(args.data_dir))
    sub = data.subset(test, args.n)
    folder = make_onnx(args, p, model_of(p, args.name, args.model))[0]
    tok = AutoTokenizer.from_pretrained(folder)
    rows = []
    for kind in ('int8', 'fp32'):
        for threads in args.threads:
            r = cpu.time_answers(os.path.join(folder, f'model_{kind}.onnx'), tok, sub, threads, args.max_length)
            r['weights'] = kind
            rows.append(r)
            print(f'{args.name} {kind} {threads} thread(s): median {r["median_ms"]} ms, '
                  f'90th percentile {r["p90_ms"]} ms per answer ({r["mean_tokens"]} tokens on average); '
                  f'memory added by the model {r["model_mem_mb"]} MB')
    import onnxruntime
    write_json(out, {'name': args.name, 'recipe': args.recipe, 'where': args.where, 'cpu': cpu_name(),
                     'cores': os.cpu_count(), 'onnxruntime': onnxruntime.__version__, 'rows': rows})


def cpu_name():
    import platform
    try:
        with open('/proc/cpuinfo') as fh:
            text = fh.read()
        name = re.search(r'model name\s*:\s*(.+)', text)
        flags = re.search(r'flags\s*:\s*(.+)', text)
        extra = [f for f in ('avx512_vnni', 'avx_vnni', 'avx512f', 'avx2') if flags and f in flags.group(1).split()]
        return (name.group(1).strip() if name else platform.machine()) + (f' ({", ".join(extra)})' if extra else '')
    except OSError:
        return platform.processor() or platform.machine()


def teacher_tokens(p, teacher, tag):
    """{answer id: [(start, end, p)]} saved by "teach" for one set of answers."""
    path = os.path.join(p['preds'], f'{teacher}_{tag}.json.gz')
    if not os.path.exists(path):
        raise SystemExit(f'No teacher labels in {path}: run "teach --name {teacher}" first.')
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        record = json.load(fh)
    return dict(zip(record['ids'], record['tokens']))


def with_teacher(samples, labels, what):
    missing = [s['id'] for s in samples if s['id'] not in labels]
    if missing:
        raise SystemExit(f'{len(missing)} {what} have no teacher labels (e.g. {missing[0]}): run "teach" again.')
    return [{**s, 'teacher': labels[s['id']]} for s in samples]


def cmd_train(args, p):
    out = os.path.join(p['results'], f'{args.name}_training.json')
    if done(out, args):
        return
    from odh import distill
    from odh.model import train
    full_train, dev, _ = data.splits(data.load_ragtruth(args.data_dir))
    train_set, dev = data.subset(full_train, args.limit), data.subset(dev, args.limit and max(args.limit // 5, 20))
    human = len(train_set)
    extra_counts = {}
    if args.teacher:                         # teacher-student: soft labels, and new answers
        train_set = with_teacher(train_set, teacher_tokens(p, args.teacher, 'train'), 'training answers')
        sources = {s['source_id']: s for s in full_train}
        labeller = args.extra_teacher or args.teacher
        sets = {}
        for g in args.extra or []:
            extra, _ = distill.load_extra(os.path.join(p['extra'], g), g, sources)
            extra = data.subset(extra, args.limit)
            if not extra:
                raise SystemExit(f'No answers from {g}: run "generate --generator {g}" first.')
            labels = teacher_tokens(p, labeller, f'extra_{g}')
            sets[g] = [s for s in extra if s['id'] in labels]        # an LLM labeller may skip a few
            if len(sets[g]) < len(extra):
                print(f'{g}: {len(extra) - len(sets[g])} answers have no labels from {labeller} and are left out.')
        total = sum(len(v) for v in sets.values())
        if args.extra_max and total > args.extra_max:            # same number of new answers in compared runs
            sets = {g: data.subset(v, round(args.extra_max * len(v) / total)) for g, v in sets.items()}
        for g, extra in sets.items():
            labels = teacher_tokens(p, labeller, f'extra_{g}')
            if args.extra_hard:
                labels = {k: distill.harden(v) for k, v in labels.items()}
            train_set += with_teacher(extra, labels, f'answers of {g}')
            extra_counts[g] = len(extra)
    folder = os.path.join(p['models'], args.name)
    t0 = time.time()
    res = train(args.base, train_set, dev, folder, epochs=args.epochs, lr=args.lr, max_length=args.max_length,
                batch_tokens=args.batch_tokens, seed=args.seed, alpha=args.alpha if args.teacher else 1.0,
                device=device())
    write_json(out, {'name': args.name, 'base': args.base, 'epochs': args.epochs, 'lr': args.lr,
                     'seed': args.seed, 'teacher': args.teacher, 'alpha': args.alpha if args.teacher else 1.0,
                     'extra_teacher': (args.extra_teacher or args.teacher) if extra_counts else None,
                     'extra_hard': bool(args.extra_hard and extra_counts), 'extra': extra_counts, 'train_answers': human, 'extra_answers': sum(extra_counts.values()),
                     'dev_answers': len(dev), 'seconds': round(time.time() - t0), **res})
    print(f'Saved the best epoch ({res["best_epoch"]}) to {folder}: dev span F1 {res["dev_f1_05"]} at 0.5, '
          f'{res["dev_f1_at_cutoff"]} at the dev cut-off {res["dev_cutoff"]}. '
          f'Next: python run.py score --name {args.name}')


def cmd_generate(args, p):
    """New answers from a small (phone-sized) model to RAGTruth's training prompts (odh/distill.py).
    No person labels them; the teacher does ("teach")."""
    import math
    from odh import distill
    gen = args.generator
    hf_id = args.model or distill.ANSWER_MODELS.get(gen)
    if hf_id is None:
        raise SystemExit(f'Unknown answering model {gen}; known: {", ".join(distill.ANSWER_MODELS)} (or pass --model).')
    four_bit = gen in distill.LARGE or args.four_bit
    full_train, _, _ = data.splits(data.load_ragtruth(args.data_dir))
    items = data.subset(distill.extra_prompts(full_train), args.limit)
    folder = os.path.join(p['extra'], gen if args.limit is None else f'{gen}_limit{args.limit}')
    out = os.path.join(p['results'], f'extra_{gen}.json' if args.limit is None else f'extra_{gen}_limit{args.limit}.json')
    chunks = math.ceil(len(items) / 64)
    have = len([f for f in os.listdir(folder) if f.endswith('.json.gz')]) if os.path.isdir(folder) else 0
    if have >= chunks and os.path.exists(out) and not args.redo:
        print(f'Already done: {folder} ({have} chunk files).')
        return
    seconds = 0.0
    if have < chunks:
        try:
            tok, model = distill.load_small_generator(hf_id, device(), four_bit=four_bit)
        except Exception as err:                       # not found, or no access (licence not accepted)
            raise SystemExit(f'Could not load {hf_id}: {type(err).__name__}: {str(err)[:300]}')
        print(f'Writing answers with {gen} ({hf_id}{", 4-bit" if four_bit else ""}) to {len(items)} training prompts.')
        seconds = distill.generate_all(model, tok, items, folder, max_new_tokens=args.max_new_tokens)
    sources = {s['source_id']: s for s in full_train}
    extra, dropped = distill.load_extra(folder, gen, sources)
    words = [len(s['answer'].split()) for s in extra]
    write_json(out, {'generator': gen, 'hf_id': hf_id, 'four_bit': four_bit, 'prompts': len(items), 'kept': len(extra),
                     'dropped_unfinished_or_empty': dropped, 'mean_words': round(sum(words) / max(len(words), 1), 1),
                     'max_new_tokens': args.max_new_tokens, 'seconds': round(seconds)})
    print(f'{gen}: {len(extra)} answers kept, {dropped} left out (unfinished or empty); '
          f'{sum(words) / max(len(words), 1):.0f} words on average.')


def cmd_teach(args, p):
    """The teacher's word probabilities for RAGTruth's training answers and for every set of new
    answers found in <work-dir>/extra (or --generators). Saved in preds/<teacher>_<set>.json.gz."""
    from odh import distill
    from odh.model import load, predict
    full_train, _, _ = data.splits(data.load_ragtruth(args.data_dir))
    jobs = [('train', data.subset(full_train, args.limit))]
    sources = {s['source_id']: s for s in full_train}
    gens = args.generators or sorted(d for d in os.listdir(p['extra'])
                                     if os.path.isdir(os.path.join(p['extra'], d)) and '_limit' not in d)
    for g in gens:
        extra, _ = distill.load_extra(os.path.join(p['extra'], g), g, sources)
        if extra:
            jobs.append((f'extra_{g}', data.subset(extra, args.limit)))
    todo = []
    for tag, samples in jobs:
        path = os.path.join(p['preds'], f'{args.name}_{tag}.json.gz')
        if os.path.exists(path) and not args.redo:
            with gzip.open(path, 'rt', encoding='utf-8') as fh:
                have = set(json.load(fh)['ids'])
            if {s['id'] for s in samples} <= have:
                print(f'Already done: {path}')
                continue
        todo.append((path, tag, samples))
    if not todo:
        return
    name = model_of(p, args.name, args.model)
    tok, model = load(name, device())
    for path, tag, samples in todo:
        t0 = time.time()
        print(f'Teacher {name} on {len(samples)} answers ({tag}).')
        tokens = predict(model, tok, samples, args.max_length, log=print)
        save_preds(path, [s['id'] for s in samples], tokens)
        print(f'  saved {path} ({time.time() - t0:.0f}s)')


def cmd_distill(args, p):
    """Table of the teacher-student test (results/distill.md, distill.json): every trained run
    next to the human-labels-only run of the same size, the published TinyLettuce of that size
    and the teacher, on the same test answers."""
    _, _, test = data.splits(data.load_ragtruth(args.data_dir))
    res = p['results']
    runs = []
    for f in sorted(os.listdir(res)):
        if f.endswith('_training.json'):
            t = read_json(os.path.join(res, f))
            if os.path.exists(os.path.join(p['preds'], f'{t["name"]}_test.json.gz')) and 'dev_cutoff' in t:
                if not args.names or t['name'] in args.names:
                    runs.append(t)
    if not runs:
        raise SystemExit('No trained and scored runs yet: run "train" and then "score" first.')

    def size_of(base):
        m = re.search(r'(\d+)m\b', base.lower())
        return m.group(1) + 'm' if m else 'other'

    names = {t['name'] for t in runs}
    published = {f'tiny{size_of(t["base"])}' for t in runs} | {'ld_large'}
    present = [n for n in sorted(published) if os.path.exists(os.path.join(p['preds'], f'{n}_test.json.gz'))]
    ids = None
    for n in sorted(names) + present:                           # answers that every model has scored
        with gzip.open(os.path.join(p['preds'], f'{n}_test.json.gz'), 'rt', encoding='utf-8') as fh:
            got = set(json.load(fh)['ids'])
        ids = got if ids is None else ids & got
    test = [s for s in test if s['id'] in ids]
    tokens = {n: load_preds(os.path.join(p['preds'], f'{n}_test.json.gz'), test) for n in sorted(names) + present}

    def summary(n, cut):
        sp = all_spans(tokens[n], cut)
        s = score(sp, test)
        return sp, {'f1': s['all'][2], 'p': s['all'][0], 'r': s['all'][1], 'ci': bootstrap(sp, test),
                    'tasks': {t: v[2] for t, v in s.items() if t != 'all'}}

    def describe(t):
        if not t['teacher']:
            return 'human labels'
        text = f'human + teacher (alpha {t["alpha"]})'
        if t['extra_answers']:
            text += f' + {t["extra_answers"]:,} new answers ({", ".join(t["extra"])})'
        return text

    def updates(t):                         # answers seen in training: epochs x (human + new answers)
        return t['epochs'] * (t['train_answers'] + t['extra_answers'])

    def control_of(t, group):
        """The human-labels-only run of the same size that saw about as many answers in training."""
        hard = [c for c in group if not c['teacher']]
        if not t['teacher'] or not hard:
            return None
        return min(hard, key=lambda c: (abs(math.log(updates(t) / updates(c))), c['name']))

    def cells(s):
        return f'{s["tasks"].get("qa", "")} | {s["tasks"].get("summary", "")} | {s["tasks"].get("data2text", "")}'

    out, lines = {'answers': len(test), 'rows': []}, []
    head = ('| Model | Learned from | Epochs | Span F1 at 0.5 [95% CI] | Span F1 at dev cut-off | QA | Summary | Data2text '
            '| Compared with | Change, 0.5 [CI] | Change, dev cut-off [CI] | vs TinyLettuce, 0.5 [CI] |')
    lines += [head, '|' + '---|' * 12]
    if 'ld_large' in present:
        _, s = summary('ld_large', 0.5)
        out['rows'].append({'name': 'ld_large', 'role': 'teacher', **s})
        lines.append(f'| LettuceDetect-large (teacher) | human labels (published) |  | {s["f1"]} {list(s["ci"])} |  | '
                     f'{cells(s)} |  |  |  |  |')
    for size in sorted({size_of(t['base']) for t in runs}, key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 0):
        group = [t for t in runs if size_of(t['base']) == size]
        group.sort(key=lambda t: (bool(t['teacher']), t['extra_answers'], t['epochs'], t['name']))
        tiny = f'tiny{size}' if f'tiny{size}' in present else None
        tiny_sp = None
        if tiny:
            tiny_sp, s = summary(tiny, 0.5)
            out['rows'].append({'name': tiny, 'role': 'published', **s})
            lines.append(f'| TinyLettuce {size.upper()} (published) | human + 3k synthetic | 5 | {s["f1"]} {list(s["ci"])} |  | '
                         f'{cells(s)} |  |  |  |  |')
        for t in group:
            sp05, s05 = summary(t['name'], 0.5)
            spc, sc = summary(t['name'], t['dev_cutoff'])
            row = {'name': t['name'], 'role': 'trained', 'learned_from': describe(t), 'epochs': t['epochs'],
                   'answers_seen': updates(t), **s05,
                   'dev_cutoff': t['dev_cutoff'], 'f1_dev_cutoff': sc['f1'], 'ci_dev_cutoff': sc['ci']}
            g05 = gcut = gtiny = cname = ''
            c = control_of(t, group)
            if c:
                c05, ccut = all_spans(tokens[c['name']], 0.5), all_spans(tokens[c['name']], c['dev_cutoff'])
                cname = row['control'] = c['name']
                row['vs_control_05'] = round(s05['f1'] - score(c05, test)['all'][2], 2)
                row['vs_control_05_ci'] = bootstrap(sp05, test, base=c05)
                row['vs_control_cut'] = round(sc['f1'] - score(ccut, test)['all'][2], 2)
                row['vs_control_cut_ci'] = bootstrap(spc, test, base=ccut)
                g05 = f'{row["vs_control_05"]:+.2f} {list(row["vs_control_05_ci"])}'
                gcut = f'{row["vs_control_cut"]:+.2f} {list(row["vs_control_cut_ci"])}'
            if tiny_sp is not None:
                row['vs_tiny_05'] = round(s05['f1'] - score(tiny_sp, test)['all'][2], 2)
                row['vs_tiny_05_ci'] = bootstrap(sp05, test, base=tiny_sp)
                gtiny = f'{row["vs_tiny_05"]:+.2f} {list(row["vs_tiny_05_ci"])}'
            out['rows'].append(row)
            lines.append(f'| {t["name"]} | {describe(t)} | {t["epochs"]} | {s05["f1"]} {list(s05["ci"])} | '
                         f'{sc["f1"]} (cut-off {t["dev_cutoff"]}) | {cells(s05)} | {cname} | {g05} | {gcut} | {gtiny} |')
    lines.append(f'\nSpan F1 on {len(test)} RAGTruth test answers. Dev cut-off: the cut-off with the best span F1 on '
                 'the dev answers (held-out training sources), chosen for each trained run. QA, Summary, Data2text: '
                 'span F1 at 0.5. "Compared with": the human-labels-only run of the same size that saw about as many '
                 'answers in training (epochs x answers), so extra training alone cannot explain a gain. Changes are '
                 'in points, with 95% intervals from resampling test sources.')
    md = '\n'.join(lines)
    write_json(os.path.join(res, 'distill.json'), out)
    with open(os.path.join(res, 'distill.md'), 'w', encoding='utf-8') as fh:
        fh.write(md + '\n')
    print(md)


def extra_sets(p):
    """Names of the answering models whose new answers are in <work-dir>/extra."""
    return sorted(d for d in os.listdir(p['extra']) if os.path.isdir(os.path.join(p['extra'], d)) and '_limit' not in d)


def cmd_llmlabel(args, p):
    """A large language model labels answers the PsiloQA way (odh/llmlabel.py): the new answers of
    the answering models (--on extra, for training), or a fixed subset of RAGTruth test answers
    (--on test, to measure the labeller against people). Labels are saved like a teacher's:
    preds/<name>_extra_<answering model>.json.gz, or preds/<name>_test.json.gz."""
    from odh import distill, llmlabel
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f'No API key: put it in the environment variable {args.api_key_env} '
                         '(on Colab: the Secrets panel, then the notebook copies it).')
    full_train, _, test = data.splits(data.load_ragtruth(args.data_dir))

    def ask(sample):
        return llmlabel.call(args.llm_model, llmlabel.messages(sample['prompt'], sample['answer']), args.api_base, key)

    if args.on == 'test':
        jobs = [('test', data.subset(test, args.limit or 300))]
    else:
        sources = {s['source_id']: s for s in full_train}
        jobs = []
        for g in args.generators or extra_sets(p):
            extra, _ = distill.load_extra(os.path.join(p['extra'], g), g, sources)
            if extra:
                jobs.append((f'extra_{g}', data.subset(extra, args.limit)))
    for tag, samples in jobs:
        folder = os.path.join(p['llm'], args.name, tag)
        print(f'{args.llm_model} labels {len(samples)} answers ({tag}).')
        t_in, t_out = llmlabel.label_all(samples, folder, ask, workers=args.workers)
        replies = llmlabel.load_replies(folder)
        ids, tokens, found, unusable = [], [], {}, 0
        for s in samples:
            spans = llmlabel.parse(replies.get(s['id']), s['answer'])
            if spans is None:
                unusable += 1
                continue
            ids.append(s['id'])
            tokens.append(distill.spans_to_tokens(s['answer'], spans))
            found[s['id']] = spans
        save_preds(os.path.join(p['preds'], f'{args.name}_{tag}.json.gz'), ids, tokens)
        info = {'labeller': args.name, 'llm_model': args.llm_model, 'api_base': args.api_base, 'set': tag,
                'answers': len(samples), 'labelled': len(ids), 'unusable_replies': unusable,
                'answers_with_spans': sum(1 for v in found.values() if v),
                'tokens_in_this_run': t_in, 'tokens_out_this_run': t_out}
        if tag == 'test':                     # how good is the labeller, against people and our teacher?
            kept = [s for s in samples if s['id'] in found]
            spans = [found[s['id']] for s in kept]
            sc = score(spans, kept)
            info['labeller_vs_people'] = {'span_f1': sc['all'][2], 'precision': sc['all'][0], 'recall': sc['all'][1],
                                          'ci95': bootstrap(spans, kept), 'by_task': {t: v[2] for t, v in sc.items() if t != 'all'}}
            teacher = os.path.join(p['preds'], 'ld_large_test.json.gz')
            if os.path.exists(teacher):
                tspans = all_spans(load_preds(teacher, kept))
                tsc = score(tspans, kept)
                info['teacher_vs_people_same_answers'] = {'span_f1': tsc['all'][2], 'ci95': bootstrap(tspans, kept),
                                                          'by_task': {t: v[2] for t, v in tsc.items() if t != 'all'}}
                info['labeller_minus_teacher_ci95'] = bootstrap(spans, kept, base=tspans)
        write_json(os.path.join(p['results'], f'llmlabel_{args.name}_{tag}.json'), info)
        line = f'{tag}: {len(ids)} of {len(samples)} answers labelled, {unusable} replies unusable'
        if 'labeller_vs_people' in info:
            line += (f'; span F1 against people {info["labeller_vs_people"]["span_f1"]}'
                     + (f' (our teacher on the same answers: {info["teacher_vs_people_same_answers"]["span_f1"]})'
                        if 'teacher_vs_people_same_answers' in info else ''))
        print(line)


def psilo_gold_path(p):
    return os.path.join(p['preds'], 'psilo_gold.json.gz')


def load_psilo_gold(p):
    with gzip.open(psilo_gold_path(p), 'rt', encoding='utf-8') as fh:
        return json.load(fh)


def cmd_psilo(args, p):
    """Score checkers on PsiloQA's English answers (never used for training), by the size of the model
    that wrote each answer (odh/psilo.py). Writes results/psilo.md."""
    from odh import psilo
    if os.path.exists(psilo_gold_path(p)) and not args.redo:
        samples = None
        gold = load_psilo_gold(p)
    else:
        samples = data.subset(psilo.load(), args.limit or 4000)
        gold = [{k: s[k] for k in ('id', 'source_id', 'task', 'model', 'size_group', 'spans')} for s in samples]
        with gzip.open(psilo_gold_path(p), 'wt', encoding='utf-8') as fh:
            json.dump(gold, fh)
    trained = [f[:-len('_training.json')] for f in sorted(os.listdir(p['results'])) if f.endswith('_training.json')]
    names = args.names or [n for n in ['tiny17m', 'tiny68m', 'ld_base', 'ld_large'] + trained
                           if os.path.exists(os.path.join(p['preds'], f'{n}_test.json.gz'))]
    for name in names:
        out = os.path.join(p['preds'], f'{name}_psilo.json.gz')
        if os.path.exists(out) and not args.redo:
            continue
        if samples is None:
            by_id = {s['id']: s for s in psilo.load()}
            samples = [by_id[g['id']] for g in gold if g['id'] in by_id]
        from odh.model import load, predict
        model = model_of(p, name)
        print(f'Scoring {name} ({model}) on {len(samples)} PsiloQA answers.')
        tok, net = load(model, device())
        save_preds(out, [s['id'] for s in samples], predict(net, tok, samples, args.max_length))
    for g in gold:
        g['spans'] = [tuple(x) for x in g['spans']]
    groups = ['all'] + psilo.BUCKETS
    lines = ['| Checker | ' + ' | '.join(f'{g} (n={sum(1 for x in gold if g == "all" or x["size_group"] == g)})'
                                       for g in groups) + ' |', '|---|' + '---|' * len(groups)]
    out = {'answers': len(gold), 'checkers': {}}
    for name in names:
        path = os.path.join(p['preds'], f'{name}_psilo.json.gz')
        if not os.path.exists(path):
            continue
        tokens = load_preds(path, gold)
        row, cells = {}, []
        for g in groups:
            idx = [i for i, x in enumerate(gold) if g == 'all' or x['size_group'] == g]
            sub = [gold[i] for i in idx]
            sp = all_spans([tokens[i] for i in idx])
            row[g] = {'span_f1': score(sp, sub)['all'][2], 'ci95': bootstrap(sp, sub)}
            cells.append(f'{row[g]["span_f1"]} {list(row[g]["ci95"])}')
        out['checkers'][name] = row
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines.append('\nSpan F1 (cut-off 0.5) on English PsiloQA answers, grouped by the size of the model that wrote '
                 'the answer; spans marked by GPT-4o (PsiloQA). 95% intervals resample Wikipedia articles.')
    md = '\n'.join(lines)
    write_json(os.path.join(p['results'], 'psilo.json'), out)
    with open(os.path.join(p['results'], 'psilo.md'), 'w', encoding='utf-8') as fh:
        fh.write(md + '\n')
    print(md)


def cmd_compare(args, p):
    """Pairs of runs side by side (A:B): B minus A with its 95% interval, on the RAGTruth test set
    (at 0.5 and at each run's dev cut-off) and, when both were scored, on PsiloQA by answer-model
    size. Writes results/compare.md."""
    from odh import psilo
    _, _, test = data.splits(data.load_ragtruth(args.data_dir))
    gold = load_psilo_gold(p) if os.path.exists(psilo_gold_path(p)) else None
    if gold:
        for g in gold:
            g['spans'] = [tuple(x) for x in g['spans']]

    def cutoff(name):
        path = os.path.join(p['results'], f'{name}_training.json')
        return read_json(path).get('dev_cutoff', 0.5) if os.path.exists(path) else 0.5

    def diff(sa, sb, samples):
        fa, fb = score(sa, samples)['all'][2], score(sb, samples)['all'][2]
        return f'{fa} → {fb} ({fb - fa:+.2f} {list(bootstrap(sb, samples, base=sa))})'

    groups = ['all'] + psilo.BUCKETS
    head = ['A → B', 'RAGTruth, cut-off 0.5', 'RAGTruth, dev cut-offs'] + [f'PsiloQA {g}' for g in groups]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for pair in args.pairs:
        a, b = pair.split(':')
        pa, pb = (os.path.join(p['preds'], f'{n}_test.json.gz') for n in (a, b))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'Skipped {pair}: score both runs first.')
            continue
        with gzip.open(pa, 'rt', encoding='utf-8') as fh:
            ids = set(json.load(fh)['ids'])
        with gzip.open(pb, 'rt', encoding='utf-8') as fh:
            ids &= set(json.load(fh)['ids'])
        sub = [s for s in test if s['id'] in ids]
        ta, tb = load_preds(pa, sub), load_preds(pb, sub)
        cells = [f'{a} → {b}', diff(all_spans(ta), all_spans(tb), sub),
                 diff(all_spans(ta, cutoff(a)), all_spans(tb, cutoff(b)), sub)]
        qa, qb = (os.path.join(p['preds'], f'{n}_psilo.json.gz') for n in (a, b))
        both = bool(gold) and os.path.exists(qa) and os.path.exists(qb)
        if both:
            pa_t, pb_t = load_preds(qa, gold), load_preds(qb, gold)
        for g in groups:
            if both:
                idx = [i for i, x in enumerate(gold) if g == 'all' or x['size_group'] == g]
                s2 = [gold[i] for i in idx]
                cells.append(diff(all_spans([pa_t[i] for i in idx]), all_spans([pb_t[i] for i in idx]), s2))
            else:
                cells.append('')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('\nEach cell: span F1 of A → span F1 of B (B minus A, 95% interval from resampling sources). '
                 'PsiloQA columns: cut-off 0.5, grouped by the size of the model that wrote the answer.')
    md = '\n'.join(lines)
    with open(os.path.join(p['results'], 'compare.md'), 'w', encoding='utf-8') as fh:
        fh.write(md + '\n')
    print(md)


def cmd_signals(args, p):
    """Re-read the answers of one open answering model and record its signals (odh/signals.py)."""
    import math
    from odh import signals
    gen = args.generator
    candidates = [args.model] if args.model else signals.GENERATORS[gen]
    samples = [s for s in data.load_ragtruth(args.data_dir) if s['model'] == gen and s['split'] == args.split]
    samples = data.subset(samples, args.limit)
    folder = os.path.join(p['signals'], gen, args.split)
    chunks = math.ceil(len(samples) / 25)
    have = len([f for f in os.listdir(folder) if f.endswith('.pkl.gz')]) if os.path.isdir(folder) else 0
    out = os.path.join(p['results'], f'signals_{gen}_{args.split}.json')
    if have >= chunks and os.path.exists(out) and not args.redo:
        print(f'Already done: {folder} ({have} chunk files).')
        return
    t0 = time.time()
    tok = model = None
    for hf_id in candidates:
        try:
            tok, model = signals.load_generator(hf_id, four_bit=not args.no_4bit, device=device())
            break
        except Exception as err:                       # not found, or no access (licence not accepted)
            print(f'Could not load {hf_id}: {type(err).__name__}: {str(err)[:300]}')
    if model is None:
        raise SystemExit(f'No weights available for {gen}; skipped. See the notebook for the Hugging Face token.')
    print(f'Recording the signals of {gen} ({hf_id}) for {len(samples)} {args.split} answers.')
    skipped = signals.record_all(model, tok, samples, folder)
    write_json(out, {'generator': gen, 'hf_id': hf_id, 'split': args.split, 'answers': len(samples),
                     'chat_template': bool(getattr(tok, 'chat_template', None)),
                     'skipped_too_long': skipped, 'four_bit': not args.no_4bit,
                     'layers': len(signals._attention_modules(model)), 'seconds': round(time.time() - t0)})
    print(f'Saved to {folder}; {skipped} answers skipped (longer than 4,096 tokens).')


def cmd_fusion(args, p):
    """Do the answering model's signals add to text checkers? (odh/fusion.py)"""
    from odh import fusion, signals
    test = [s for s in data.load_ragtruth(args.data_dir) if s['split'] == 'test']
    records = {}
    for g in args.generators or list(signals.GENERATORS):
        records.update(signals.load_records(os.path.join(p['signals'], g, 'test')))
    samples = [s for s in test if s['id'] in records]
    if not samples:
        raise SystemExit('No signals found: run the "signals" step first.')
    samples = data.subset(samples, args.limit)
    names = [n for n in (args.text or ['tiny17m', 'tiny32m', 'tiny68m', 'ld_base'])
             if os.path.exists(os.path.join(p['preds'], f'{n}_test.json.gz'))]
    if not names:
        raise SystemExit('No text-checker predictions found: run the "score" step first.')
    for n in names:                               # keep answers that every checker has scored
        with gzip.open(os.path.join(p['preds'], f'{n}_test.json.gz'), 'rt', encoding='utf-8') as fh:
            scored = set(json.load(fh)['ids'])
        samples = [s for s in samples if s['id'] in scored]
    text_tokens = {n: load_preds(os.path.join(p['preds'], f'{n}_test.json.gz'), samples) for n in names}
    print(f'{len(samples)} test answers with signals from '
          f'{", ".join(sorted({s["model"] for s in samples}))}; text checkers: {", ".join(names)}.')
    t0 = time.time()
    results = fusion.run(samples, text_tokens, records, folds=args.folds)
    results['setup'] = {'answers': len(samples), 'models': sorted({s['model'] for s in samples}),
                        'folds': args.folds, 'seconds': round(time.time() - t0)}
    write_json(os.path.join(p['results'], 'fusion.json'), results)
    md = fusion.table(results)
    with open(os.path.join(p['results'], 'fusion.md'), 'w', encoding='utf-8') as fh:
        fh.write(md + '\n')
    print(md)


def cmd_table(args, p):
    res = p['results']
    names = [n for n, _, _ in LADDER] + sorted(
        f[:-len('_score.json')] for f in os.listdir(res) if f.endswith('_score.json')
        and f[:-len('_score.json')] not in {n for n, _, _ in LADDER})
    labels = {n: lab for n, lab, _ in LADDER}
    rows = []
    for n in names:
        path = os.path.join(res, f'{n}_score.json')
        if not os.path.exists(path):
            continue
        s = read_json(path)
        e = read_json(os.path.join(res, f'{n}_export.json')) if os.path.exists(os.path.join(res, f'{n}_export.json')) else {}
        q = read_json(os.path.join(res, f'{n}_int8.json')) if os.path.exists(os.path.join(res, f'{n}_int8.json')) else {}
        bench = {}
        for f in sorted(os.listdir(res)):
            if f.startswith(f'{n}_bench_') and f.endswith('.json'):
                b = read_json(os.path.join(res, f))
                for r in b['rows']:
                    if r['weights'] == 'int8':
                        bench[f'{b["where"]}, {r["threads"]} thr'] = r['median_ms']
        rows.append({'name': n, 'label': labels.get(n, n), 'params_m': round(s['params'] / 1e6, 1 if s['params'] < 1e7 else None),
                     'int8_mb': e.get('int8_mb'), 'span_f1': s['span_f1'], 'ci': s['f1_ci95'],
                     'qa': s['by_task'].get('qa', {}).get('span_f1'),
                     'summary': s['by_task'].get('summary', {}).get('span_f1'),
                     'data2text': s['by_task'].get('data2text', {}).get('span_f1'),
                     'int8_change': q.get('change'), 'int8_recipe': q.get('recipe', 'dyn'), 'bench': bench})
    if not rows:
        print('No results yet.')
        return
    timing_cols = sorted({k for r in rows for k in r['bench']})
    head = ['Model', 'Parameters', '8-bit size (MB)', 'Span F1 [95% CI]', 'QA', 'Summary', 'Data2text',
            '8-bit change'] + [f'ms/answer ({c})' for c in timing_cols]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        change = '' if r['int8_change'] is None else f'{r["int8_change"]:+.2f} ({r["int8_recipe"]})'
        cells = [r['label'], f'{r["params_m"]}M', r['int8_mb'] or '', f'{r["span_f1"]} [{r["ci"][0]}, {r["ci"][1]}]',
                 r['qa'], r['summary'], r['data2text'], change] + [r['bench'].get(c, '') for c in timing_cols]
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
    lines.append('\n8-bit change: span F1 of the 8-bit file minus full precision on the same fixed test '
                 'answers; the recipe that made the 8-bit file is in brackets (see odh/cpu.py). '
                 'Times are for the 8-bit file.')
    table = '\n'.join(lines)
    with open(os.path.join(res, 'table.md'), 'w', encoding='utf-8') as fh:
        fh.write(table + '\n')
    print(table)
    plotted = [r for r in rows if r['int8_mb']]
    if plotted:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        sizes = [r['int8_mb'] for r in plotted]
        left, right = min(sizes) / 2, max(max(sizes) * 5, 300)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.axvspan(left, 150, color='tab:blue', alpha=0.08, label='Phone budget: 150 MB or less')
        ax.scatter([r['int8_mb'] for r in plotted], [r['span_f1'] for r in plotted], color='0.35', zorder=3)
        for r in plotted:
            ax.annotate(f'{r["label"]} ({r["span_f1"]})', (r['int8_mb'], r['span_f1']),
                        textcoords='offset points', xytext=(6, 4), fontsize=8)
        ax.set_xscale('log')
        ax.set_xlim(left, right)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.margins(y=0.15)
        ax.set_xlabel('Model size in 8-bit (MB, log scale)')
        ax.set_ylabel('Span F1 on RAGTruth test (word level)')
        ax.set_title('Word-level accuracy against model size')
        ax.legend(loc='lower right', fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(res, 'ladder.png'), dpi=150)
        print(f'Chart saved to {os.path.join(res, "ladder.png")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['models', 'score', 'export', 'check8bit', 'bench', 'train', 'table',
                                     'signals', 'fusion', 'teach', 'generate', 'distill', 'llmlabel', 'psilo',
                                     'compare'])
    ap.add_argument('--name', help='run name, e.g. tiny68m')
    ap.add_argument('--model', help='Hugging Face id or folder (default: from "models" or a trained run)')
    ap.add_argument('--base', help='train: the model to start from')
    ap.add_argument('--data-dir', default='RAGTruth')
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--onnx-dir', help='where ONNX files go (default: <work-dir>/onnx); they are remade when missing')
    ap.add_argument('--max-length', type=int, default=4096)
    ap.add_argument('--n', type=int, default=300, help='check8bit / bench: answers in the fixed test subset')
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 2], help='bench: CPU threads to try')
    ap.add_argument('--where', default='colab', help='bench: a label for this machine, e.g. laptop')
    ap.add_argument('--recipe', default='w8', choices=['w8', 'mix', 'dyn'],
                    help='how the 8-bit file is made (see odh/cpu.py); default w8')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-tokens', type=int, default=8192)
    ap.add_argument('--limit', type=int, help='use only about N answers (quick checks)')
    ap.add_argument('--generator', help='signals: RAGTruth name of the answering model, e.g. llama-2-7b-chat; '
                    'generate: a small model, e.g. qwen2.5-1.5b')
    ap.add_argument('--split', default='test', help='signals: which RAGTruth answers (test or train)')
    ap.add_argument('--no-4bit', action='store_true', help='signals: load the answering model in 16-bit')
    ap.add_argument('--generators', nargs='+', help='fusion: answering models to use; teach: sets of new '
                    'answers to label (default: all found)')
    ap.add_argument('--text', nargs='+', help='fusion: text checkers (default: tiny17m tiny32m tiny68m ld_base)')
    ap.add_argument('--folds', type=int, default=5, help='fusion: cross-validation folds')
    ap.add_argument('--teacher', help='train: run name of the teacher whose labels to learn from, e.g. ld_large')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='train with --teacher: weight of the human label (the teacher gets 1 - alpha)')
    ap.add_argument('--extra', nargs='+', help='train with --teacher: also learn from these small models\' answers')
    ap.add_argument('--max-new-tokens', type=int, default=450, help='generate: longest answer, in tokens')
    ap.add_argument('--names', nargs='+', help='distill, psilo: runs to show or score (default: all scored ones)')
    ap.add_argument('--extra-teacher', help='train: who labels the new answers (default: --teacher), e.g. gpt4o')
    ap.add_argument('--extra-hard', action='store_true', help='train: round the labels of the new answers to 0 or 1')
    ap.add_argument('--extra-max', type=int, help='train: use at most this many new answers (same share per model)')
    ap.add_argument('--four-bit', action='store_true', help='generate: load the answering model in 4-bit')
    ap.add_argument('--llm-model', default='gpt-4o', help='llmlabel: the model name the API expects')
    ap.add_argument('--api-base', default='https://api.openai.com/v1', help='llmlabel: OpenAI-compatible API address')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY', help='llmlabel: environment variable with the key')
    ap.add_argument('--workers', type=int, default=8, help='llmlabel: requests sent at the same time')
    ap.add_argument('--on', choices=['extra', 'test'], default='extra',
                    help='llmlabel: label the new answers (extra) or 300 RAGTruth test answers (test)')
    ap.add_argument('--pairs', nargs='+', help='compare: pairs of runs A:B, e.g. kd17m:kdlarge17m')
    ap.add_argument('--redo', action='store_true')
    args = ap.parse_args()
    if args.step not in ('models', 'table', 'signals', 'fusion', 'generate', 'distill', 'psilo', 'compare') \
            and not args.name:
        ap.error('--name is required')
    if args.step in ('signals', 'generate') and not args.generator:
        ap.error(f'{args.step} needs --generator')
    if args.step == 'compare' and not args.pairs:
        ap.error('compare needs --pairs A:B ...')
    if args.extra and not args.teacher:
        ap.error('--extra needs --teacher (the new answers have no human labels)')
    if args.name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
        ap.error('--name may use letters, digits, ".", "_" and "-"')
    if args.step == 'train' and not args.base:
        ap.error('train needs --base')
    p = paths(args.work_dir)
    if args.onnx_dir:
        p['onnx'] = args.onnx_dir
        os.makedirs(p['onnx'], exist_ok=True)
    {'models': cmd_models, 'score': cmd_score, 'export': cmd_export, 'check8bit': cmd_check8bit,
     'bench': cmd_bench, 'train': cmd_train, 'table': cmd_table, 'signals': cmd_signals,
     'fusion': cmd_fusion, 'teach': cmd_teach, 'generate': cmd_generate, 'distill': cmd_distill,
     'llmlabel': cmd_llmlabel, 'psilo': cmd_psilo, 'compare': cmd_compare}[args.step](args, p)


if __name__ == '__main__':
    main()
