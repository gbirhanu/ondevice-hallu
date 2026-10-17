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

Every command skips work whose result file already exists (add --redo to repeat it).
Results are small JSON files in --work-dir/results; nothing else is ever overwritten.
"""
import argparse
import gzip
import json
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
    p = {k: os.path.join(work, k) for k in ('results', 'preds', 'onnx', 'models', 'signals')}
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


def cmd_train(args, p):
    out = os.path.join(p['results'], f'{args.name}_training.json')
    if done(out, args):
        return
    from odh.model import train
    train_set, dev, _ = data.splits(data.load_ragtruth(args.data_dir))
    train_set, dev = data.subset(train_set, args.limit), data.subset(dev, args.limit and max(args.limit // 5, 20))
    folder = os.path.join(p['models'], args.name)
    history = train(args.base, train_set, dev, folder, epochs=args.epochs, lr=args.lr,
                    max_length=args.max_length, batch_tokens=args.batch_tokens, seed=args.seed,
                    device=device())
    write_json(out, {'name': args.name, 'base': args.base, 'epochs': args.epochs, 'lr': args.lr,
                     'seed': args.seed, 'train_answers': len(train_set), 'history': history})
    print(f'Saved the best epoch to {folder}. Next: python run.py score --name {args.name}')


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
                                     'signals', 'fusion'])
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
    ap.add_argument('--generator', help='signals: RAGTruth name of the answering model, e.g. llama-2-7b-chat')
    ap.add_argument('--split', default='test', help='signals: which RAGTruth answers (test or train)')
    ap.add_argument('--no-4bit', action='store_true', help='signals: load the answering model in 16-bit')
    ap.add_argument('--generators', nargs='+', help='fusion: answering models to use (default: all found)')
    ap.add_argument('--text', nargs='+', help='fusion: text checkers (default: tiny17m tiny32m tiny68m ld_base)')
    ap.add_argument('--folds', type=int, default=5, help='fusion: cross-validation folds')
    ap.add_argument('--redo', action='store_true')
    args = ap.parse_args()
    if args.step not in ('models', 'table', 'signals', 'fusion') and not args.name:
        ap.error('--name is required')
    if args.step == 'signals' and not args.generator:
        ap.error('signals needs --generator')
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
     'fusion': cmd_fusion}[args.step](args, p)


if __name__ == '__main__':
    main()
