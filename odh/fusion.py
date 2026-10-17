"""First test of the generator-aware checker: do the answering model's signals add to a
text checker's word-level scores?

For every checker token we put side by side: the text checker's probability (TinyLettuce
or LettuceDetect, as already predicted), the answering model's confidence and its
attention to the passages (odh.signals), each also averaged over neighbouring tokens. A
small gradient-boosted model learns the combination. Every version is trained and scored
the same way: 5-fold cross-validation on the test answers, folds split by source, so no
passage is in both training and scoring. Spans use the plain 0.5 cut-off.

Versions: text (the text score alone, re-learned the same way, so any gain is not just
recalibration), confidence, attention, confidence+attention, text+confidence,
text+attention, text+confidence+attention.
"""
import numpy as np

from odh.metrics import bootstrap, score
from odh.spans import all_spans

GROUPS = {'text': ['T'], 'confidence': ['C'], 'attention': ['A'], 'confidence+attention': ['C', 'A'],
          'text+confidence': ['T', 'C'], 'text+attention': ['T', 'A'],
          'text+confidence+attention': ['T', 'C', 'A']}
BINS = 4          # layers are pooled into 4 depth bins, so every answering model gives the same features


def _window(x, k, how='mean'):
    """Mean (or max/min) over tokens i-k..i+k, shrinking at the ends."""
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[max(0, i - k):i + k + 1]
        out[i] = seg.mean() if how == 'mean' else (seg.max() if how == 'max' else seg.min())
    return out


def _shift(x, d):
    out = np.empty_like(x)
    if d > 0:
        out[d:], out[:d] = x[:-d], x[0]
    elif d < 0:
        out[:d], out[d:] = x[-d:], x[-1]
    else:
        out[:] = x
    return out


def generator_features(rec):
    """Per answering-model token: 4 confidence numbers and 2 x BINS attention numbers."""
    conf = rec['conf'].astype(np.float32)
    look = rec['look'].astype(np.float32)                    # tokens x layers x (mean, min, max)
    bins = np.array_split(np.arange(look.shape[1]), BINS)
    att = np.concatenate([np.stack([look[:, b, k].mean(1) for b in bins], 1) for k in (0, 1)], 1)
    return conf, att


def align(offsets, gen_offsets, values):
    """values (one row per answering-model token) averaged onto the checker's tokens by
    character overlap; a checker token that overlaps none takes the nearest one."""
    s, e = offsets[:, 0][:, None], offsets[:, 1][:, None]
    gs, ge = gen_offsets[:, 0][None, :], gen_offsets[:, 1][None, :]
    m = ((ge > s) & (gs < e)).astype(np.float32)
    none = m.sum(1) == 0
    if none.any():
        nearest = np.abs(gs - s).argmin(1)
        m[np.where(none)[0], nearest[none]] = 1.0
    return (m / m.sum(1, keepdims=True)) @ values


def token_table(sample, tokens, rec):
    """Feature blocks (T, C, A, base), labels and offsets for one answer's checker tokens."""
    offsets = np.array([(t[0], t[1]) for t in tokens], dtype=np.float32)
    p = np.clip(np.array([t[2] for t in tokens], dtype=np.float32), 1e-5, 1 - 1e-5)
    n = len(p)
    conf_g, att_g = generator_features(rec)
    conf = align(offsets, rec['gen_offsets'].astype(np.float32), conf_g)
    att = align(offsets, rec['gen_offsets'].astype(np.float32), att_g)
    T = np.stack([p, np.log(p / (1 - p)), _shift(p, 1), _shift(p, -1), _shift(p, 2), _shift(p, -2),
                  _window(p, 3), _window(p, 3, 'max')], 1)
    lp, ent = conf[:, 0], conf[:, 1]
    C = np.concatenate([conf, np.stack([_window(lp, 3), _window(ent, 3), _window(lp, 3, 'min')], 1)], 1)
    look_all = att[:, :BINS].mean(1)
    A = np.concatenate([att, np.stack([_window(look_all, 3), _window(look_all, 3, 'min')], 1)], 1)
    base = np.stack([np.arange(n) / max(n - 1, 1), np.full(n, np.log(n + 1))], 1).astype(np.float32)
    y = np.array([1 if any(e > gs and s < ge for gs, ge in sample['spans']) else 0
                  for s, e in offsets], dtype=np.int8)
    return {'T': T, 'C': C, 'A': A, 'base': base}, y


def cross_validate(samples, blocks, labels, groups, folds=5, seed=0, max_iter=200):
    """Out-of-fold token probabilities for one feature version (list of blocks)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X = [np.concatenate([b[g] for g in groups] + [b['base']], 1) for b in blocks]
    sources = np.array([s['source_id'] for s in samples])
    probs = [None] * len(samples)
    for tr, te in GroupKFold(n_splits=folds).split(np.zeros(len(samples)), groups=sources):
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(np.concatenate([X[i] for i in tr]), np.concatenate([labels[i] for i in tr]))
        for i in te:
            probs[i] = clf.predict_proba(X[i])[:, 1]
    return probs


def _summary(preds, samples, base=None):
    s = score(preds, samples)
    out = {'span_f1': s['all'][2], 'precision': s['all'][0], 'recall': s['all'][1],
           'by_task': {t: v[2] for t, v in s.items() if t != 'all'},
           'by_model': {}}
    for m in sorted({x['model'] for x in samples}):
        idx = [i for i, x in enumerate(samples) if x['model'] == m]
        out['by_model'][m] = score([preds[i] for i in idx], [samples[i] for i in idx])['all'][2]
    if base is not None:
        out['gain_over_text_ci95'] = bootstrap(preds, samples, base=base)
    return out


def run(samples, text_tokens, records, versions=None, folds=5, log=print):
    """samples: test answers that have signals; text_tokens: {text checker: per-answer
    [(start, end, p)]}; records: {answer id: signals}. Returns {'signals_only': {...},
    '<text checker>': {...}}; gains are against the text-only version of the same checker."""
    versions = versions or list(GROUPS)
    with_text = [v for v in versions if 'T' in GROUPS[v]]
    if with_text and 'text' not in with_text:
        with_text = ['text'] + with_text
    results = {'signals_only': {}}
    for k, (name, toks) in enumerate(text_tokens.items()):
        blocks, labels = [], []
        for s, t in zip(samples, toks):
            b, y = token_table(s, t, records[s['id']])
            blocks.append(b)
            labels.append(y)
        if k == 0:                                     # versions without text are the same for every checker
            for v in versions:
                if 'T' not in GROUPS[v]:
                    probs = cross_validate(samples, blocks, labels, GROUPS[v], folds)
                    spans = all_spans([[(int(a), int(b), float(q)) for (a, b, _), q in zip(t, pr)]
                                       for t, pr in zip(toks, probs)])
                    results['signals_only'][v] = _summary(spans, samples)
                    log(f'  {"-":9s} {v:27s} span F1 {results["signals_only"][v]["span_f1"]:6.2f}')
        res = {'answers': len(samples), 'tokens': int(sum(len(y) for y in labels)),
               'as_published': _summary(all_spans(toks), samples), 'versions': {}}
        text_spans = None
        for v in with_text:
            probs = cross_validate(samples, blocks, labels, GROUPS[v], folds)
            spans = all_spans([[(int(a), int(b), float(q)) for (a, b, _), q in zip(t, pr)]
                               for t, pr in zip(toks, probs)])
            if v == 'text':
                text_spans = spans
            res['versions'][v] = _summary(spans, samples, None if v == 'text' else text_spans)
            log(f'  {name:9s} {v:27s} span F1 {res["versions"][v]["span_f1"]:6.2f}')
        results[name] = res
    return results


def table(results):
    """Markdown: one row per text checker and version, then the versions without text."""
    lines = ['| Text checker | Version | Span F1 | QA | Summary | Data2text | Gain over text only [95% CI] |',
             '|---|---|---|---|---|---|---|']
    for name, res in results.items():
        if not isinstance(res, dict) or 'as_published' not in res:      # signals_only, setup
            continue
        pub = res['as_published']
        lines.append(f'| {name} | as published (0.5 cut-off) | {pub["span_f1"]} | {pub["by_task"].get("qa", "")} | '
                     f'{pub["by_task"].get("summary", "")} | {pub["by_task"].get("data2text", "")} | |')
        text_f1 = res['versions'].get('text', {}).get('span_f1')
        for v, r in res['versions'].items():
            gain = ''
            if 'gain_over_text_ci95' in r and text_f1 is not None:
                gain = f'{r["span_f1"] - text_f1:+.2f} {list(r["gain_over_text_ci95"])}'
            lines.append(f'| {name} | {v} | {r["span_f1"]} | {r["by_task"].get("qa", "")} | '
                         f'{r["by_task"].get("summary", "")} | {r["by_task"].get("data2text", "")} | {gain} |')
    for v, r in results.get('signals_only', {}).items():
        lines.append(f'| (none) | {v} | {r["span_f1"]} | {r["by_task"].get("qa", "")} | '
                     f'{r["by_task"].get("summary", "")} | {r["by_task"].get("data2text", "")} | |')
    return '\n'.join(lines)
