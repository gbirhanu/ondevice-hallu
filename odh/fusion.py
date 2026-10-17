"""First test of the generator-aware checker: do the answering model's signals add to a
text checker's word-level scores?

For every checker token we put side by side: the text checker's probability (TinyLettuce
or LettuceDetect, as already predicted), the answering model's confidence and its
attention to the passages (odh.signals), each also averaged over neighbouring tokens. A
small gradient-boosted model learns the combination. Every version is trained and scored
the same way: 5-fold cross-validation on the test answers, folds split by source, so no
passage is in both training and scoring.

Each version is reported three ways:
  ap        token-level average precision: how well the scores rank hallucinated tokens
            above the rest, whatever the cut-off;
  f1_05     span F1 with the plain 0.5 cut-off;
  f1_tuned  span F1 with the cut-off that gives the best span F1 on the other four folds
            (the same rule for every version, including the text checker as published).

Versions: text, probability only (the checker's probability alone, no neighbours or
position, to see what re-learning alone costs); text (with neighbours and position);
confidence; attention; confidence+attention; text+confidence; text+attention;
text+confidence+attention.
"""
import numpy as np

from odh.metrics import bootstrap, score
from odh.spans import all_spans

GROUPS = {'text, probability only': ['P'], 'text': ['T'], 'confidence': ['C'], 'attention': ['A'],
          'confidence+attention': ['C', 'A'], 'text+confidence': ['T', 'C'], 'text+attention': ['T', 'A'],
          'text+confidence+attention': ['T', 'C', 'A']}
BINS = 4          # layers are pooled into 4 depth bins, so every answering model gives the same features
CUTOFFS = np.round(np.arange(0.05, 0.951, 0.05), 2)


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
    """Feature blocks (P, T, C, A, base) and labels for one answer's checker tokens."""
    offsets = np.array([(t[0], t[1]) for t in tokens], dtype=np.float32)
    p = np.clip(np.array([t[2] for t in tokens], dtype=np.float32), 1e-5, 1 - 1e-5)
    n = len(p)
    conf_g, att_g = generator_features(rec)
    conf = align(offsets, rec['gen_offsets'].astype(np.float32), conf_g)
    att = align(offsets, rec['gen_offsets'].astype(np.float32), att_g)
    P = np.stack([p, np.log(p / (1 - p))], 1)
    T = np.concatenate([P, np.stack([_shift(p, 1), _shift(p, -1), _shift(p, 2), _shift(p, -2),
                                     _window(p, 3), _window(p, 3, 'max')], 1)], 1)
    lp, ent = conf[:, 0], conf[:, 1]
    C = np.concatenate([conf, np.stack([_window(lp, 3), _window(ent, 3), _window(lp, 3, 'min')], 1)], 1)
    look_all = att[:, :BINS].mean(1)
    A = np.concatenate([att, np.stack([_window(look_all, 3), _window(look_all, 3, 'min')], 1)], 1)
    base = np.stack([np.arange(n) / max(n - 1, 1), np.full(n, np.log(n + 1))], 1).astype(np.float32)
    y = np.array([1 if any(e > gs and s < ge for gs, ge in sample['spans']) else 0
                  for s, e in offsets], dtype=np.int8)
    return {'P': P, 'T': T, 'C': C, 'A': A, 'base': base}, y


def folds_of(samples, folds=5):
    """Fold number of every answer; all answers about one source share a fold."""
    from sklearn.model_selection import GroupKFold
    fold = np.zeros(len(samples), dtype=int)
    sources = np.array([s['source_id'] for s in samples])
    for k, (_, te) in enumerate(GroupKFold(n_splits=folds).split(np.zeros(len(samples)), groups=sources)):
        fold[te] = k
    return fold


def cross_validate(blocks, labels, groups, fold, seed=0, max_iter=200):
    """Out-of-fold token probabilities for one version (its list of feature blocks)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    use_base = 'P' not in groups
    X = [np.concatenate([b[g] for g in groups] + ([b['base']] if use_base else []), 1) for b in blocks]
    probs = [None] * len(blocks)
    for k in sorted(set(fold)):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(np.concatenate([X[i] for i in tr]), np.concatenate([labels[i] for i in tr]))
        for i in te:
            probs[i] = clf.predict_proba(X[i])[:, 1]
    return probs


def _spans(toks, probs, cut):
    """Spans of every answer; cut = one cut-off, or one per answer."""
    cuts = np.broadcast_to(np.asarray(cut, dtype=float), (len(toks),))
    return [all_spans([[(int(a), int(b), float(q)) for (a, b, _), q in zip(t, pr)]], c)[0]
            for t, pr, c in zip(toks, probs, cuts)]


def tuned_spans(toks, probs, samples, fold):
    """For each fold, the cut-off with the best span F1 on the other folds, applied to it."""
    from odh.metrics import counts, prf
    by_cut = [_spans(toks, probs, c) for c in CUTOFFS]
    c = np.array([[counts(sp, s['spans']) for sp, s in zip(spans, samples)] for spans in by_cut],
                 dtype=np.float64)                                   # cut-offs x answers x 3
    cut_of_fold, chosen = {}, np.zeros(len(samples), dtype=int)
    for k in sorted(set(fold)):
        f1 = prf(c[:, fold != k].sum(1))[2]
        best = int(np.argmax(f1))
        cut_of_fold[int(k)] = float(CUTOFFS[best])
        chosen[fold == k] = best
    return [by_cut[chosen[i]][i] for i in range(len(samples))], cut_of_fold


def _summary(samples, labels, probs, spans05, spans_tuned, cut_of_fold, base=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, q = np.concatenate(labels), np.concatenate(probs)
    s05, st = score(spans05, samples), score(spans_tuned, samples)
    out = {'ap': round(100 * float(average_precision_score(y, q)), 2),
           'auroc': round(100 * float(roc_auc_score(y, q)), 2),
           'f1_05': s05['all'][2], 'p_05': s05['all'][0], 'r_05': s05['all'][1],
           'f1_tuned': st['all'][2], 'p_tuned': st['all'][0], 'r_tuned': st['all'][1],
           'cutoffs': cut_of_fold,
           'by_task_tuned': {t: v[2] for t, v in st.items() if t != 'all'}, 'by_model_tuned': {}}
    for m in sorted({x['model'] for x in samples}):
        idx = [i for i, x in enumerate(samples) if x['model'] == m]
        out['by_model_tuned'][m] = score([spans_tuned[i] for i in idx], [samples[i] for i in idx])['all'][2]
    if base is not None:
        out['gain_tuned_ci95'] = bootstrap(spans_tuned, samples, base=base[1])
        out['gain_05_ci95'] = bootstrap(spans05, samples, base=base[0])
    return out


def _version(samples, toks, labels, probs, fold, base=None):
    spans05 = _spans(toks, probs, 0.5)
    spans_tuned, cuts = tuned_spans(toks, probs, samples, fold)
    return _summary(samples, labels, probs, spans05, spans_tuned, cuts, base), (spans05, spans_tuned)


def run(samples, text_tokens, records, versions=None, folds=5, log=print):
    """samples: test answers that have signals; text_tokens: {text checker: per-answer
    [(start, end, p)]}; records: {answer id: signals}. Gains are against the 'text' version of
    the same checker, at 0.5 and with tuned cut-offs."""
    versions = versions or list(GROUPS)
    fold = folds_of(samples, folds)
    results = {'signals_only': {}}
    for k, (name, toks) in enumerate(text_tokens.items()):
        blocks, labels = [], []
        for s, t in zip(samples, toks):
            b, y = token_table(s, t, records[s['id']])
            blocks.append(b)
            labels.append(y)
        if k == 0:                                     # versions without text are the same for every checker
            for v in versions:
                if not set(GROUPS[v]) & {'T', 'P'}:
                    probs = cross_validate(blocks, labels, GROUPS[v], fold)
                    results['signals_only'][v] = _version(samples, toks, labels, probs, fold)[0]
                    log(f'  {"-":9s} {v:27s} AP {results["signals_only"][v]["ap"]:6.2f}')
        raw = [np.array([t[2] for t in tk]) for tk in toks]
        res = {'answers': len(samples), 'tokens': int(sum(len(y) for y in labels)), 'versions': {}}
        res['versions']['as published'] = _version(samples, toks, labels, raw, fold)[0]
        text_spans = None
        for v in [v for v in versions if set(GROUPS[v]) & {'T', 'P'}]:
            probs = cross_validate(blocks, labels, GROUPS[v], fold)
            summary, spans = _version(samples, toks, labels, probs, fold,
                                      None if v in ('text', 'text, probability only') else text_spans)
            if v == 'text':
                text_spans = spans
            res['versions'][v] = summary
            log(f'  {name:9s} {v:27s} AP {summary["ap"]:6.2f}  span F1 {summary["f1_05"]:6.2f} at 0.5, '
                f'{summary["f1_tuned"]:6.2f} tuned')
        results[name] = res
    return results


def table(results):
    """Markdown: one row per text checker and version, then the versions without text."""
    head = ('| Text checker | Version | AP (tokens) | Span F1 at 0.5 | Span F1, tuned cut-off | QA | Summary | '
            'Data2text | Gain over text, tuned [95% CI] |')
    lines = [head, '|---|---|---|---|---|---|---|---|---|']

    def row(name, v, r, text_f1):
        gain = ''
        if 'gain_tuned_ci95' in r and text_f1 is not None:
            gain = f'{r["f1_tuned"] - text_f1:+.2f} {list(r["gain_tuned_ci95"])}'
        t = r['by_task_tuned']
        return (f'| {name} | {v} | {r["ap"]} | {r["f1_05"]} | {r["f1_tuned"]} | {t.get("qa", "")} | '
                f'{t.get("summary", "")} | {t.get("data2text", "")} | {gain} |')

    for name, res in results.items():
        if not isinstance(res, dict) or 'versions' not in res:      # signals_only, setup
            continue
        text_f1 = res['versions'].get('text', {}).get('f1_tuned')
        for v, r in res['versions'].items():
            lines.append(row(name, v, r, text_f1))
    for v, r in results.get('signals_only', {}).items():
        lines.append(row('(none)', v, r, None))
    lines.append('\nAP: average precision of the token scores (cut-off free). Tuned: the cut-off with the best '
                 'span F1 on the other four folds. QA, Summary, Data2text: span F1, tuned.')
    return '\n'.join(lines)
