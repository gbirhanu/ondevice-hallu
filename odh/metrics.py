"""Span F1 at the character level, summed over the whole test set.

Computed exactly as LettuceDetect's evaluate_detector_char_level: the overlap is summed
over every (predicted, gold) span pair, so published numbers compare like with like.
"""
import numpy as np


def counts(pred, gold):
    """(overlap, predicted characters, gold characters) for one answer."""
    overlap = 0
    for ps, pe in pred:
        for gs, ge in gold:
            a, b = max(ps, gs), min(pe, ge)
            if b > a:
                overlap += b - a
    return overlap, sum(e - s for s, e in pred), sum(e - s for s, e in gold)


def prf(c):
    o, p, g = c[..., 0], c[..., 1], c[..., 2]
    prec = np.where(p > 0, o / np.maximum(p, 1), 0.0)
    rec = np.where(g > 0, o / np.maximum(g, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return prec, rec, f1


def _counts(preds, samples):
    if len(preds) != len(samples):
        raise ValueError(f'{len(preds)} predictions for {len(samples)} answers')
    return np.array([counts(p, s['spans']) for p, s in zip(preds, samples)],
                    dtype=np.float64).reshape(-1, 3)


def score(preds, samples):
    """{'all': (P, R, F1), 'qa': ..., 'summary': ..., 'data2text': ...} in percent."""
    c = _counts(preds, samples)
    out = {}
    groups = {'all': np.ones(len(samples), bool)}
    for task in sorted({s['task'] for s in samples}):
        groups[task] = np.array([s['task'] == task for s in samples])
    for name, mask in groups.items():
        p, r, f = prf(c[mask].sum(0))
        out[name] = (round(100 * float(p), 2), round(100 * float(r), 2), round(100 * float(f), 2))
    return out


def bootstrap(preds, samples, base=None, n=1000, seed=0):
    """95% interval of span F1 (points), resampling sources. With base: of F1(preds) - F1(base)."""
    sources = sorted({s['source_id'] for s in samples})
    index = {src: i for i, src in enumerate(sources)}
    idx = np.array([index[s['source_id']] for s in samples])
    per_source = np.zeros((len(sources), 3))
    np.add.at(per_source, idx, _counts(preds, samples))
    weights = np.random.default_rng(seed).multinomial(
        len(sources), np.ones(len(sources)) / len(sources), size=n)
    f = prf(weights @ per_source)[2]
    if base is not None:
        per_base = np.zeros((len(sources), 3))
        np.add.at(per_base, idx, _counts(base, samples))
        f = f - prf(weights @ per_base)[2]
    lo, hi = np.percentile(100 * f, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)
