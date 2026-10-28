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
    if not samples:
        return float('nan'), float('nan')
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


def _per_source(preds, samples, idx, n_sources):
    c = np.zeros((n_sources, 3))
    np.add.at(c, idx, _counts(preds, samples))
    return c


def bootstrap_mean(runs, samples, bases=None, n=1000, seed=0):
    """95% interval of the mean span F1 of several runs on the same answers (e.g. one training run per
    seed), or, with bases (one per run, matched), of the mean of F1(run) - F1(base). Every run is scored
    on the same resampled sources, so the interval shows how much the choice of test answers moves the
    average; the runs themselves are kept as they are. With one run it equals bootstrap()."""
    if not samples or not runs:
        return float('nan'), float('nan')
    if bases is not None and len(bases) != len(runs):
        raise ValueError(f'{len(runs)} runs but {len(bases)} base runs')
    sources = sorted({s['source_id'] for s in samples})
    index = {src: i for i, src in enumerate(sources)}
    idx = np.array([index[s['source_id']] for s in samples])
    weights = np.random.default_rng(seed).multinomial(
        len(sources), np.ones(len(sources)) / len(sources), size=n)
    f = np.mean([prf(weights @ _per_source(r, samples, idx, len(sources)))[2] for r in runs], axis=0)
    if bases is not None:
        f = f - np.mean([prf(weights @ _per_source(b, samples, idx, len(sources)))[2] for b in bases], axis=0)
    lo, hi = np.percentile(100 * f, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)


def answer_f1(preds, samples):
    """Answer-level F1 in percent, like LettuceDetect's example-level score: an answer counts as flagged
    when at least one span is marked in it, and F1 is that of the flagged answers."""
    if len(preds) != len(samples):
        raise ValueError(f'{len(preds)} predictions for {len(samples)} answers')
    pred = np.array([bool(p) for p in preds])
    gold = np.array([bool(s['spans']) for s in samples])
    tp, fp, fn = int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(100 * (2 * p * r / (p + r) if p + r else 0.0), 2)
