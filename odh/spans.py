"""From per-token probabilities to character spans of the answer.

A token is marked when its hallucination probability is above the cut-off (0.5 =
the class with the higher probability, LettuceDetect's rule). Neighbouring marked
tokens join into one span. There is deliberately no tuning of the cut-off.
"""


def to_spans(tokens, cutoff=0.5):
    """tokens: [(start, end, probability), ...] for the answer tokens, in order."""
    spans, current = [], None
    for start, end, prob in tokens:
        if end <= start:
            continue
        if prob > cutoff:
            if current is None:
                current = [start, end]
            else:
                current[1] = end
        elif current is not None:
            spans.append(tuple(current))
            current = None
    if current is not None:
        spans.append(tuple(current))
    return spans


def all_spans(token_lists, cutoff=0.5):
    return [to_spans(tokens, cutoff) for tokens in token_lists]
