"""PsiloQA (Rykov et al., 2025; CC-BY-4.0) as a second test set, split by the size of the model
that wrote each answer.

PsiloQA has answers from 24 open models (135M-9B) to questions about Wikipedia passages, with the
unsupported spans marked by GPT-4o (checked against people on 100 English answers). The answers
were written without the passage, but the spans are marked against it, so a checker that reads
the passage can be scored on them. We use the English answers only and never train on PsiloQA.
"""
import re

HF_ID = 's-nlp/PsiloQA'
KNOWN_SIZES = {'phi-3-mini': 3.8, 'phi-3.5-mini': 3.8, 'phi-3-medium': 14.0, 'phi-3-small': 7.0}
BUCKETS = ['up to 2B', '2-4B', 'over 4B']


def model_size(checkpoint):
    """Parameters in billions from a model name ('TinyLlama-1.1B-Chat' -> 1.1, 'SmolLM2-135M' -> 0.135)."""
    name = checkpoint.lower().split('/')[-1]
    for key, size in KNOWN_SIZES.items():
        if key in name:
            return size
    for m in re.finditer(r'(?<![\d.])(\d+(?:\.\d+)?)([bm])(?![a-z])', name):
        value = float(m.group(1))
        return value / 1000 if m.group(2) == 'm' else value
    return None


def bucket(checkpoint):
    size = model_size(checkpoint)
    if size is None:
        return 'unknown size'
    return 'up to 2B' if size <= 2 else ('2-4B' if size <= 4 else 'over 4B')


def prompt_of(question, passage):
    """The question and passage in the style of RAGTruth's question-answering prompts."""
    return (f'Briefly answer the following question:\n{question}\n'
            f'Bear in mind that your response should be strictly based on the following passage:\n'
            f'passage 1:{passage}\noutput:')


def samples_from(records, lang='en', split=''):
    """Records with PsiloQA's fields -> samples in the format of odh.data."""
    out = []
    for i, r in enumerate(records):
        if lang and r.get('lang') != lang:
            continue
        answer = r['llm_answer']
        spans = [(int(a), int(b)) for a, b in (r.get('labels') or []) if 0 <= int(a) < int(b) <= len(answer)]
        out.append({'id': f"{split}:{r.get('id') or i}", 'source_id': r.get('wiki_title') or str(i),
                    'task': 'qa', 'split': split, 'model': r['llm_checkpoint'],
                    'size_group': bucket(r['llm_checkpoint']),
                    'prompt': prompt_of(r['question'], r['wiki_passage']), 'answer': answer, 'spans': spans})
    return out


def load(splits=('test', 'validation', 'train'), lang='en'):
    from datasets import load_dataset
    out = []
    for split in splits:
        out += samples_from(load_dataset(HF_ID, split=split), lang, split)
    return out
