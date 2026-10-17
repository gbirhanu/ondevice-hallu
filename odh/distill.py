"""Teacher-student training (knowledge distillation) for small span checkers.

The teacher is a large checker (LettuceDetect-large, 396M parameters); the student is a small
encoder (Ettin 17M/32M/68M, the bases of TinyLettuce). The teacher gives the student two things:

  1. soft labels on RAGTruth's training answers: its probability for every word, used next to
     the human label (target = alpha * human label + (1 - alpha) * teacher probability);
  2. labels for new answers that no person has labelled: answers that small, phone-sized models
     write to RAGTruth's training prompts (target = teacher probability).

The loss is the usual cross-entropy with these soft targets. For two classes, mixing the targets
is the same as mixing the two losses: alpha * CE(human) + (1 - alpha) * CE(teacher).

This file holds the parts that are new: writing the extra answers, and turning the teacher's
word probabilities into targets for the student's own tokens (the two models cut words into
tokens differently, so targets are matched by character overlap).
"""
import gzip
import json
import os
import time

import numpy as np

# Small open answering models (phone-sized), tried for the extra answers. The first three
# download without sign-in; the last two need a Hugging Face token whose account has accepted
# the model's licence.
SMALL_GENERATORS = {
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'smollm2-1.7b': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'gemma-2-2b': 'google/gemma-2-2b-it',
}
DEFAULT_GENERATORS = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'smollm2-1.7b']
CUTOFFS = np.round(np.arange(0.05, 0.951, 0.05), 2)


# ---------------------------------------------------------------- targets for the student

def soft_targets(offsets, teacher_tokens):
    """Teacher probability for each student token: the mean over the teacher tokens that share
    characters with it; a student token that shares none takes the nearest teacher token.

    offsets: [(start, end)] of the student's answer tokens; teacher_tokens: [(start, end, p)]."""
    if not offsets:
        return []
    if not teacher_tokens:
        raise ValueError('The teacher has no tokens for this answer.')
    s = np.array([o[0] for o in offsets], dtype=np.float64)[:, None]
    e = np.array([o[1] for o in offsets], dtype=np.float64)[:, None]
    t = np.array(teacher_tokens, dtype=np.float64)
    ts, te, tp = t[:, 0][None, :], t[:, 1][None, :], t[:, 2]
    m = ((te > s) & (ts < e) & (te > ts)).astype(np.float64)
    none = m.sum(1) == 0
    if none.any():
        mid = (s[:, 0] + e[:, 0]) / 2
        nearest = np.abs((ts[0] + te[0]) / 2 - mid[:, None]).argmin(1)
        m[np.where(none)[0], nearest[none]] = 1.0
    return ((m / m.sum(1, keepdims=True)) @ tp).tolist()


def hard_targets(spans, offsets):
    """1.0 for a token that overlaps a human-marked span, else 0.0."""
    return [1.0 if any(e > gs and s < ge for gs, ge in spans) else 0.0 for s, e in offsets]


def targets_for(sample, offsets, alpha=0.5):
    """The training target of every answer token of one sample.

    sample['spans'] is None for answers that no person labelled; sample.get('teacher') holds the
    teacher's [(start, end, p)] when there is one. alpha = weight of the human label."""
    human = None if sample.get('spans') is None else hard_targets(sample['spans'], offsets)
    teacher = soft_targets(offsets, sample['teacher']) if sample.get('teacher') else None
    if human is None and teacher is None:
        raise ValueError(f'Answer {sample.get("id")} has neither human nor teacher labels.')
    if teacher is None:
        return human
    if human is None:
        return teacher
    return [alpha * h + (1 - alpha) * q for h, q in zip(human, teacher)]


def best_cutoff(token_lists, samples):
    """(cut-off, span F1 at it, span F1 at 0.5) on labelled answers, e.g. the dev set."""
    from odh.metrics import score
    from odh.spans import all_spans
    f1 = [score(all_spans(token_lists, c), samples)['all'][2] for c in CUTOFFS]
    k = int(np.argmax(f1))
    return float(CUTOFFS[k]), f1[k], score(all_spans(token_lists, 0.5), samples)['all'][2]


# ---------------------------------------------------------------- extra answers

def extra_prompts(train_samples):
    """One prompt per training source (RAGTruth's own prompt), sorted by source."""
    first = {}
    for s in train_samples:
        first.setdefault(s['source_id'], s)
    return [{'id': k, 'source_id': k, 'task': v['task'], 'prompt': v['prompt']}
            for k, v in sorted(first.items(), key=lambda kv: (len(kv[0]), kv[0]))]


def load_small_generator(hf_id, device='cuda'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == 'cuda' else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
    except TypeError:                                   # older transformers call it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    return tok, model.to(device).eval()


def chat_text(tok, prompt):
    """The prompt as the model's own chat turn (or the plain prompt if it has no chat format)."""
    if getattr(tok, 'chat_template', None):
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                       add_generation_prompt=True)
    return prompt


def generate(model, tok, prompts, max_new_tokens=450, temperature=0.7, top_p=0.9,
             batch_tokens=24000, max_rows=32, max_prompt_tokens=3500, seed=0):
    """One sampled answer per prompt: [{'answer', 'finished', 'new_tokens'}] or None when the
    prompt is too long. Prompts of similar length are batched together."""
    import torch
    from odh.model import batches
    device = next(model.parameters()).device
    texts = [chat_text(tok, p) for p in prompts]
    ids = [tok(t, add_special_tokens=False)['input_ids'] for t in texts]
    keep = [i for i, x in enumerate(ids) if len(x) <= max_prompt_tokens]
    lengths = {i: len(ids[i]) + max_new_tokens for i in keep}
    full = [lengths.get(i, 0) for i in range(len(prompts))]
    out = [None] * len(prompts)
    cfg_eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = set(cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos]) | {tok.eos_token_id}
    eos.discard(None)
    for b in batches(full, sorted(keep, key=lambda i: full[i]), batch_tokens, max_rows):
        torch.manual_seed(seed + b[0])
        width = max(len(ids[i]) for i in b)
        input_ids = torch.full((len(b), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(b), width), dtype=torch.long)
        for r, i in enumerate(b):                                  # left padding
            input_ids[r, width - len(ids[i]):] = torch.tensor(ids[i])
            mask[r, width - len(ids[i]):] = 1
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                 top_p=top_p, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
        new = gen[:, width:].cpu()
        for r, i in enumerate(b):
            row = new[r].tolist()
            stop = next((k for k, t in enumerate(row) if t in eos), None)
            body = row if stop is None else row[:stop]
            out[i] = {'answer': tok.decode(body, skip_special_tokens=True).strip(),
                      'finished': stop is not None, 'new_tokens': len(body)}
    return out


def generate_all(model, tok, items, folder, chunk=64, log=print, **kwargs):
    """Answers for every item, saved as folder/NNNN.json.gz (chunk prompts each). Existing chunk
    files are kept, so a stopped run continues where it left off. Returns seconds spent."""
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    for c in range(0, len(items), chunk):
        path = os.path.join(folder, f'{c // chunk:04d}.json.gz')
        if os.path.exists(path):
            continue
        part = items[c:c + chunk]
        answers = generate(model, tok, [x['prompt'] for x in part], seed=c, **kwargs)
        rows = [{'source_id': x['source_id'], **(a or {'answer': None, 'finished': False, 'new_tokens': 0})}
                for x, a in zip(part, answers)]
        with gzip.open(path + '.tmp', 'wt', encoding='utf-8') as fh:
            json.dump(rows, fh)
        os.replace(path + '.tmp', path)
        log(f'  {min(c + chunk, len(items))}/{len(items)} prompts ({time.time() - t0:.0f}s)')
    return time.time() - t0


def load_extra(folder, generator, sources, min_words=3):
    """Extra answers as samples (spans=None: no human labels). Unfinished answers (cut at the
    token limit, often repeating themselves) and near-empty ones are left out.

    sources: {source_id: a RAGTruth sample of that source} for prompt and task."""
    out, dropped = [], 0
    if not os.path.isdir(folder):
        return out, dropped
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.json.gz'):
            continue
        with gzip.open(os.path.join(folder, name), 'rt', encoding='utf-8') as fh:
            for row in json.load(fh):
                src = sources.get(row['source_id'])
                if (src is None or not row.get('answer') or not row.get('finished')
                        or len(row['answer'].split()) < min_words):
                    dropped += 1
                    continue
                out.append({'id': f'{generator}/{row["source_id"]}', 'source_id': row['source_id'],
                            'task': src['task'], 'split': 'extra', 'model': generator,
                            'prompt': src['prompt'], 'answer': row['answer'], 'spans': None})
    return out, dropped
