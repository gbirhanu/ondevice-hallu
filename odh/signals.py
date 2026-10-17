"""Signals of the answering model, for every word of its answer.

On a phone these are by-products of writing the answer. For RAGTruth we get them by
re-reading each answer with the open model that wrote it (one forward pass over prompt +
answer, the answer "teacher-forced"). For answer token j we record what the model had
when it chose that token:

  confidence (Level 1): log-probability of the chosen token, entropy of the next-token
      distribution, how far the chosen token is below the top one, and the gap between
      the top two (all in nats).
  attention (Level 2): the lookback ratio of Lookback Lens per layer, summarised over
      heads by mean, min and max. For one head: mean attention on the prompt (passages
      and question) divided by (that + mean attention on the answer written so far).
      1.0 = looking only at the prompt.

Records are saved in small chunk files, never rewritten, so a stopped run continues.
"""
import gzip
import os
import pickle
import time

import numpy as np
import torch

# RAGTruth model name -> open weights of the same model, tried in this order. The
# NousResearch copies of Llama 2 need no sign-in; the official repositories (and Mistral)
# need a free Hugging Face token whose account has accepted the model's licence.
GENERATORS = {
    'llama-2-7b-chat': ['NousResearch/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-7b-chat-hf'],
    'llama-2-13b-chat': ['NousResearch/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf'],
    'mistral-7B-instruct': ['mistralai/Mistral-7B-Instruct-v0.1'],
}
CONF_NAMES = ['logp', 'entropy', 'below_top', 'top_gap']


def load_generator(hf_id, four_bit=True, device='cuda'):
    """Eager attention, so the attention weights exist and the hooks can read them."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    kwargs = {'attn_implementation': 'eager'}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
        kwargs['device_map'] = {'': 0}
        kwargs['dtype'] = torch.float16
    else:
        kwargs['dtype'] = torch.float16 if device == 'cuda' else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except TypeError:            # older transformers call it torch_dtype
        kwargs['torch_dtype'] = kwargs.pop('dtype')
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    if not four_bit:
        model = model.to(device)
    return tok, model.eval()


def chat_prefix(tok, prompt):
    """The text the model saw before its answer: its own chat format around the prompt."""
    if getattr(tok, 'chat_template', None):
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                       add_generation_prompt=True)
    return f'[INST] {prompt.strip()} [/INST]'


def tokenize(tok, prompt, answer):
    """(ids, index of the first answer token, answer-relative (start, end) of each answer token)."""
    prefix = chat_prefix(tok, prompt)
    bos = tok.bos_token or ''
    if bos and prefix.startswith(bos):
        prefix = prefix[len(bos):]
    text = prefix + ' ' + answer
    a_char = len(prefix) + 1
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc['input_ids']), [tuple(o) for o in enc['offset_mapping']]
    if tok.bos_token_id is not None:
        ids, offs = [tok.bos_token_id] + ids, [(0, 0)] + offs
    first = next((i for i, (s, e) in enumerate(offs) if e > a_char), None)
    if first is None or first == 0:
        return ids, None, []
    ans = [(max(s - a_char, 0), max(e - a_char, 0)) for s, e in offs[first:]]
    return ids, first, ans


def _attention_modules(model):
    """The self-attention block of every decoder layer, in order."""
    mods = [m for n, m in model.named_modules() if n.endswith('self_attn')]
    if not mods:
        raise ValueError('No self_attn modules found: this model type is not supported.')
    return mods


@torch.no_grad()
def extract(model, tok, prompt, answer, max_tokens=4096):
    """Signals for one answer, or None when it cannot be read (too long, empty)."""
    ids, a0, offsets = tokenize(tok, prompt, answer)
    n = len(ids)
    if a0 is None or n > max_tokens or n - a0 < 1:
        return None
    rows = slice(a0 - 1, n - 1)          # position p predicts token p + 1
    t_ans = n - a0
    count = torch.arange(t_ans, dtype=torch.float32)
    look = []

    def hook(module, args, output):
        w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if w is None:
            raise RuntimeError('No attention weights: load the model with attn_implementation="eager".')
        W = w[0, :, rows, :].float()                       # heads x answer tokens x all positions
        ctx = W[..., :a0].mean(-1)
        new = W[..., a0:].sum(-1).cpu() / count.clamp(min=1)
        ctx = ctx.cpu()
        lr = ctx / (ctx + new + 1e-9)
        look.append(torch.stack([lr.mean(0), lr.min(0).values, lr.max(0).values], -1))   # tokens x 3
        return (output[0], None) + tuple(output[2:])

    handles = [m.register_forward_hook(hook) for m in _attention_modules(model)]
    try:
        device = next(model.parameters()).device
        x = torch.tensor([ids], device=device)
        try:
            out = model(input_ids=x, use_cache=False, logits_to_keep=t_ans + 1)
            logits = out.logits[0, :t_ans]
        except TypeError:                                   # older transformers
            out = model(input_ids=x, use_cache=False)
            logits = out.logits[0, rows]
    finally:
        for h in handles:
            h.remove()
    logp = torch.log_softmax(logits.float(), -1)
    chosen = torch.tensor(ids[a0:], device=logp.device)
    lp = logp[torch.arange(t_ans, device=logp.device), chosen]
    ent = -(logp.exp() * logp).sum(-1)
    top2 = logp.topk(2, -1).values
    conf = torch.stack([lp, ent, top2[:, 0] - lp, top2[:, 0] - top2[:, 1]], -1).cpu().numpy()
    return {'gen_offsets': np.array(offsets, dtype=np.int32),
            'conf': conf.astype(np.float16),
            'look': torch.stack(look, 1).numpy().astype(np.float16)}   # tokens x layers x 3


def record_all(model, tok, samples, folder, chunk=25, log=print):
    """Signals for every sample, saved as folder/NNNN.pkl.gz (chunk answers each). Existing
    chunk files are kept, so a stopped run continues where it left off."""
    os.makedirs(folder, exist_ok=True)
    samples = sorted(samples, key=lambda s: int(s['id']) if s['id'].isdigit() else s['id'])
    t0, done, skipped = time.time(), 0, 0
    for c in range(0, len(samples), chunk):
        path = os.path.join(folder, f'{c // chunk:04d}.pkl.gz')
        if os.path.exists(path):
            continue
        part = {}
        for s in samples[c:c + chunk]:
            rec = extract(model, tok, s['prompt'], s['answer'])
            if rec is None:
                skipped += 1
                continue
            part[s['id']] = rec
        with gzip.open(path + '.tmp', 'wb') as fh:
            pickle.dump(part, fh)
        os.replace(path + '.tmp', path)
        done += len(samples[c:c + chunk])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f'  {min(c + chunk, len(samples))}/{len(samples)} answers ({time.time() - t0:.0f}s, '
            f'{skipped} skipped)')
    return skipped


def load_records(folder):
    out = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if name.endswith('.pkl.gz'):
            with gzip.open(os.path.join(folder, name), 'rb') as fh:
                out.update(pickle.load(fh))
    return out
