"""Token classifiers (ModernBERT, Ettin, TinyLettuce, LettuceDetect): load, predict, train.

Input = tokenizer(prompt, answer) with only the prompt cut when too long, exactly as
LettuceDetect builds it for RAGTruth. Only answer tokens get a label and a prediction.
"""
import math
import random
import time

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from odh.metrics import score
from odh.spans import all_spans


def load(name, device='cpu', new_head=False):
    """new_head=True (training only) also accepts a checkpoint trained for another task,
    e.g. a 3-label NLI model: its encoder is kept, the 2-class output layer starts fresh."""
    tok = AutoTokenizer.from_pretrained(name)
    kwargs = {'num_labels': 2}
    if new_head and AutoConfig.from_pretrained(name).num_labels != 2:
        kwargs.update(ignore_mismatched_sizes=True, id2label={0: 'supported', 1: 'hallucinated'},
                      label2id={'supported': 0, 'hallucinated': 1})
    try:   # ModernBERT can compile parts of itself; that fails on some GPUs and on CPU export
        model = AutoModelForTokenClassification.from_pretrained(name, reference_compile=False, **kwargs)
    except (TypeError, ValueError):
        model = AutoModelForTokenClassification.from_pretrained(name, **kwargs)
    return tok, model.float().to(device).eval()      # our trained runs are saved in 16-bit


def encode(tok, prompt, answer, max_length=4096):
    """(input ids, positions of the answer tokens, their (start, end) offsets in the answer)."""
    enc = tok(prompt, answer, truncation='only_first', max_length=max_length,
              return_offsets_mapping=True)
    positions = [i for i, sid in enumerate(enc.sequence_ids()) if sid == 1]
    offsets = [tuple(enc['offset_mapping'][i]) for i in positions]
    return enc['input_ids'], positions, offsets


def batches(lengths, order, batch_tokens, max_rows=64):
    """Groups of answers whose padded size stays within batch_tokens."""
    out, cur, longest = [], [], 0
    for i in order:
        if cur and (max(longest, lengths[i]) * (len(cur) + 1) > batch_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, lengths[i])
    if cur:
        out.append(cur)
    return out


def _pad(seqs, pad_id, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def predict(model, tok, samples, max_length=4096, batch_tokens=16384, fp16=True, log=None):
    """Per answer: [(start, end, p_hallucinated), ...] for each answer token."""
    model.eval()
    device = next(model.parameters()).device
    encs = [encode(tok, s['prompt'], s['answer'], max_length) for s in samples]
    lengths = [len(e[0]) for e in encs]
    order = sorted(range(len(samples)), key=lambda i: lengths[i])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    out, done, t0 = [None] * len(samples), 0, time.time()
    for b in batches(lengths, order, batch_tokens):
        ids, mask = _pad([encs[i][0] for i in b], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=fp16 and device.type == 'cuda'):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits.float(), -1)[..., 1].cpu().numpy()
        for r, i in enumerate(b):
            _, positions, offsets = encs[i]
            out[i] = [(int(s), int(e), round(float(probs[r, j]), 5))
                      for j, (s, e) in zip(positions, offsets)]
        done += len(b)
        if log and (done == len(samples) or done // 500 > (done - len(b)) // 500):
            log(f'  predicted {done}/{len(samples)} answers ({time.time() - t0:.0f}s)')
    return out


def train(base, train_samples, dev_samples, out_dir, epochs=3, lr=2e-5, max_length=4096,
          batch_tokens=8192, seed=42, device='cuda', fp16=True, warmup=0.1, alpha=0.5, log=print):
    """Fine-tune `base` as a 2-class token classifier on answer tokens (cross-entropy).

    Targets come from odh.distill.targets_for: the human label; or, when a sample carries the
    teacher's word probabilities ('teacher'), alpha * human + (1 - alpha) * teacher; or, for answers
    no person labelled (spans None), the teacher alone. Keeps the epoch with the best dev span F1
    at the plain 0.5 cut-off and saves only that model to out_dir (16-bit), once, at the end.
    Returns the per-epoch history and the cut-off that is best on dev for the kept epoch.
    """
    from odh.distill import best_cutoff, targets_for
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    tok, model = load(base, device, new_head=True)
    data = []
    for s in train_samples:
        ids, positions, offsets = encode(tok, s['prompt'], s['answer'], max_length)
        if positions:
            data.append((ids, positions, targets_for(s, offsets, alpha)))
    lengths = [len(d[0]) for d in data]
    rng = random.Random(seed)

    def epoch_batches():
        order = list(range(len(data)))
        rng.shuffle(order)
        order = [i for c in range(0, len(order), 2000)          # similar lengths together
                 for i in sorted(order[c:c + 2000], key=lambda i: lengths[i])]
        bs = batches(lengths, order, batch_tokens)
        rng.shuffle(bs)
        return bs

    steps = math.ceil(len(epoch_batches()) * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    use_amp = fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    best_f1, best_state, best_dev, history = -1.0, None, None, []
    log(f'Training {base} on {len(data)} answers for {epochs} epochs ({steps} steps).')
    for ep in range(epochs):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for step, b in enumerate(epoch_batches()):
            ids, mask = _pad([data[i][0] for i in b], pad_id, device)
            rows = torch.cat([torch.full((len(data[i][1]),), r, dtype=torch.long)
                              for r, i in enumerate(b)]).to(device)
            cols = torch.cat([torch.tensor(data[i][1], dtype=torch.long) for i in b]).to(device)
            y = torch.cat([torch.tensor(data[i][2], dtype=torch.float32) for i in b]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            logp = torch.log_softmax(logits.float()[rows, cols], -1)
            loss = -(y * logp[:, 1] + (1 - y) * logp[:, 0]).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
            seen += len(b)
            if (step + 1) % 200 == 0:
                log(f'  epoch {ep + 1} step {step + 1}: loss {total / seen:.4f} ({time.time() - t0:.0f}s)')
        dev_tokens = predict(model, tok, dev_samples, max_length, batch_tokens * 2, fp16)
        f1 = score(all_spans(dev_tokens), dev_samples)['all'][2]
        history.append({'epoch': ep + 1, 'loss': round(total / max(seen, 1), 4), 'dev_span_f1': f1,
                        'seconds': round(time.time() - t0)})
        log(f'Epoch {ep + 1}: loss {total / max(seen, 1):.4f}, dev span F1 {f1:.2f}')
        if f1 > best_f1:
            best_f1, best_dev = f1, dev_tokens
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.half().save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    cut, f1_cut, _ = best_cutoff(best_dev, dev_samples)
    return {'history': history, 'best_epoch': max(history, key=lambda h: h['dev_span_f1'])['epoch'],
            'dev_f1_05': best_f1, 'dev_cutoff': cut, 'dev_f1_at_cutoff': f1_cut}
