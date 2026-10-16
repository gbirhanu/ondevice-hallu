"""The phone side: export to ONNX, shrink to 8-bit, predict and time on a CPU.

ONNX Runtime is what runs these models on Android, so the CPU numbers here use the
same engine. Timing feeds one answer at a time, as a phone app would.
"""
import os
import statistics
import time

import numpy as np
import torch

from odh.model import encode, load


class _Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(name, out_dir, opset=17):
    """Writes model_fp32.onnx and the tokenizer to out_dir. Returns the parameter count."""
    os.makedirs(out_dir, exist_ok=True)
    tok, model = load(name, 'cpu')
    tok.save_pretrained(out_dir)
    # A dummy input longer than ModernBERT's 128-token local attention window, so that
    # the traced graph is the general one; exported graphs are checked by parity() below.
    text = ' '.join(['the passage says this'] * 80)
    enc = tok(text, 'an answer to check', return_tensors='pt', truncation='only_first', max_length=400)
    path = os.path.join(out_dir, 'model_fp32.onnx')
    axes = {'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'},
            'logits': {0: 'batch', 1: 'seq'}}
    args = (enc['input_ids'], enc['attention_mask'])
    kwargs = dict(input_names=['input_ids', 'attention_mask'], output_names=['logits'],
                  dynamic_axes=axes, opset_version=opset)
    with torch.no_grad():
        try:
            torch.onnx.export(_Logits(model), args, path, dynamo=False, **kwargs)
        except TypeError:          # older torch without the dynamo switch
            torch.onnx.export(_Logits(model), args, path, **kwargs)
    return sum(p.numel() for p in model.parameters())


def quantize(fp32_path, int8_path):
    """8-bit weights (dynamic quantization): about 4 times smaller, usually faster on CPU."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)


def session(path, threads=None):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads or 0
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])


def _probs(sess, ids):
    x = np.array([ids], dtype=np.int64)
    logits = sess.run(['logits'], {'input_ids': x, 'attention_mask': np.ones_like(x)})[0][0]
    logits = logits - logits.max(-1, keepdims=True)
    e = np.exp(logits)
    return e[:, 1] / e.sum(-1)


def predict(sess, tok, samples, max_length=4096):
    """Same output as odh.model.predict, one answer at a time on the CPU."""
    out = []
    for s in samples:
        ids, positions, offsets = encode(tok, s['prompt'], s['answer'], max_length)
        p = _probs(sess, ids)
        out.append([(int(a), int(b), round(float(p[j]), 5)) for j, (a, b) in zip(positions, offsets)])
    return out


def parity(name, onnx_path, samples, max_length=4096):
    """Largest difference in hallucination probability between PyTorch and the ONNX file."""
    tok, model = load(name, 'cpu')
    sess = session(onnx_path)
    worst = 0.0
    for s in samples:
        ids, _, _ = encode(tok, s['prompt'], s['answer'], max_length)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).logits
        ref = torch.softmax(logits.float(), -1)[0, :, 1].numpy()
        worst = max(worst, float(np.abs(ref - _probs(sess, ids)).max()))
    return worst


def time_answers(onnx_path, tok, samples, threads, max_length=4096, warmup=3):
    """Time per answer (ms) on the CPU with a fixed number of threads, plus memory in use."""
    import psutil
    sess = session(onnx_path, threads)
    encs = [encode(tok, s['prompt'], s['answer'], max_length)[0] for s in samples]
    for ids in encs[:warmup]:
        _probs(sess, ids)
    times = []
    for ids in encs:
        t0 = time.perf_counter()
        _probs(sess, ids)
        times.append(1000 * (time.perf_counter() - t0))
    times.sort()
    return {'threads': threads, 'answers': len(times), 'median_ms': round(statistics.median(times), 1),
            'p90_ms': round(times[int(0.9 * (len(times) - 1))], 1),
            'mean_tokens': round(sum(map(len, encs)) / len(encs)),
            'rss_mb': round(psutil.Process().memory_info().rss / 2**20)}


def size_mb(path):
    return round(os.path.getsize(path) / 2**20, 1)
