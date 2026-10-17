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


# Ways to make the 8-bit file. All store the weights in 8 bits (about 4 times smaller than float).
#   w8    The default. Weights in 8 bits, in blocks of 32 numbers with one scale each (the
#         word embeddings too); the arithmetic stays in float. Close to the float model,
#         and about as fast as it.
#   mix   Faster but less exact. Most layers also do their arithmetic in 8 bits: their input
#         is rounded to 8 bits on the fly, with one scale for the whole input. Layers whose
#         input has a few huge values (found by running the float model on a few training
#         answers) stay as in w8, because one scale would round their ordinary values to zero.
#   dyn   onnxruntime's dynamic quantization as first used here: 8-bit arithmetic in every
#         layer. Kept only to show why it failed.
RECIPES = ('w8', 'mix', 'dyn')
EXTREME = 0.5   # a layer input is "extreme" when one 8-bit step is over half its typical value
BLOCK = 32


def _float_nodes(model):
    """The final layer (two outputs per word) stays in float: it is tiny and decides every label."""
    return [n.name for n in model.graph.node if n.op_type in ('MatMul', 'Gemm') and 'classifier' in n.name]


def _weight_matmuls(model):
    inits = {i.name for i in model.graph.initializer}
    return [n for n in model.graph.node if n.op_type == 'MatMul' and n.input[1] in inits]


def input_ranges(fp32_path, encoded):
    """For every layer with weights: how coarse 8-bit rounding of its input would be, as one
    8-bit step (whole range of the input / 255) over the typical (median) size of its values.
    The worst over the given answers (lists of token ids). Above about 0.5, most ordinary
    values would round to zero."""
    import onnx
    import onnxruntime as ort
    model = onnx.load(fp32_path)
    layers = _weight_matmuls(model)
    names = list(dict.fromkeys(n.input[0] for n in layers))
    known = {o.name for o in model.graph.output}
    for x in names:
        if x not in known:
            model.graph.output.append(onnx.helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    del model
    worst = dict.fromkeys(names, 0.0)
    for ids in encoded:
        x = np.array([ids], dtype=np.int64)
        for name, a in zip(names, sess.run(names, {'input_ids': x, 'attention_mask': np.ones_like(x)})):
            step = (max(float(a.max()), 0.0) - min(float(a.min()), 0.0)) / 255
            worst[name] = max(worst[name], step / max(float(np.median(np.abs(a))), 1e-12))
    return {n.name: round(worst[n.input[0]], 3) for n in layers}


def _embeddings_8bit(model):
    """Word embeddings in 8 bits, one scale per block of 32 numbers of each row: look up the
    rows of the answer's words, then scale them back to float."""
    from onnx import helper, numpy_helper
    inits = {i.name: i for i in model.graph.initializer}
    nodes = []
    for n in model.graph.node:
        table = inits.get(n.input[0]) if n.op_type == 'Gather' else None
        if (table is None or table.data_type != 1 or len(table.dims) != 2 or table.dims[0] < 1000
                or table.dims[1] % BLOCK):
            nodes.append(n)
            continue
        w = numpy_helper.to_array(table)
        rows, width = w.shape
        blocks = w.reshape(rows, width // BLOCK, BLOCK)
        scale = np.maximum(np.abs(blocks).max(2), 1e-12) / 127
        q = np.clip(np.round(blocks / scale[:, :, None]), -127, 127).astype(np.int8).reshape(rows, width)
        b, p = table.name, (n.name or table.name) + '_q8'
        model.graph.initializer.remove(table)
        model.graph.initializer.extend([
            numpy_helper.from_array(q, b + '_q8'),
            numpy_helper.from_array(scale.astype(np.float32), b + '_scale'),
            numpy_helper.from_array(np.array([-1], np.int64), b + '_last'),
            numpy_helper.from_array(np.array([0, 0, width // BLOCK, BLOCK], np.int64), b + '_blocks'),
            numpy_helper.from_array(np.array([0, 0, width], np.int64), b + '_width')])
        ids = n.input[1]
        nodes += [helper.make_node('Gather', [b + '_q8', ids], [p + '_rows'], name=p + '_rows'),
                  helper.make_node('Cast', [p + '_rows'], [p + '_float'], name=p + '_cast', to=1),
                  helper.make_node('Reshape', [p + '_float', b + '_blocks'], [p + '_b'], name=p + '_split'),
                  helper.make_node('Gather', [b + '_scale', ids], [p + '_s'], name=p + '_scales'),
                  helper.make_node('Unsqueeze', [p + '_s', b + '_last'], [p + '_s4'], name=p + '_unsq'),
                  helper.make_node('Mul', [p + '_b', p + '_s4'], [p + '_scaled'], name=p + '_mul'),
                  helper.make_node('Reshape', [p + '_scaled', b + '_width'], [n.output[0]], name=p + '_join')]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def _weights_8bit_float_sums(model, nodes):
    """8-bit weights in blocks of 32 (onnxruntime MatMulNBits); arithmetic in float."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    nodes = set(nodes)
    if not nodes:
        return model
    others = [n.name for n in model.graph.node if n.op_type == 'MatMul' and n.name not in nodes]
    q = MatMulNBitsQuantizer(model, bits=8, block_size=BLOCK, is_symmetric=True, nodes_to_exclude=others,
                             op_types_to_quantize=('MatMul',))
    q.process()
    return q.model.model


def quantize(fp32_path, int8_path, recipe='w8', calib=None):
    """Writes the 8-bit model (see RECIPES above) and returns what was done. 'mix' needs calib:
    a few answers as lists of token ids, used only to find the layers with extreme inputs."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    if recipe == 'dyn':
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        return {'recipe': recipe}
    if recipe not in RECIPES:
        raise ValueError(f'Unknown 8-bit recipe {recipe}; choose from {RECIPES}')
    if recipe == 'mix' and not calib:
        raise ValueError('the mix recipe needs a few calibration answers (calib)')
    ranges = input_ranges(fp32_path, calib) if recipe == 'mix' else None
    model = onnx.load(fp32_path)
    keep = set(_float_nodes(model))
    layers = [n.name for n in _weight_matmuls(model) if n.name not in keep]
    slow = [n for n in layers if ranges is None or ranges[n] > EXTREME]
    model = _weights_8bit_float_sums(model, slow)
    fast = [n for n in layers if n not in set(slow)]
    if fast:
        quantize_dynamic(model, int8_path, op_types_to_quantize=['MatMul'], nodes_to_quantize=fast,
                         per_channel=True, weight_type=QuantType.QInt8)
        model = onnx.load(int8_path)
    onnx.save(_embeddings_8bit(model), int8_path)
    return {'recipe': recipe, 'layers': len(layers), 'layers_with_8bit_arithmetic': len(fast),
            'kept_float': sorted(keep)}


def agreement(fp32_path, int8_path, encoded):
    """How close the 8-bit file is to the float file on the same inputs (lists of token ids):
    the largest change in hallucination probability, and the share of words whose 0.5 decision flips."""
    a, b = session(fp32_path), session(int8_path)
    worst, flips, total = 0.0, 0, 0
    for ids in encoded:
        p, q = _probs(a, ids), _probs(b, ids)
        worst = max(worst, float(np.abs(p - q).max()))
        flips += int(((p > 0.5) != (q > 0.5)).sum())
        total += len(ids)
    return {'max_prob_diff': round(worst, 4), 'flipped_share': round(flips / max(total, 1), 5)}


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
    import gc
    import psutil
    encs = [encode(tok, s['prompt'], s['answer'], max_length)[0] for s in samples]
    gc.collect()
    before = psutil.Process().memory_info().rss
    sess = session(onnx_path, threads)
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
            'rss_mb': round(psutil.Process().memory_info().rss / 2**20),
            # what loading and running the model added to this process (the rest is Python itself)
            'model_mem_mb': round((psutil.Process().memory_info().rss - before) / 2**20)}


def size_mb(path):
    return round(os.path.getsize(path) / 2**20, 1)
