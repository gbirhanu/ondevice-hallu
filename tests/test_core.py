"""CPU checks; no downloads. Run: python -m pytest tests -q"""
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (ModernBertConfig, ModernBertForSequenceClassification,
                          ModernBertForTokenClassification, PreTrainedTokenizerFast)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odh import cpu, data, metrics, model, spans  # noqa: E402


class SpanF1(unittest.TestCase):
    def test_counts_follow_lettucedetect(self):
        # overlap summed over every pair: two overlapping gold spans both count
        self.assertEqual(metrics.counts([(0, 10)], [(0, 5), (3, 8)]), (10, 10, 10))
        self.assertEqual(metrics.counts([], [(0, 5)]), (0, 0, 5))

    def test_score_by_task(self):
        samples = [{'id': '1', 'source_id': 'a', 'task': 'qa', 'spans': [(0, 4)]},
                   {'id': '2', 'source_id': 'b', 'task': 'summary', 'spans': [(2, 6)]}]
        s = metrics.score([[(0, 4)], [(10, 12)]], samples)
        self.assertEqual(s['qa'], (100.0, 100.0, 100.0))
        self.assertEqual(s['summary'], (0.0, 0.0, 0.0))
        self.assertEqual(s['all'][2], 57.14)   # counts are summed over the set, not averaged

    def test_tokens_to_spans(self):
        toks = [(0, 3, 0.9), (4, 7, 0.8), (8, 9, 0.1), (10, 12, 0.6), (12, 12, 0.9)]
        self.assertEqual(spans.to_spans(toks), [(0, 7), (10, 12)])


class Subsets(unittest.TestCase):
    def test_subsets_are_nested_and_balanced(self):
        samples = [{'id': str(i), 'task': ['qa', 'summary', 'data2text'][i % 3]} for i in range(300)]
        small, large = data.subset(samples, 30), data.subset(samples, 90)
        self.assertEqual(len(small), 30)
        self.assertTrue({s['id'] for s in small} <= {s['id'] for s in large})
        self.assertEqual(sum(s['task'] == 'qa' for s in small), 10)


def tiny_tokenizer():
    words = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'] + [f'w{i}' for i in range(40)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', cls_token='[CLS]',
                                   sep_token='[SEP]', pad_token='[PAD]', mask_token='[MASK]')


def tiny_config(labels=2):
    return ModernBertConfig(vocab_size=45, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                            num_attention_heads=2, max_position_embeddings=1024, pad_token_id=3,
                            cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2,
                            num_labels=labels, local_attention=16)


def sample(i, n_words):
    words = ' '.join(f'w{(i * 7 + k) % 40}' for k in range(n_words))
    return {'id': str(i), 'source_id': str(i), 'task': 'qa', 'split': 'test',
            'prompt': words, 'answer': 'w1 w2 w3 w4 w5', 'spans': [(3, 8)]}


class Models(unittest.TestCase):
    def setUp(self):
        self.tmp = ROOT / 'work_test' / ('t-' + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def save(self, net, name):
        folder = self.tmp / name
        net.save_pretrained(folder)
        tiny_tokenizer().save_pretrained(folder)
        return str(folder)

    def test_nli_start_keeps_encoder_and_adds_two_class_layer(self):
        nli = ModernBertForSequenceClassification(tiny_config(3))
        folder = self.save(nli, 'nli')
        _, net = model.load(folder, new_head=True)
        self.assertEqual(net.classifier.out_features, 2)
        saved = nli.state_dict()
        for k, v in net.state_dict().items():
            if k.startswith('model.'):
                torch.testing.assert_close(v, saved[k])

    def test_onnx_matches_pytorch_on_short_and_long_inputs(self):
        torch.manual_seed(0)
        folder = self.save(ModernBertForTokenClassification(tiny_config()), 'tok')
        out = self.tmp / 'onnx'
        cpu.export(folder, str(out))
        cpu.quantize(str(out / 'model_fp32.onnx'), str(out / 'model_int8.onnx'))
        samples = [sample(i, n) for i, n in enumerate([5, 40, 300])]   # below and above the window
        self.assertLess(cpu.parity(folder, str(out / 'model_fp32.onnx'), samples), 1e-4)
        tok, net = model.load(folder)
        a = model.predict(net, tok, samples, fp16=False)
        b = cpu.predict(cpu.session(str(out / 'model_fp32.onnx')), tok, samples)
        self.assertEqual([[t[:2] for t in x] for x in a], [[t[:2] for t in x] for x in b])
        ids = [model.encode(tok, s['prompt'], s['answer'], 4096)[0] for s in samples]
        self.assertLess(cpu.agreement(str(out / 'model_fp32.onnx'), str(out / 'model_int8.onnx'), ids)['max_prob_diff'], 0.05)
        made = cpu.quantize(str(out / 'model_fp32.onnx'), str(out / 'model_mix.onnx'), 'mix', calib=ids)
        self.assertEqual(made['layers'], 13)   # 3 layers x 4 weight matrices + the head; the classifier stays float
        self.assertLess(cpu.agreement(str(out / 'model_fp32.onnx'), str(out / 'model_mix.onnx'), ids)['max_prob_diff'], 0.2)
        timing = cpu.time_answers(str(out / 'model_int8.onnx'), tok, samples, threads=1, warmup=1)
        self.assertEqual(timing['answers'], 3)


class Embeddings8bit(unittest.TestCase):
    def test_lookup_matches_block_rounding(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper
        rng = np.random.default_rng(0)
        table = rng.normal(size=(2000, 64)).astype(np.float32)
        table[:, 5] *= 30                           # an outlier column, as real embeddings have
        graph = helper.make_graph(
            [helper.make_node('Gather', ['table', 'ids'], ['out'], name='emb')], 'g',
            [helper.make_tensor_value_info('ids', TensorProto.INT64, ['b', 's'])],
            [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['b', 's', 64])],
            [numpy_helper.from_array(table, 'table')])
        m = cpu._embeddings_8bit(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8))
        onnx.checker.check_model(m)
        ids = np.array([[0, 7, 1999, 7]], dtype=np.int64)
        got = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider']).run(None, {'ids': ids})[0]
        blocks = table.reshape(2000, 2, 32)
        scale = np.abs(blocks).max(2, keepdims=True) / 127
        want = (np.round(blocks / scale) * scale).reshape(2000, 64)[ids]
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        self.assertLess(np.abs(got - table[ids]).max(), np.abs(table).max() / 127)


if __name__ == '__main__':
    unittest.main()
