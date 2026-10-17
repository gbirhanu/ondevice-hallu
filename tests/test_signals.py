"""Checks of the answering-model signals and the fusion test; CPU, no downloads.
Run: python -m pytest tests -q"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odh import fusion, signals  # noqa: E402


def tiny_generator():
    words = ['[UNK]', '<s>', '</s>', '[INST]', '[/INST]'] + [f'w{i}' for i in range(40)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', bos_token='<s>', eos_token='</s>')
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      bos_token_id=1, eos_token_id=2)
    model = LlamaForCausalLM._from_config(cfg, attn_implementation='eager').eval()
    return fast, model


PROMPT = 'w1 w2 w3 w4 w5 w6 w7 w8'
ANSWER = 'w9 w10 w11 w12 w13'


class Signals(unittest.TestCase):
    def setUp(self):
        self.tok, self.model = tiny_generator()

    def test_offsets_point_at_the_answer_words(self):
        ids, a0, offsets = signals.tokenize(self.tok, PROMPT, ANSWER)
        self.assertEqual([ANSWER[s:e] for s, e in offsets], ANSWER.split())
        self.assertEqual(ids[0], self.tok.bos_token_id)
        self.assertEqual(len(ids) - a0, 5)

    def test_confidence_and_lookback_match_a_direct_computation(self):
        rec = signals.extract(self.model, self.tok, PROMPT, ANSWER)
        ids, a0, _ = signals.tokenize(self.tok, PROMPT, ANSWER)
        n = len(ids)
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids]), output_attentions=True)
        logp = torch.log_softmax(out.logits[0].float(), -1)
        direct = [float(logp[a0 - 1 + j, ids[a0 + j]]) for j in range(n - a0)]
        np.testing.assert_allclose(rec['conf'][:, 0].astype(float), direct, atol=2e-3)
        self.assertTrue((rec['conf'][:, 2] >= -1e-3).all())            # chosen is never above the top
        # lookback ratio, layer 1, written out with loops
        att = out.attentions[1][0].float()                              # heads x n x n
        for j in range(n - a0):
            q = a0 - 1 + j
            ctx = att[:, q, :a0].mean(-1)
            new = att[:, q, a0:q + 1].sum(-1) / max(j, 1) if j else torch.zeros_like(ctx)
            lr = ctx / (ctx + new + 1e-9)
            self.assertAlmostEqual(float(rec['look'][j, 1, 0]), float(lr.mean()), places=2)
            self.assertAlmostEqual(float(rec['look'][j, 1, 1]), float(lr.min()), places=2)
        self.assertTrue(np.allclose(rec['look'][0].astype(float), 1.0))  # first word: nothing written yet
        self.assertEqual(rec['look'].shape, (5, 3, 3))

    def test_too_long_answers_are_skipped(self):
        self.assertIsNone(signals.extract(self.model, self.tok, PROMPT, ANSWER, max_tokens=5))


class Fusion(unittest.TestCase):
    def test_align_averages_overlapping_tokens(self):
        checker = np.array([[0, 4], [4, 9], [20, 22]], dtype=np.float32)
        gen = np.array([[0, 2], [2, 6], [6, 9]], dtype=np.float32)
        vals = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
        out = fusion.align(checker, gen, vals)
        np.testing.assert_allclose(out[:, 0], [2.0, 4.0, 5.0])        # last: no overlap, nearest

    def test_signals_that_carry_information_raise_span_f1(self):
        rng = np.random.default_rng(0)
        samples, toks, records = [], [], {}
        for i in range(120):
            n = 30
            bad = np.zeros(n, bool)
            start = rng.integers(0, 20)
            bad[start:start + rng.integers(2, 8)] = rng.random() < 0.7
            offsets = [(3 * k, 3 * k + 2) for k in range(n)]
            spans = []
            k = 0
            while k < n:
                if bad[k]:
                    j = k
                    while j < n and bad[j]:
                        j += 1
                    spans.append((offsets[k][0], offsets[j - 1][1]))
                    k = j
                else:
                    k += 1
            p = np.clip(0.3 * bad + 0.35 + 0.25 * rng.standard_normal(n), 0.01, 0.99)   # weak text score
            conf = np.zeros((n, 4), np.float16)
            conf[:, 0] = -0.2 - 3.0 * bad + 0.3 * rng.standard_normal(n)                 # strong signal
            conf[:, 1] = 1.0 + bad
            look = np.full((n, 8, 3), 0.8, np.float16)
            sid = f'src{i // 3}'
            samples.append({'id': str(i), 'source_id': sid, 'task': ['qa', 'summary'][i % 2],
                            'model': 'llama-2-7b-chat', 'spans': spans})
            toks.append([(a, b, float(q)) for (a, b), q in zip(offsets, p)])
            records[str(i)] = {'gen_offsets': np.array(offsets, np.int32), 'conf': conf, 'look': look}
        res = fusion.run(samples, {'weak': toks}, records,
                         versions=['text, probability only', 'text', 'text+confidence'], folds=3,
                         log=lambda *_: None)
        v = res['weak']['versions']
        self.assertGreater(v['text+confidence']['f1_tuned'], v['text']['f1_tuned'] + 10)
        self.assertGreater(v['text+confidence']['ap'], v['text']['ap'] + 10)
        self.assertIn('gain_tuned_ci95', v['text+confidence'])
        self.assertIn('as published', v)
        # the published probabilities and their re-learned version rank tokens about equally well
        self.assertLess(abs(v['as published']['ap'] - v['text, probability only']['ap']), 10)
        self.assertEqual(set(v['text']['cutoffs']), {0, 1, 2})
        self.assertIn('| weak | text+confidence |', fusion.table(res))


if __name__ == '__main__':
    unittest.main()
