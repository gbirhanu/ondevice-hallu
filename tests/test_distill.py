"""Checks of the teacher-student code; CPU, no downloads. Run: python -m pytest tests -q"""
import gzip
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from odh import distill, model  # noqa: E402
from test_core import tiny_config, tiny_tokenizer  # noqa: E402
from test_signals import tiny_generator  # noqa: E402
from transformers import ModernBertForTokenClassification  # noqa: E402


class Targets(unittest.TestCase):
    def test_teacher_probabilities_follow_characters(self):
        # student tokens cut the words differently from the teacher's
        teacher = [(0, 5, 0.2), (6, 11, 0.8), (12, 15, 0.4)]
        student = [(0, 3), (3, 8), (8, 11), (12, 15), (20, 22)]
        got = distill.soft_targets(student, teacher)
        self.assertAlmostEqual(got[0], 0.2)
        self.assertAlmostEqual(got[1], 0.5)          # shares characters with both of the first two
        self.assertAlmostEqual(got[2], 0.8)
        self.assertAlmostEqual(got[3], 0.4)
        self.assertAlmostEqual(got[4], 0.4)          # no overlap: the nearest teacher token

    def test_targets_mix_human_and_teacher(self):
        offsets = [(0, 2), (3, 5)]
        teacher = [(0, 2, 0.6), (3, 5, 0.1)]
        both = distill.targets_for({'spans': [(0, 2)], 'teacher': teacher}, offsets, alpha=0.5)
        self.assertEqual([round(x, 3) for x in both], [0.8, 0.05])
        self.assertEqual(distill.targets_for({'spans': [(0, 2)]}, offsets), [1.0, 0.0])
        self.assertEqual(distill.targets_for({'spans': None, 'teacher': teacher}, offsets), [0.6, 0.1])
        with self.assertRaises(ValueError):
            distill.targets_for({'spans': None}, offsets)

    def test_best_cutoff(self):
        samples = [{'id': '1', 'source_id': 'a', 'task': 'qa', 'spans': [(0, 4)]}]
        toks = [[(0, 4, 0.3), (5, 9, 0.1)]]
        cut, f1, f05 = distill.best_cutoff(toks, samples)
        self.assertEqual((f1, f05), (100.0, 0.0))
        self.assertLess(cut, 0.3)


class Training(unittest.TestCase):
    def setUp(self):
        self.tmp = ROOT / 'work_test' / ('d-' + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_student_trains_on_human_teacher_and_unlabelled_answers(self):
        torch.manual_seed(0)
        base = self.tmp / 'base'
        ModernBertForTokenClassification(tiny_config()).save_pretrained(base)
        tiny_tokenizer().save_pretrained(base)
        prompt = ' '.join(f'w{k}' for k in range(20))
        answer = 'w1 w2 w3 w4 w5'
        teacher = [(0, 2, 0.1), (3, 5, 0.9), (6, 8, 0.9), (9, 11, 0.1), (12, 14, 0.1)]
        train = [{'id': f't{i}', 'source_id': f's{i}', 'task': 'qa', 'prompt': prompt, 'answer': answer,
                  'spans': [(3, 8)], 'teacher': teacher} for i in range(6)]
        train += [{'id': f'x{i}', 'source_id': f's{i}', 'task': 'qa', 'prompt': prompt, 'answer': answer,
                   'spans': None, 'teacher': teacher} for i in range(4)]
        dev = [{'id': 'd', 'source_id': 'd', 'task': 'qa', 'prompt': prompt, 'answer': answer, 'spans': [(3, 8)]}]
        res = model.train(str(base), train, dev, str(self.tmp / 'out'), epochs=2, lr=1e-3, device='cpu',
                          fp16=False, batch_tokens=256, log=lambda *_: None)
        self.assertEqual(len(res['history']), 2)
        self.assertIn(res['dev_cutoff'], list(distill.CUTOFFS))
        self.assertGreaterEqual(res['dev_f1_at_cutoff'], res['dev_f1_05'])
        tok, net = model.load(str(self.tmp / 'out'))
        self.assertEqual(next(net.parameters()).dtype, torch.float32)   # saved in 16-bit, used in 32-bit
        self.assertEqual(len(model.predict(net, tok, dev, fp16=False)[0]), 5)


class ExtraAnswers(unittest.TestCase):
    def setUp(self):
        self.tmp = ROOT / 'work_test' / ('g-' + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_generation_is_batched_resumable_and_skips_long_prompts(self):
        tok, net = tiny_generator()
        tok.padding_side = 'left'
        tok.pad_token = tok.eos_token
        items = [{'source_id': str(i), 'prompt': ' '.join(f'w{k}' for k in range(3 + i))} for i in range(5)]
        out = distill.generate(net, tok, [x['prompt'] for x in items], max_new_tokens=4, max_prompt_tokens=6)
        self.assertEqual([o is None for o in out], [False, False, False, False, True])
        self.assertTrue(all(o['new_tokens'] <= 4 for o in out[:4]))
        folder = self.tmp / 'gen'
        distill.generate_all(net, tok, items, str(folder), chunk=2, max_new_tokens=3, log=lambda *_: None)
        files = sorted(folder.glob('*.json.gz'))
        self.assertEqual(len(files), 3)
        stamp = files[0].stat().st_mtime_ns
        distill.generate_all(net, tok, items, str(folder), chunk=2, max_new_tokens=3, log=lambda *_: None)
        self.assertEqual(files[0].stat().st_mtime_ns, stamp)                # kept, not redone

    def test_unfinished_and_empty_answers_are_left_out(self):
        folder = self.tmp / 'extra'
        folder.mkdir()
        rows = [{'source_id': 'a', 'answer': 'The capital is Addis Ababa.', 'finished': True, 'new_tokens': 8},
                {'source_id': 'b', 'answer': 'and and and', 'finished': False, 'new_tokens': 400},
                {'source_id': 'c', 'answer': '', 'finished': True, 'new_tokens': 0},
                {'source_id': 'zz', 'answer': 'Unknown source here.', 'finished': True, 'new_tokens': 4}]
        with gzip.open(folder / '0000.json.gz', 'wt', encoding='utf-8') as fh:
            json.dump(rows, fh)
        sources = {k: {'task': 'qa', 'prompt': f'prompt {k}'} for k in 'abc'}
        extra, dropped = distill.load_extra(str(folder), 'small', sources)
        self.assertEqual(dropped, 3)
        self.assertEqual(extra[0]['id'], 'small/a')
        self.assertIsNone(extra[0]['spans'])
        self.assertEqual(extra[0]['prompt'], 'prompt a')

    def test_one_prompt_per_training_source(self):
        samples = [{'id': str(i), 'source_id': str(i // 6), 'task': 'qa', 'prompt': f'p{i // 6}'} for i in range(18)]
        items = distill.extra_prompts(samples)
        self.assertEqual([x['source_id'] for x in items], ['0', '1', '2'])
        self.assertEqual(items[1]['prompt'], 'p1')


if __name__ == '__main__':
    unittest.main()
