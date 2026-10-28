"""Checks of run.py's table-making helpers on tiny made-up results. CPU, no downloads.
Run: python -m pytest tests -q"""
import gzip
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402

SAMPLES = [
    {'id': 'a', 'source_id': 's1', 'task': 'qa', 'answer': 'one two three', 'spans': [(4, 7)]},
    {'id': 'b', 'source_id': 's2', 'task': 'qa', 'answer': 'four five', 'spans': [(5, 9)]},
]
TOKENS = {'a': [(0, 3, 0.1), (4, 7, 0.9), (8, 13, 0.1)], 'b': [(0, 4, 0.1), (5, 9, 0.8)]}


def save_preds(path, ids):
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump({'ids': ids, 'tokens': [TOKENS[i] for i in ids]}, fh)


class SameAnswers(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / 'work_test' / ('c-' + uuid.uuid4().hex)
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        self.p = run.paths(str(self.work))

    def test_unreadable_reply_counts_as_no_marks_and_checkers_use_the_same_answers(self):
        save_preds(Path(self.p['preds']) / 'kd17m_test.json.gz', ['a', 'b'])
        save_preds(Path(self.p['preds']) / 'tiny17m_test.json.gz', ['a'])          # another subset: left out
        found = {'a': [(4, 7)]}                                                     # b: reply could not be read
        out = run.same_answers(self.p, SAMPLES, found, ['kd17m', 'tiny17m', 'not_scored'])
        self.assertEqual(out['answers'], 2)
        self.assertEqual(out['labeller']['span_f1'], 60.0)       # 3 of 7 marked characters found, none wrong
        self.assertEqual(list(out['checkers']), ['kd17m'])
        self.assertEqual(out['checkers']['kd17m']['span_f1'], 100.0)
        lo, hi = out['checkers']['kd17m']['minus_labeller_ci95']
        self.assertTrue(0 <= lo <= hi <= 100)


if __name__ == '__main__':
    unittest.main()
