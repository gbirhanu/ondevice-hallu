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


WORDS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]          # 'aa bb cc dd ee ff'
TEST = [{'id': f't{i}', 'source_id': f's{i // 2}', 'task': 'qa', 'answer': 'aa bb cc dd ee ff',
         'spans': [(3, 8)] if i % 2 == 0 else []} for i in range(12)]
GOLD = [{'id': f'q{i}', 'source_id': f'art{i // 2}', 'task': 'qa', 'model': 'm', 'answer': 'aa bb',
         'size_group': 'up to 2B' if i % 2 else 'over 4B', 'spans': [(0, 2)]} for i in range(8)]


def marks(marked, extra=()):
    """Token probabilities for every test answer: words in `marked` on answers with an error; extra = (answer, word)."""
    return [[(s, e, 0.9 if (i % 2 == 0 and w in marked) or (i, w) in extra else 0.1) for w, (s, e) in enumerate(WORDS)]
            for i in range(len(TEST))]


class Summary(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / 'work_test' / ('s-' + uuid.uuid4().hex)
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        self.p = run.paths(str(self.work))

    def add(self, name, tokens, psilo_p=None, seed=None, answers=len(TEST), params=17_000_000):
        res, preds = Path(self.p['results']), Path(self.p['preds'])
        (res / f'{name}_score.json').write_text(json.dumps({'answers': answers, 'params': params}))
        if seed is not None:
            (res / f'{name}_training.json').write_text(json.dumps({'seed': seed, 'dev_cutoff': 0.5}))
        with gzip.open(preds / f'{name}_test.json.gz', 'wt', encoding='utf-8') as fh:
            json.dump({'ids': [s['id'] for s in TEST], 'tokens': tokens}, fh)
        if psilo_p is not None:
            with gzip.open(preds / f'{name}_psilo.json.gz', 'wt', encoding='utf-8') as fh:
                json.dump({'ids': [g['id'] for g in GOLD], 'tokens': [[(0, 2, psilo_p), (3, 5, 0.1)] for _ in GOLD]}, fh)

    def test_seed_runs_are_averaged_and_differences_count_only_when_all_runs_agree(self):
        self.add('ld_large', marks({1, 2}), 0.9, params=396_000_000)                  # published teacher
        self.add('tiny17m', marks({1}), 0.9)                                          # published, one fixed run
        self.add('kd17m', marks({1, 2}), 0.9, seed=42)
        self.add('kd17m-s2', marks({1, 2}, extra={(0, 3)}), 0.9, seed=2)
        self.add('kd17m-s3', marks({1, 2}, extra={(1, 0)}), 0.9, seed=3)
        self.add('hard17m-8ep', marks({1}), 0.1, seed=42)
        self.add('hard17m-8ep-s2', marks({1}), 0.1, seed=2)
        self.add('hard17m-8ep-s3', marks({2}), 0.1, seed=3)
        self.add('soft17m', marks({1, 2}), seed=42, answers=5)                        # scored on part of the test set
        self.add('llm17m', marks({2}), 0.9, seed=42)                                  # only one run so far
        md, rec = run.summarize(self.p, TEST, GOLD, pairs=['hard17m-8ep:kd17m', 'tiny17m:kd17m', 'kd17m:llm17m'])
        g = rec['groups']
        self.assertEqual(list(g), ['tiny17m', 'ld_large', 'hard17m-8ep', 'kd17m', 'llm17m'])   # published first
        self.assertEqual(g['kd17m']['runs'], ['kd17m', 'kd17m-s2', 'kd17m-s3'])
        self.assertEqual(g['kd17m']['seeds'], [42, 2, 3])
        self.assertEqual(rec['measures'], ['RAGTruth, 0.5', 'RAGTruth, dev cut-off', 'PsiloQA all',
                                           'PsiloQA up to 2B', 'PsiloQA over 4B'])
        kd = g['kd17m']['measures']['RAGTruth, 0.5']
        self.assertEqual(kd['per_run'][0], 100.0)
        self.assertLess(kd['per_run'][1], 100.0)
        self.assertAlmostEqual(kd['mean'], round(sum(kd['per_run']) / 3, 2), places=2)
        self.assertIsNotNone(kd['sd'])
        self.assertIsNone(g['tiny17m']['measures']['RAGTruth, 0.5']['sd'])
        self.assertEqual(g['kd17m']['share_of_teacher_pct']['PsiloQA all'], 100.0)
        s1 = rec['pairs'][0]
        self.assertEqual(s1['matched'], [('hard17m-8ep', 'kd17m'), ('hard17m-8ep-s2', 'kd17m-s2'),
                                         ('hard17m-8ep-s3', 'kd17m-s3')])                # matched by seed
        self.assertEqual(s1['measures']['RAGTruth, 0.5']['counts'], 'yes, B better')
        self.assertEqual(s1['measures']['PsiloQA all']['counts'], 'yes, B better')
        tiny = rec['pairs'][1]                                                          # the fixed run, three times
        self.assertEqual([a for a, _ in tiny['matched']], ['tiny17m'] * 3)
        one = rec['pairs'][2]                                      # llm17m has seed 42 only: one matched pair
        self.assertEqual(one['matched'], [('kd17m', 'llm17m')])
        self.assertEqual(one['measures']['RAGTruth, 0.5']['counts'], 'needs 3 runs')
        self.assertIn('| hard17m-8ep → kd17m | RAGTruth, 0.5 |', md)

    def test_a_difference_that_flips_between_runs_does_not_count(self):
        self.add('a', marks({1}), seed=42)
        self.add('a-s2', marks({1}), seed=2)
        self.add('a-s3', marks({1}), seed=3)
        self.add('b', marks({1, 2}), seed=42)
        self.add('b-s2', marks({1, 2}), seed=2)
        self.add('b-s3', marks(set()), seed=3)                                       # this run marks nothing
        _, rec = run.summarize(self.p, TEST, None, pairs=['a:b'])
        m = rec['pairs'][0]['measures']['RAGTruth, 0.5']
        self.assertEqual(len(m['per_run']), 3)
        self.assertTrue(m['per_run'][0] > 0 > m['per_run'][2])
        self.assertEqual(m['counts'], 'no')
        self.assertEqual(rec['measures'], ['RAGTruth, 0.5', 'RAGTruth, dev cut-off'])  # no PsiloQA gold


class SeedRuns(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / 'work_test' / ('r-' + uuid.uuid4().hex)
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        self.p = run.paths(str(self.work))
        self.recipe = {'name': 'kd17m', 'base': 'jhu-clsp/ettin-encoder-17m', 'epochs': 5, 'lr': 2e-5, 'seed': 42,
                       'teacher': 'ld_large', 'alpha': 0.5, 'extra': {'qwen2.5-0.5b': 2220}, 'train_answers': 13584,
                       'extra_answers': 2220, 'dev_answers': 1506}
        (Path(self.p['results']) / 'kd17m_training.json').write_text(json.dumps(self.recipe))

    def test_names_with_a_seed(self):
        self.assertEqual(run.base_and_seed('kd17m-s2'), ('kd17m', 2))
        self.assertEqual(run.base_and_seed('kd17m-n4509-s3'), ('kd17m-n4509', 3))
        self.assertEqual(run.base_and_seed('hard17m-8ep'), ('hard17m-8ep', None))

    def test_a_seed_run_must_repeat_the_recipe_with_its_own_seed(self):
        run.same_recipe_as_base(self.p, {**self.recipe, 'name': 'kd17m', 'seed': 7})   # not a seed run: no check
        run.same_recipe_as_base(self.p, {**self.recipe, 'name': 'kd17m-s2', 'seed': 2,
                                         'max_length': 4096, 'batch_tokens': 8192})     # new fields: fine
        with self.assertRaisesRegex(SystemExit, 'add --seed 2'):
            run.same_recipe_as_base(self.p, {**self.recipe, 'name': 'kd17m-s2', 'seed': 42})
        with self.assertRaisesRegex(SystemExit, 'epochs: 5 -> 8'):
            run.same_recipe_as_base(self.p, {**self.recipe, 'name': 'kd17m-s2', 'seed': 2, 'epochs': 8})
        with self.assertRaisesRegex(SystemExit, 'extra'):
            run.same_recipe_as_base(self.p, {**self.recipe, 'name': 'kd17m-s3', 'seed': 3, 'extra': {'qwen2.5-0.5b': 100}})
        with self.assertRaisesRegex(SystemExit, 'has not been trained'):
            run.same_recipe_as_base(self.p, {**self.recipe, 'name': 'kd68m-s2', 'seed': 2})


class Averages(unittest.TestCase):
    def test_one_run_gives_the_plain_interval_and_identical_runs_give_zero(self):
        from odh.metrics import bootstrap, bootstrap_mean
        a, b = run.all_spans(marks({1})), run.all_spans(marks({1, 2}))
        self.assertEqual(bootstrap_mean([b], TEST), bootstrap(b, TEST))
        self.assertEqual(bootstrap_mean([b], TEST, bases=[a]), bootstrap(b, TEST, base=a))
        self.assertEqual(bootstrap_mean([a, b], TEST, bases=[a, b]), (0.0, 0.0))
        with self.assertRaises(ValueError):
            bootstrap_mean([a, b], TEST, bases=[a])

    def test_answer_level_f1(self):
        from odh.metrics import answer_f1
        self.assertEqual(answer_f1(run.all_spans(marks({1})), TEST), 100.0)       # every answer with an error flagged
        flag_all = run.all_spans(marks(set(), extra={(i, 0) for i in range(12)}))
        self.assertEqual(answer_f1(flag_all, TEST), 66.67)                        # precision 1/2, recall 1


if __name__ == '__main__':
    unittest.main()
