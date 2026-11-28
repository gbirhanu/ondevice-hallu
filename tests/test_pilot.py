"""Checks of the pilot test set (step 16) and of "pilotgen" (step 17) with a made-up answering model.
CPU, no downloads. Run: python -m pytest tests -q"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402
from odh import distill, pilot  # noqa: E402


class PilotFile(unittest.TestCase):
    def test_the_questions_are_final_and_complete(self):
        qs = pilot.load_questions()
        self.assertEqual(len(qs), 67)
        self.assertEqual(len({q['id'] for q in qs}), 67)
        self.assertEqual(sum(q['domain'] == 'course' for q in qs), 34)
        self.assertEqual(sum(q['domain'] == 'health' for q in qs), 33)
        for q in qs:
            self.assertIn(q['answerable'], ('yes', 'partly', 'no'))
            self.assertTrue(q['question'].endswith('?'), q['id'])
            self.assertGreater(len(q['passage'].split()), 200, q['id'])
            self.assertTrue(q['source'] and q['licence'] and q['reference_answer'], q['id'])
            doc = ROOT / 'pilot' / 'documents' / q['domain'] / f"{q['doc_id']}.md"
            self.assertIn(' '.join(q['passage'].split()), ' '.join(doc.read_text(encoding='utf-8').split()), q['id'])

    def test_the_prompt_follows_ragtruth(self):
        text = pilot.prompt_of('What is X?', 'X is a letter.')
        self.assertTrue(text.startswith('Briefly answer the following question:\nWhat is X?\nBear in mind that your '
                                        'response should be strictly based on the following passage:\npassage 1:X is'))
        self.assertTrue(text.endswith('please reply with: "Unable to answer based on given passages."\noutput:'))
        self.assertTrue(pilot.is_refusal('Unable to answer based on given passages.'))
        self.assertTrue(pilot.is_refusal('I am unable to answer based on given  passage.'))
        self.assertFalse(pilot.is_refusal('X is a letter.'))


def fake_generate(model, tok, prompts, max_new_tokens=450, seed=0, **kwargs):
    """Answers 'Answer to <question>'. A prompt about logic bombs stops at the token limit on the first try."""
    out = []
    for p in prompts:
        question = p.split('\n')[1]
        if 'antivirus' in question:
            out.append({'answer': 'Unable to answer based on given passages.', 'finished': True, 'new_tokens': 9})
        elif 'logic bomb' in question and seed == 1000:
            out.append({'answer': 'loop ' * 50, 'finished': False, 'new_tokens': max_new_tokens})
        else:
            out.append({'answer': f'Answer to {question}', 'finished': True, 'new_tokens': 12})
    return out


class PilotAnswers(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / 'work_test' / ('p-' + uuid.uuid4().hex)
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        self.p = run.paths(str(self.work))
        self.args = argparse.Namespace(questions=None, generators=['qwen2.5-0.5b', 'smollm2-1.7b'], redo=False,
                                       attempts=3, max_new_tokens=450)

    def test_every_question_gets_one_answer_per_model_and_finished_work_is_kept(self):
        loads = []
        with mock.patch.object(distill, 'load_small_generator', lambda hf_id, dev: loads.append(hf_id) or (None, None)), \
                mock.patch.object(distill, 'generate', fake_generate), mock.patch.object(run, 'device', lambda: 'cpu'):
            run.cmd_pilotgen(self.args, self.p)
            rows = list(pilot.load_answers(Path(self.p['pilot']) / 'answers.jsonl').values())
            self.assertEqual(len(rows), 67 * 2)
            self.assertEqual(loads, ['Qwen/Qwen2.5-0.5B-Instruct', 'HuggingFaceTB/SmolLM2-1.7B-Instruct'])
            bomb = [r for r in rows if r['question_id'] == 'c03-q2']
            self.assertTrue(all(r['finished'] and r['attempts'] == 2 for r in bomb))     # written again
            self.assertTrue(all(r['attempts'] == 1 for r in rows if r['question_id'] != 'c03-q2'))
            self.assertTrue(rows[0]['prompt'].startswith('Briefly answer the following question:'))
            run.cmd_pilotgen(self.args, self.p)                                          # nothing new to do
            self.assertEqual(len(loads), 2)
        stats = json.loads((Path(self.p['results']) / 'pilot_answers.json').read_text())['generators']
        self.assertEqual(stats['qwen2.5-0.5b']['refused']['no'], 1)                       # the antivirus question
        self.assertEqual(stats['qwen2.5-0.5b']['retried'], 1)
        self.assertEqual(stats['qwen2.5-0.5b']['asked'], {'yes': 51, 'partly': 8, 'no': 8})
        self.assertIn('| qwen2.5-0.5b | 67 |', (Path(self.p['results']) / 'pilot_answers.md').read_text())

    def test_an_unknown_model_is_refused(self):
        self.args.generators = ['no-such-model']
        with self.assertRaisesRegex(SystemExit, 'Unknown answering model'):
            run.cmd_pilotgen(self.args, self.p)


if __name__ == '__main__':
    unittest.main()
