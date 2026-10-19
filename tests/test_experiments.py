"""Checks of the two novelty experiments (large answering models; an LLM labeller) and the PsiloQA
test. CPU, no downloads. Run: python -m pytest tests -q"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import shutil
import sys
import threading
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from odh import distill, llmlabel, psilo  # noqa: E402

ANSWER = 'The capital is Addis Ababa and it has 5 million people.'


class LabellerReplies(unittest.TestCase):
    def test_tags_become_character_spans(self):
        spans = llmlabel.parse('The capital is Addis Ababa and it has [HAL]5 million[/HAL] people.', ANSWER)
        self.assertEqual([ANSWER[s:e] for s, e in spans], ['5 million'])

    def test_small_copying_slips_and_fences_are_tolerated(self):
        reply = '```\nThe capital is  Addis Ababa and it has [HAL] 5 million [/HAL]people.\n```'
        spans = llmlabel.parse(reply, ANSWER)
        self.assertEqual([ANSWER[s:e] for s, e in spans], ['5 million'])

    def test_no_tags_means_no_spans_and_open_tag_runs_to_the_end(self):
        self.assertEqual(llmlabel.parse(ANSWER, ANSWER), [])
        spans = llmlabel.parse('The capital is Addis Ababa [HAL]and it has 5 million people.', ANSWER)
        self.assertEqual([ANSWER[s:e] for s, e in spans], ['and it has 5 million people.'])

    def test_rewritten_or_missing_replies_are_not_used(self):
        self.assertIsNone(llmlabel.parse('Addis Ababa is the capital of Ethiopia, a city of many people.', ANSWER))
        self.assertIsNone(llmlabel.parse(None, ANSWER))

    def test_labelling_is_saved_in_chunks_and_resumes(self):
        tmp = ROOT / 'work_test' / ('l-' + uuid.uuid4().hex)
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        samples = [{'id': str(i), 'prompt': 'p', 'answer': ANSWER} for i in range(5)]
        calls = []

        def ask(s):
            calls.append(s['id'])
            return ANSWER.replace('5 million', '[HAL]5 million[/HAL]'), {'prompt_tokens': 10, 'completion_tokens': 3}
        t_in, t_out = llmlabel.label_all(samples, str(tmp), ask, workers=2, chunk=2, log=lambda *_: None)
        self.assertEqual((t_in, t_out, len(calls)), (50, 15, 5))
        llmlabel.label_all(samples, str(tmp), ask, workers=2, chunk=2, log=lambda *_: None)
        self.assertEqual(len(calls), 5)                                   # nothing asked twice
        self.assertEqual(len(llmlabel.load_replies(str(tmp))), 5)

    def test_api_request_follows_the_openai_format(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen['path'] = self.path
                seen['auth'] = self.headers.get('Authorization')
                seen['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                reply = json.dumps({'choices': [{'message': {'content': 'tagged'}}],
                                    'usage': {'prompt_tokens': 7, 'completion_tokens': 2}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        text, usage = llmlabel.call('gpt-4o', llmlabel.messages('passages', ANSWER),
                                    f'http://127.0.0.1:{server.server_port}/v1', 'key123')
        self.assertEqual((text, usage['prompt_tokens']), ('tagged', 7))
        self.assertEqual(seen['path'], '/v1/chat/completions')
        self.assertEqual(seen['auth'], 'Bearer key123')
        self.assertEqual(seen['body']['temperature'], 0)
        self.assertIn(ANSWER, seen['body']['messages'][1]['content'])


class Labels(unittest.TestCase):
    def test_spans_become_teacher_tokens(self):
        start = ANSWER.index('5 million')
        toks = distill.spans_to_tokens(ANSWER, [(start, start + 9)])
        self.assertEqual(''.join(ANSWER[s:e] for s, e, _ in toks), ANSWER.replace(' ', ''))
        self.assertEqual([ANSWER[s:e] for s, e, q in toks if q == 1.0], ['5', 'million'])
        cut = distill.spans_to_tokens('abcdef', [(2, 4)])                 # a span inside a word
        self.assertEqual([(s, e, q) for s, e, q in cut], [(0, 2, 0.0), (2, 4, 1.0), (4, 6, 0.0)])

    def test_hard_labels(self):
        self.assertEqual(distill.harden([(0, 1, 0.7), (2, 3, 0.2)]), [(0, 1, 1.0), (2, 3, 0.0)])

    def test_large_answering_models_are_known(self):
        self.assertTrue(set(distill.LARGE_GENERATORS) <= set(distill.ANSWER_MODELS))
        self.assertFalse(set(distill.DEFAULT_GENERATORS) & distill.LARGE)


class PsiloQA(unittest.TestCase):
    def test_model_sizes_from_names(self):
        cases = {'TinyLlama/TinyLlama-1.1B-Chat-v1.0': 1.1, 'HuggingFaceTB/SmolLM2-135M-Instruct': 0.135,
                 'Qwen/Qwen2.5-0.5B-Instruct': 0.5, 'Qwen/Qwen2.5-7B-Instruct': 7.0,
                 'meta-llama/Llama-3.1-8B-Instruct': 8.0, 'google/gemma-2-9b-it': 9.0,
                 'CohereForAI/aya-23-8B': 8.0, 'microsoft/Phi-3-mini-4k-instruct': 3.8,
                 'mistralai/Mistral-7B-Instruct-v0.3': 7.0, 'bigscience/bloomz-560m': 0.56}
        for name, size in cases.items():
            self.assertAlmostEqual(psilo.model_size(name), size, msg=name)
        self.assertEqual(psilo.bucket('Qwen/Qwen2.5-1.5B-Instruct'), 'up to 2B')
        self.assertEqual(psilo.bucket('microsoft/Phi-3-mini-4k-instruct'), '2-4B')
        self.assertEqual(psilo.bucket('google/gemma-2-9b-it'), 'over 4B')
        self.assertEqual(psilo.bucket('some/model'), 'unknown size')

    def test_records_become_samples(self):
        recs = [{'id': 'a', 'lang': 'en', 'wiki_title': 'Addis Ababa', 'wiki_passage': 'Addis Ababa is ...',
                 'question': 'What is ...?', 'llm_answer': ANSWER, 'labels': [[37, 46], [50, 99]],
                 'llm_checkpoint': 'Qwen/Qwen2.5-1.5B-Instruct'},
                {'id': 'b', 'lang': 'fi', 'wiki_title': 'x', 'wiki_passage': 'y', 'question': 'z',
                 'llm_answer': 'w', 'labels': [], 'llm_checkpoint': 'm'}]
        out = psilo.samples_from(recs, 'en', 'test')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['id'], 'test:a')
        self.assertEqual(out[0]['spans'], [(37, 46)])                    # the out-of-range span is dropped
        self.assertEqual(out[0]['size_group'], 'up to 2B')
        self.assertIn('Addis Ababa is ...', out[0]['prompt'])


if __name__ == '__main__':
    unittest.main()
