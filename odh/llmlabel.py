"""A large language model as the labeller (the PsiloQA way), to compare with our 396M span teacher.

The model gets what the answering model saw (passages + question, i.e. RAGTruth's prompt) and the
answer, and returns the answer with every unsupported span wrapped in [HAL]...[/HAL], as in
PsiloQA. We map the tags back to character spans of the original answer. Any OpenAI-compatible
chat API works (OpenAI, Google Gemini's OpenAI endpoint, OpenRouter, a local server).

Raw replies are saved in chunk files, so a stopped run continues where it left off.
"""
import difflib
import gzip
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

SYSTEM = ('You are an expert annotator of hallucinations in answers written by AI assistants that use '
          'retrieved documents.')
INSTRUCTION = """Below is the input an AI assistant received (reference passages and the question or task) and the
answer it wrote. Mark every part of the answer that the reference does not support: content that
contradicts the reference, and content that adds facts the reference does not contain. Do not mark
content that the reference supports, faithful paraphrases, or wording without factual content.

Return the answer exactly as given, character for character, with each unsupported span wrapped in
[HAL] and [/HAL]. Mark the shortest spans that cover the unsupported content. If everything is
supported, return the answer unchanged. Return nothing else.

### Input the assistant received
{prompt}

### Answer to annotate
{answer}"""
TAG = re.compile(r'\[/?HAL\]')


def messages(prompt, answer):
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': INSTRUCTION.format(prompt=prompt, answer=answer)}]


def parse(tagged, answer, min_ratio=0.9):
    """Character spans of the original answer from a [HAL]-tagged copy, or None when the copy is too
    different from the answer to trust (it was rewritten rather than copied)."""
    if tagged is None:
        return None
    tagged = tagged.strip()
    if tagged.startswith('```'):
        tagged = re.sub(r'^```[a-zA-Z]*\n?|\n?```$', '', tagged).strip()
    plain, spans, depth, start = [], [], 0, 0
    pos = 0
    for m in TAG.finditer(tagged):
        plain.append(tagged[pos:m.start()])
        here = sum(len(x) for x in plain)
        if m.group() == '[HAL]':
            if depth == 0:
                start = here
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0 and here > start:
                spans.append((start, here))
        pos = m.end()
    plain.append(tagged[pos:])
    plain = ''.join(plain)
    if depth > 0 and len(plain) > start:                 # a tag left open runs to the end
        spans.append((start, len(plain)))
    sm = difflib.SequenceMatcher(None, plain, answer, autojunk=False)
    if sm.ratio() < min_ratio:
        return None
    blocks = sm.get_matching_blocks()

    def to_answer(i):
        for a, b, n in blocks:
            if a <= i < a + n:
                return b + (i - a)
        later = [b for a, b, n in blocks if a >= i and n]
        return later[0] if later else len(answer)

    out = []
    for s, e in spans:
        a, b = to_answer(s), to_answer(e - 1) + 1
        while a < b and answer[a].isspace():
            a += 1
        while b > a and answer[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
    return out


def call(model, msgs, api_base, api_key, max_tokens=1024, timeout=120, tries=6):
    """One chat completion from an OpenAI-compatible API; retries with back-off on errors."""
    import requests
    url = api_base.rstrip('/') + '/chat/completions'
    body = {'model': model, 'messages': msgs, 'temperature': 0, 'max_tokens': max_tokens}
    wait = 2.0
    for attempt in range(tries):
        try:
            r = requests.post(url, json=body, timeout=timeout,
                              headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
            if r.status_code == 200:
                data = r.json()
                usage = data.get('usage') or {}
                return data['choices'][0]['message']['content'], usage
            if r.status_code in (400, 401, 403, 404):
                raise RuntimeError(f'API error {r.status_code}: {r.text[:300]}')
        except RuntimeError:
            raise
        except Exception:                                  # network trouble: try again
            pass
        time.sleep(wait)
        wait = min(wait * 2, 60)
    return None, {}


def label_all(samples, folder, ask, workers=8, chunk=100, log=print):
    """Replies for every sample (id -> tagged text), saved as folder/NNNN.json.gz. ask(sample) returns
    (text, usage). Existing chunk files are kept."""
    os.makedirs(folder, exist_ok=True)
    t0, tokens_in, tokens_out = time.time(), 0, 0
    for c in range(0, len(samples), chunk):
        path = os.path.join(folder, f'{c // chunk:04d}.json.gz')
        if os.path.exists(path):
            continue
        part = samples[c:c + chunk]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(ask, part))
        rows = []
        for s, (text, usage) in zip(part, results):
            rows.append({'id': s['id'], 'reply': text})
            tokens_in += usage.get('prompt_tokens', 0) or 0
            tokens_out += usage.get('completion_tokens', 0) or 0
        with gzip.open(path + '.tmp', 'wt', encoding='utf-8') as fh:
            json.dump(rows, fh)
        os.replace(path + '.tmp', path)
        log(f'  {min(c + chunk, len(samples))}/{len(samples)} answers labelled ({time.time() - t0:.0f}s; '
            f'{tokens_in:,} tokens in, {tokens_out:,} out this run)')
    return tokens_in, tokens_out


def load_replies(folder):
    out = {}
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith('.json.gz'):
                with gzip.open(os.path.join(folder, name), 'rt', encoding='utf-8') as fh:
                    out.update({r['id']: r['reply'] for r in json.load(fh)})
    return out
