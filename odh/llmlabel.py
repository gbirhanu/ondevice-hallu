"""A large language model as the labeller (the PsiloQA way), to compare with our 396M span teacher.

The model gets what the answering model saw (passages + question, i.e. RAGTruth's prompt) and the
answer, and marks every unsupported span, in one of two styles:
  tags    the answer copied back with [HAL]...[/HAL] around unsupported spans, as in PsiloQA;
  quotes  a JSON list of the unsupported parts, copied from the answer (far fewer tokens to write,
          so it suits a free open model running on the Colab GPU).
Either way the spans are mapped back to character positions in the original answer.

Two ways to run the labeller:
  free     an open model (e.g. Qwen2.5-7B-Instruct, 4-bit) on the Colab GPU: local_replies();
  paid     any OpenAI-compatible chat API (OpenAI, Gemini, OpenRouter, ...): call() + label_all().

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
QUOTES = """Below is the input an AI assistant received (reference passages and the question or task) and the
answer it wrote. Find every part of the answer that the reference does not support: content that
contradicts the reference, and content that adds facts the reference does not contain. Do not list
content that the reference supports, faithful paraphrases, or wording without factual content.

Reply with a JSON list of those parts, each copied exactly from the answer and as short as possible,
for example ["16-bit", "64 KB"]. Reply [] if everything is supported. Reply with the list only.

### Input the assistant received
{prompt}

### Answer to check
{answer}"""
TAG = re.compile(r'\[/?HAL\]')


def messages(prompt, answer, style='tags'):
    text = (INSTRUCTION if style == 'tags' else QUOTES).format(prompt=prompt, answer=answer)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]


def parse_quotes(reply, answer):
    """Character spans of the quoted parts, or None when the reply holds no list. Each quote is found
    in the answer (exactly, else ignoring case); a quote that is not in the answer is dropped."""
    if reply is None:
        return None
    m = re.search(r'\[.*\]', reply, re.S)
    if not m:
        return None
    try:
        quotes = json.loads(m.group())
    except ValueError:
        quotes = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group())
    if not isinstance(quotes, list):
        return None
    spans, low = [], answer.lower()
    for q in quotes:
        if not isinstance(q, str) or not q.strip():
            continue
        q = q.strip()
        start = answer.find(q)
        if start < 0:
            start = low.find(q.lower())
        while start >= 0 and any(s <= start < e for s, e in spans):      # repeated quote: next occurrence
            nxt = low.find(q.lower(), start + 1)
            start = nxt
        if start >= 0:
            spans.append((start, start + len(q)))
    return sorted(spans)


def parse_reply(reply, answer, style='tags'):
    return parse(reply, answer) if style == 'tags' else parse_quotes(reply, answer)


def chat_texts(tok, samples, style):
    """The labelling request in the model's own chat format (system turn folded into the user turn
    for models whose template has no system role)."""
    out = []
    for s in samples:
        msgs = messages(s['prompt'], s['answer'], style)
        if not getattr(tok, 'chat_template', None):              # a plain model: just the text
            out.append(msgs[0]['content'] + '\n\n' + msgs[1]['content'] + '\n')
            continue
        try:
            out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            merged = [{'role': 'user', 'content': msgs[0]['content'] + '\n\n' + msgs[1]['content']}]
            out.append(tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True))
    return out


def local_replies(model, tok, samples, folder, style='quotes', chunk=64, max_new_tokens=None, log=print):
    """Replies of an open model on this machine's GPU (greedy), saved like label_all's."""
    from odh.distill import generate
    max_new_tokens = max_new_tokens or (200 if style == 'quotes' else 700)
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    for c in range(0, len(samples), chunk):
        path = os.path.join(folder, f'{c // chunk:04d}.json.gz')
        if os.path.exists(path):
            continue
        part = samples[c:c + chunk]
        got = generate(model, tok, chat_texts(tok, part, style), max_new_tokens=max_new_tokens, chat=False,
                       sample=False, max_prompt_tokens=4500)
        rows = [{'id': s['id'], 'reply': g['answer'] if g else None} for s, g in zip(part, got)]
        with gzip.open(path + '.tmp', 'wt', encoding='utf-8') as fh:
            json.dump(rows, fh)
        os.replace(path + '.tmp', path)
        log(f'  {min(c + chunk, len(samples))}/{len(samples)} answers labelled ({time.time() - t0:.0f}s)')


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
