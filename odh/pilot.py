"""The pilot test set (PLAN.md steps 16-20): questions over course notes and health guidelines,
answered by phone-sized models and marked word by word by two people. It is a test set only.

pilot/questions.jsonl holds the questions with their passages (step 16). Step 17 asks each phone-sized
model every question with RAGTruth's question-answering prompt, so that the answers look like the ones
the checkers were trained and tested on, and saves the answers in <work-dir>/pilot/answers.jsonl.
"""
import json
import os
import re

# The phone-sized models that answer the pilot questions. Granite was added on 28 November because
# Qwen2.5-0.5B replied "Unable to answer based on given passages." to 66 of the 67 questions.
GENERATORS = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'smollm2-1.7b', 'granite-3.3-2b']
QUESTIONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pilot', 'questions.jsonl')
REFUSAL = 'Unable to answer based on given passages.'


def load_questions(path=QUESTIONS):
    """The final questions of the pilot set, in file order."""
    with open(path, encoding='utf-8') as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get('status') == 'final']


def prompt_of(question, passage):
    """RAGTruth's question-answering prompt (as in its source_info.jsonl), with one passage instead of three.
    The last sentence is RAGTruth's own, word for word."""
    return ('Briefly answer the following question:\n'
            f'{question}\n'
            'Bear in mind that your response should be strictly based on the following passage:\n'
            f'passage 1:{passage}\n\n'
            'In case the passages do not contain the necessary information to answer the question, '
            f'please reply with: "{REFUSAL}"\n'
            'output:')


def is_refusal(answer):
    """True when the answer says that the passage does not answer the question (RAGTruth's reply)."""
    return 'unable to answer based on given passage' in ' '.join(answer.lower().split())


def is_bare_refusal(answer):
    """True when the answer is only RAGTruth's refusal sentence. Such an answer makes no claim, so the annotators
    do not mark it; it counts as an answer without errors."""
    return re.sub(r'[\s"\'\u201c\u201d.]+', ' ', answer.lower()).strip() == 'unable to answer based on given passages'


def answer_id(question_id, generator):
    return f'{question_id}/{generator}'


def load_answers(path):
    """{answer id: row} of the answers saved so far."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return {r['id']: r for r in map(json.loads, (line for line in fh if line.strip()))}


def save_answers(path, rows):
    """All answers, one per line, written to a temporary file first so a stopped run never leaves half a file."""
    with open(path + '.tmp', 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(path + '.tmp', path)


def summary(rows, questions, generators):
    """Counts per answering model: answers, unfinished ones, refusals by whether the passage answers the
    question (yes / partly / no), and the mean length in words."""
    label = {q['id']: q['answerable'] for q in questions}
    out = {}
    for g in generators:
        rs = [r for r in rows if r['generator'] == g]
        if not rs:
            continue
        words = [len(r['answer'].split()) for r in rs]
        refused = {a: sum(1 for r in rs if label[r['question_id']] == a and is_refusal(r['answer']))
                   for a in ('yes', 'partly', 'no')}
        asked = {a: sum(1 for r in rs if label[r['question_id']] == a) for a in ('yes', 'partly', 'no')}
        out[g] = {'answers': len(rs), 'to_mark': sum(1 for r in rs if not is_bare_refusal(r['answer'])),
                  'unfinished': sum(1 for r in rs if not r['finished']),
                  'retried': sum(1 for r in rs if r['attempts'] > 1),
                  'refused': refused, 'asked': asked, 'mean_words': round(sum(words) / len(words), 1)}
    return out
