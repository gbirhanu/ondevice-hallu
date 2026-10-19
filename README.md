# On-device hallucination detection

Can a checker small enough for a phone mark the words of an AI answer that the user's own
notes do not support? This repository scores small span detectors on RAGTruth, shrinks them
to 8-bit, and times them on a CPU. It is separate from the earlier `tse` / `hallu_poc` work.

Concept note: "Checking AI answers on the phone: a lightweight hallucination detector for offline RAG".

## What it measures

| Measure | How |
| --- | --- |
| Word-level accuracy | Span F1 on the 2,700 RAGTruth test answers, scored exactly as LettuceDetect scores it, at the plain 0.5 cut-off (no tuning). QA and summary are also reported on their own. 95% intervals resample test sources. |
| What 8-bit costs | Span F1 of the 8-bit ONNX model against the full-precision model on a fixed subset of 300 test answers (the same answers for every model). |
| Size | ONNX file size in full precision and in 8-bit. |
| Speed | Time per answer on the CPU, one answer at a time (as a phone app works), with 1 and 2 threads, ONNX Runtime (the engine used on Android). |

## 8-bit files

The weights are stored in 8 bits, in blocks of 32 numbers with one scale each (the word
embeddings too), and the arithmetic stays in float (`--recipe w8`, the default). The first
attempt (`--recipe dyn`, onnxruntime's dynamic quantization) also rounded every layer's input
to 8 bits with one scale for the whole input. ModernBERT-type models feed some layers inputs
with a few values thousands of times larger than the rest; one scale then rounds the ordinary
values to zero, and span F1 fell by 21 to 50 points. `--recipe mix` keeps 8-bit arithmetic only
in the layers whose inputs are safe; it is faster but less exact. `export` reports, per model,
how close the 8-bit file is to the float file and how many layers have such extreme inputs.

## Run it on Colab (first experiment: the size ladder)

1. Create an empty **public** GitHub repository `gbirhanu/ondevice-hallu` and push this folder to it.
2. Open <https://colab.research.google.com/github/gbirhanu/ondevice-hallu/blob/main/colab/size_ladder.ipynb>,
   choose a T4 GPU, and run the cells in order. It scores the published TinyLettuce 17M/32M/68M and LettuceDetect base/large
models, exports them to 8-bit, times them, and writes `results/table.md` and `results/ladder.png`
to `MyDrive/ondevice_hallu/work`. Every step skips work that is already saved, so a stopped
session continues where it left off. ONNX files stay on Colab's own disk (they are remade
when needed) so Drive does not fill up.

## First test of the generator-aware checker

`colab/signals_test.ipynb` asks whether a small checker gets better when it also reads the
answering model's own signals. It re-reads RAGTruth's test answers of the open models
(Llama-2-7B/13B-chat, Mistral-7B-Instruct) with the model that wrote them, in 4-bit, and records
per answer token: confidence (log-probability, entropy, gaps to the top token) and the lookback
ratio of every layer (`odh/signals.py`). It then combines these with the text checkers' scores
from the size ladder and compares text only, + confidence, + attention and + both, all trained
and scored the same way (5-fold cross-validation split by source; `odh/fusion.py`). Results:
`results/fusion.md`: token average precision (cut-off free), span F1 at 0.5, and span F1 with
the cut-off chosen on the other folds. It needs the size ladder's `preds` folder.

## Teacher-student test (distillation)

`colab/distill.ipynb` asks whether a small checker learns more from a big one than from the
human labels alone. The teacher is LettuceDetect-large (396M); the students are Ettin 17M and 68M,
the encoders TinyLettuce is built on. The teacher gives (1) soft labels, its probability for every
word of RAGTruth's training answers, used half and half with the human label, and (2) labels for
new answers that small, phone-sized models (Qwen2.5 0.5B/1.5B, SmolLM2 1.7B) write to RAGTruth's
training prompts (never the test prompts). Three students per size are trained the same way:
human labels only (`hard`), + soft labels (`soft`), + new answers (`kd`). The two models cut words
into tokens differently, so the teacher's probabilities are matched to the student's tokens by
character overlap (`odh/distill.py`). Results: `results/distill.md`: span F1 at 0.5 and at the
cut-off chosen on the dev answers, each student against the human-labels-only student of the
same size that saw about as many answers in training (so extra training alone cannot explain a
gain) and against the published TinyLettuce of the same size, with 95% intervals.

## Two tests of what is new

`colab/novelty.ipynb` tests the two claims a reviewer will ask about.
**A. Does it matter who writes the new answers?** Large models (Qwen2.5-7B, OLMo-2-7B, Mistral-7B
v0.3, in 4-bit) write answers to the same training questions; the same teacher marks them, and a
student is trained exactly like `kd17m` with the same number of new answers (`kdlarge17m`).
**B. Our 396M teacher against a large language model as the labeller** (the PsiloQA way): by default a
free open model, Qwen2.5-7B-Instruct in 4-bit on the Colab GPU, lists the unsupported parts of the same
small-model answers (`odh/llmlabel.py`; a paid model behind an OpenAI-compatible API, such as GPT-4o,
also works, with PsiloQA's `[HAL]` tags); a student learns from them (`llm17m`), next to one trained on our teacher's labels
rounded to yes/no (`kdhard17m`). The LLM is also scored against people on 300 RAGTruth test answers.
All students are scored on RAGTruth and on English PsiloQA answers grouped by the size of the model
that wrote them (`odh/psilo.py`; PsiloQA is CC-BY-4.0 and never used for training); `compare` gives the
differences with 95% intervals.

## Commands

```
python run.py models                               # find the published small detectors
python run.py score     --name tiny68m             # span F1 on the full test set
python run.py export    --name tiny68m             # ONNX + 8-bit (--recipe w8 | mix | dyn)
python run.py check8bit --name tiny68m --n 300     # F1 change from 8-bit
python run.py bench     --name tiny68m --threads 1 2 --where colab
python run.py table                                # results/table.md + results/ladder.png
python run.py train     --name mine68m --base jhu-clsp/ettin-encoder-68m   # our own model
python run.py signals   --generator llama-2-7b-chat    # answering-model signals (GPU, 4-bit)
python run.py fusion                               # do the signals add to the text checkers?
python run.py teach     --name ld_large            # teacher's word probabilities (training + new answers)
python run.py generate  --generator qwen2.5-1.5b   # new answers from a small model (GPU)
python run.py train     --name kd17m --base jhu-clsp/ettin-encoder-17m --teacher ld_large --extra qwen2.5-1.5b
python run.py distill                              # table of the teacher-student test
python run.py generate  --generator qwen2.5-7b     # answers from a large model (4-bit), for test A
python run.py llmlabel  --name qwen7b --local qwen2.5-7b   # a free open LLM labels the new answers (GPU)
python run.py llmlabel  --name qwen7b --local qwen2.5-7b --on test   # the LLM against people, 300 test answers
python run.py psilo                                # PsiloQA scores by answer-model size
python run.py compare   --pairs kd17m:kdlarge17m kd17m:llm17m   # differences with 95% intervals
```

All take `--data-dir` (RAGTruth) and `--work-dir`. Timing on your own laptop: copy the
`work` folder from Drive, then `python run.py bench --name tiny68m --threads 1 4 --where laptop`.

## Code

| File | What it does |
| --- | --- |
| `odh/data.py` | Loads RAGTruth; dev split by source; fixed subsets for CPU checks |
| `odh/metrics.py` | Span F1 (LettuceDetect-compatible) and bootstrap intervals |
| `odh/spans.py` | Token probabilities to character spans (0.5 cut-off) |
| `odh/model.py` | Load, predict, train token classifiers (ModernBERT, Ettin, TinyLettuce, LettuceDetect) |
| `odh/cpu.py` | ONNX export, 8-bit quantization, CPU prediction and timing |
| `odh/signals.py` | The answering model's confidence and attention for every answer token |
| `odh/fusion.py` | Text checker + answering-model signals: cross-validated comparison |
| `odh/distill.py` | Teacher-student training: new answers from small models, teacher targets for the student |
| `odh/llmlabel.py` | A large language model as the labeller (PsiloQA-style `[HAL]` tags, any OpenAI-compatible API) |
| `odh/psilo.py` | PsiloQA as a second test set, grouped by the size of the model that wrote each answer |
| `run.py` | The commands above |

Checks without a GPU or downloads: `python -m pytest tests -q` and, for an end-to-end run,
`python tools/make_tiny.py --data-dir RAGTruth --out work_test/tiny` then the commands with
`--model work_test/tiny --limit 60`. The scorer was checked against the earlier pipeline,
which reproduces the published LettuceDetect-large result (58.93) exactly.
