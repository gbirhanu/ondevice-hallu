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

## Run it on Colab (first experiment: the size ladder)

1. Create an empty **public** GitHub repository `gbirhanu/ondevice-hallu` and push this folder to it.
2. Open <https://colab.research.google.com/github/gbirhanu/ondevice-hallu/blob/main/colab/size_ladder.ipynb>,
   choose a T4 GPU, and run the cells in order. It scores the published TinyLettuce 17M/32M/68M and LettuceDetect base/large
models, exports them to 8-bit, times them, and writes `results/table.md` and `results/ladder.png`
to `MyDrive/ondevice_hallu/work`. Every step skips work that is already saved, so a stopped
session continues where it left off. ONNX files stay on Colab's own disk (they are remade
when needed) so Drive does not fill up.

## Commands

```
python run.py models                               # find the published small detectors
python run.py score     --name tiny68m             # span F1 on the full test set
python run.py export    --name tiny68m             # ONNX + 8-bit
python run.py check8bit --name tiny68m --n 300     # F1 change from 8-bit
python run.py bench     --name tiny68m --threads 1 2 --where colab
python run.py table                                # results/table.md + results/ladder.png
python run.py train     --name mine68m --base jhu-clsp/ettin-encoder-68m   # our own model
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
| `run.py` | The commands above |

Checks without a GPU or downloads: `python -m pytest tests -q` and, for an end-to-end run,
`python tools/make_tiny.py --data-dir RAGTruth --out work_test/tiny` then the commands with
`--model work_test/tiny --limit 60`. The scorer was checked against the earlier pipeline,
which reproduces the published LettuceDetect-large result (58.93) exactly.
