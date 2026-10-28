# How we evaluate

**Fixed on 28 October 2026, before the repeated training runs of steps 6 and 7.** · Gemechu Birhanu, one
colleague, Claude (assistant). This file is step 2 of [PLAN.md](PLAN.md).

This file says how every result is measured and when a result counts. We write it down before the new runs so
that we cannot bend the rules to fit the results. A change after this date goes into
[Changes](#9-changes-after-28-october-2026) with its date and reason, and both of us must agree to it first.

## 1. The questions

1. **RQ1.** Does a teacher (a larger published checker), together with new answers written by small models,
   reliably improve checkers under 100M parameters, on RAGTruth and on answers not seen in training?
2. **RQ2.** Why do small models' answers teach better than 7B models' answers: the number of errors in them,
   the number of answers, or which models write them?
3. **RQ3.** Does the gain survive 8-bit compression, and is the checker small and fast enough for a phone?
4. **RQ4.** Does it hold on answers that phone-sized models write over course notes and health guidelines,
   marked by people?

## 2. The data

No choice of any kind is made on a test set. Test sets give final numbers only.

| Set | What it is | Size | Who marked the errors | Used for |
|---|---|---|---|---|
| RAGTruth test | Answers by six models (GPT-3.5, GPT-4, Llama-2-7B/13B/70B-chat, Mistral-7B) to 450 sources: question answering, summaries, data-to-text, 900 answers each | 2,700 answers (943 with errors) | people | main test set |
| PsiloQA (English) | Answers by 24 open models (135M–9B) to questions about Wikipedia passages, written without the passage; a fixed subset (chosen by a hash of the answer id, the same for every checker) | 4,000 answers: up to 2B 862, over 4B 2,640, size unknown 498 | GPT-4o | second test set, by size of the answering model (CC-BY-4.0) |
| Pilot set (steps 16–19) | Answers by Qwen2.5-0.5B, Qwen2.5-1.5B and SmolLM2-1.7B to about 50 questions over course notes and open health guidelines | about 150 answers | the two of us, separately, then agreed | third test set, the real use |
| RAGTruth dev | 10% of RAGTruth's training sources, split by source | 1,506 answers (251 sources) | people | every choice: best epoch, the dev cut-off, the selections of steps 10 and 11 |

Nothing is ever trained on PsiloQA or on the pilot set. New answers for training are written only for RAGTruth's
2,264 training sources, never for dev or test sources.

## 3. The checkers compared

| Name | Size | What it is |
|---|---|---|
| `ld_large` | 396M | LettuceDetect-large, published; **our teacher** |
| `ld_base` | 150M | LettuceDetect-base, published |
| `tiny17m`, `tiny68m` | 17M, 68M | TinyLettuce, published; the baselines of our size |
| `hard17m-8ep`, `hard68m-8ep` | 17M, 68M | ours: people's labels only, 8 epochs (sees about as many answers as the distilled runs) |
| `kd17m`, `kd68m` | 17M, 68M | ours: people's labels + the teacher, and 6,719 new answers from small models marked by the teacher |
| `kd17m-n4509`, `kdlarge17m` (and the 68M pair) | 17M, 68M | test A: 4,509 new answers from small models, or from 7B models |
| `kdeq*`, `kdrich17m` | 17M | steps 10 and 11 |
| `selfcheck05`, `selfcheck15`, `qwen7b` | 0.5B, 1.5B, 7B | chat models asked to list the unsupported parts of an answer (step 12 and test B) |

All our students start from the Ettin encoders and use one frozen recipe (PLAN.md, rule R2): AdamW, learning rate
2e-5, 5 epochs (8 for people's labels only), α = 0.5, maximum length 4,096 tokens, batches of 8,192 tokens, best
epoch by dev span F1.

## 4. The measures

**Main measure: span F1.** Computed at the character level exactly as LettuceDetect computes it. For every answer:
the overlap (characters shared by each pair of a marked span and a true error span, summed over all pairs), the
marked characters and the true error characters. These are summed over the whole test set; precision = overlap /
marked, recall = overlap / true, F1 = 2PR / (P + R), in points (0–100). A checker gives every answer token a
probability; tokens above the cut-off are marked, and neighbouring marked tokens form one span.

| Measure | Where | Status |
|---|---|---|
| Span F1 at cut-off 0.5 (no tuning) | RAGTruth, PsiloQA all, pilot | **main** |
| Span F1 at the dev cut-off (each run picks 0.05, 0.10, …, 0.95 on the dev answers) | RAGTruth | secondary |
| Span F1 by task (QA, summary, data-to-text) | RAGTruth | secondary |
| Span F1 by size of the answering model (up to 2B, over 4B; size unknown in an appendix) | PsiloQA | secondary (needed for RQ2) |
| Answer-level F1: an answer counts as flagged when at least one span is marked; F1 of the flagged answers. Published papers such as TinyLettuce report only this | RAGTruth | secondary |
| Share of the teacher: our 3-run mean span F1 / the teacher's span F1 × 100 | RAGTruth, PsiloQA all | for S3 |
| 8-bit cost: span F1 of the 8-bit file minus that of the full-precision file, same answers | all 2,700 RAGTruth answers, PsiloQA all | for S5 |
| Size: parameters; file size in MB of the full-precision and the 8-bit ONNX file | — | for S5 |
| Time: per answer, one answer at a time (as an app works), ONNX Runtime, 1 and 2 threads, after 3 warm-up answers; median and 90th percentile over 300 fixed test answers; splitting the text into tokens is done before timing; the machine (CPU or phone model, RAM, Android version) is named | laptop CPU; phone if possible | for S5 |
| Memory: what loading and running the checker adds to the process (MB) | same | for S5 |
| Agreement of the two annotators: word-level Cohen's κ (each word marked or not), and span F1 of one against the other | pilot | for S6 |
| A chat model as the checker: its span F1 against people; a reply the program cannot read counts as marking nothing | 300 fixed RAGTruth test answers; pilot | baseline for S6 |

## 5. Training runs and uncertainty

**Three training runs.** Each main student of ours is trained three times, with seeds 42, 2 and 3. Runs are named
`<run>`, `<run>-s2` and `<run>-s3`. Only the seed changes (the start of the new output layer, the order of the
training answers, dropout); the data, the recipe and the chosen new answers stay the same. `run.py train` checks
this and refuses a seed run whose recipe differs from the first run's.

- Three runs: `hard17m-8ep`, `kd17m`, `hard68m-8ep`, `kd68m` (step 6); `kd17m-n4509`, `kdlarge17m` (step 7);
  `kdeq*` and `kdrich17m` (steps 10 and 11).
- One run, always labelled "one run": the 68M pair of test A (step 5), the test B students, `kdrich68m`.

**Spread.** For each checker: the mean over its runs ± the standard deviation (n − 1), and each run's number.

**95% intervals.** 1,000 bootstrap resamples (random seed 0). What is resampled is the unit whose answers belong
together: RAGTruth sources (each source has six answers), PsiloQA Wikipedia articles, pilot questions. Every
checker is scored on the same resamples. For a mean over runs, the mean is taken inside each resample. For a
difference, runs are matched by seed (seed 42 with seed 42, and so on); a published checker is one fixed run and
is compared with each of our runs. These intervals show how much the choice of test answers moves a number; how
much training moves it is shown by the three runs. Command: `python run.py summary` writes `results/summary.md`.

**When a difference counts (rule R3).** Only when all three hold:
1. at least three matched runs;
2. every run points the same way (all differences above zero, or all below);
3. the 95% interval of the mean difference does not include zero.

Otherwise we write "no clear difference", never "a trend". A one-run result is reported with its interval and
called a single run.

**Many comparisons.** The success criteria below name every comparison we test (about 17). Each tests a
separate claim, so we do not correct for multiple comparisons, but we report all of them, met or not.
Anything else we look at is labelled exploratory.

## 6. Success criteria and exactly how we check them

Copied from PLAN.md, with the test for each. "Counts" means rule R3 in section 5.

| | Claim | We say it is met when |
|---|---|---|
| S1 | The teacher and small-model answers help | At 17M and at 68M: `kd − hard-8ep` on RAGTruth span F1 at 0.5 counts and its 3-run mean is ≥ +2.0 points; and on PsiloQA all it counts with `kd` better |
| S2 | The best checker of its size | `kd68m − tiny68m` counts with `kd68m` better on RAGTruth, on PsiloQA all and on the pilot set |
| S3 | Close to the 6× larger teacher | The share of the teacher for `kd68m` (3-run mean) is ≥ 95% on RAGTruth and ≥ 100% on PsiloQA all |
| S4 | Small models' answers are the better source | On PsiloQA all, `kd17m-n4509 − kdlarge17m` counts with the small-model side better; the 68M run (step 5) points the same way; and step 10 or step 11 explains why (step 10: with the same share of answers with errors, the gap is no longer clear; step 11: the error-rich selection beats `kd17m-n4509` by ≥ 1 point on PsiloQA in all three runs) |
| S5 | Fits on a phone | For `kd68m` (`kd17m` is reported too): the 8-bit cost is ≥ −1.0 point on RAGTruth; the 8-bit file is ≤ 150 MB; the median time is ≤ 2 s per answer on a mid-range Android phone (if no phone: ≤ 1 s on one laptop CPU core, 1 thread) |
| S6 | Works on the real use | Word-level κ ≥ 0.6 between the two annotators; and on the pilot set, `kd68m − tiny68m` and `kd68m` minus the better self-check model (`selfcheck05` or `selfcheck15`) both count with `kd68m` better |

S1, S2 and S5 are enough for a solid paper; S3 and S6 make it strong; S4 with steps 10 and 11 is the new
finding. A criterion that is met on one test set but not on another is **not met**, and we say so.

## 7. What we report whatever happens

- Every run, including runs that fail or are stopped (rule R4), and every comparison in section 6.
- Numbers at 0.5 next to numbers at the dev cut-off, by task and by answering-model size.
- PsiloQA's marks come from GPT-4o, not people, and its answers were written without the passage; RAGTruth's
  marks are known to miss errors. We say this next to the results.

## 8. What we will not do

- Choose anything (cut-off, epoch, selection, which runs to show) on a test set.
- Change the recipe for some runs only, or add a fourth seed because a result looks bad.
- Report the best seed instead of the mean.
- Drop a test set, a measure or a comparison after seeing the results.
- Train on PsiloQA or on the pilot set, or use a paid service.

## 9. Changes after 28 October 2026

| Date | Change | Reason | Agreed by |
|---|---|---|---|
| — | none yet | | |
