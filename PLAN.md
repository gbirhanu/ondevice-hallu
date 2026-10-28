# Experiment plan: a compact word-level hallucination checker for offline RAG

**Period:** 20 October – 19 November 2026 (one month) · **People:** Gemechu Birhanu, one colleague, Claude (assistant)
**Last updated:** 20 October 2026

## How to use this plan

- The work is a list of **numbered steps**, done roughly in order. Say "do step 6" or "implement step 10" and Claude
  knows what is meant: every step says who does it, what it needs, what to do, what it produces, and when it is done.
- Steps marked **(runs now)** need no new code and can run in the background while other steps go on.
- When a step is finished, tick its box and add one line to the [status log](#status-log) at the end.
- The [rules](#rules) and the [success criteria](#success-criteria-fixed-before-the-new-runs) apply to every step.
  They are fixed now, before the new runs, so we cannot bend them to fit the results later.

## Where we stand (single training runs, 16–19 October 2026)

Span F1 (cut-off 0.5). RAGTruth: 2,700 test answers, people's labels. PsiloQA: 4,000 English answers, GPT-4o labels,
never used for training.

| Checker | Size | RAGTruth | PsiloQA |
|---|---|---|---|
| LettuceDetect-large (our teacher, published) | 396M | 58.92 | 70.22 |
| LettuceDetect-base (published) | 150M | 55.43 | 69.60 |
| TinyLettuce 68M / 17M (published) | 68M / 17M | 50.05 / 44.26 | 69.38 / 66.45 |
| **Ours 68M, distilled** | 68M | **56.02** | **70.64** |
| Ours 68M, people's labels only | 68M | 53.48 | 68.73 |
| **Ours 17M, distilled** | 17M | **44.77** | **62.55** |
| Ours 17M, people's labels only | 17M | 39.25 | 61.15 |

- Distillation helps at both sizes and on both test sets. Our 68M beats TinyLettuce-68M and ties with the teacher on PsiloQA.
- Test A (17M): new answers from small models (0.5–1.7B) teach better than the same number from 7B models
  (PsiloQA +2.5 [1.79, 3.17]); on RAGTruth no clear difference.
- Test B: a free 7B chat model is a poor labeller (11.1 span F1 against people, our teacher 63.0).
- Weak point: at 17M, TinyLettuce-17M is better on PsiloQA (−3.9).
- All of this is one training run each. Making it solid is the first job of this month.

## What we will claim, and what we must not claim

From the literature review of 20 October 2026 (FastTrack connector, full texts of the closest papers).

**We aim to show** (each claim only if its success criterion below is met):
1. The most accurate word-level checker under 100M parameters that we know of, close to a 6× larger teacher.
2. A free recipe: a mid-sized published detector teaches a tiny one, using new answers written by phone-sized models.
3. Answers from phone-sized models teach better than answers from 7B models, and why.
4. The checker still works in 8-bit, and we measure its size, memory and speed.
5. A small test set of phone-sized models' answers over course notes and health guidelines, marked by people.

**We must not claim:**
- that we are the first to distil a hallucination checker into an encoder (Enoki 2026 trains ~400M encoders on a
  120B model's labels);
- that we are the first to train on real answers rather than made-up errors (PsiloQA 2025 and HDM-2 2025 do);
- that we are the first to run a hallucination checker on edge hardware (HALL-OPT 2026 does, for whole outputs);
- that we beat the teacher on RAGTruth (we do not).

## Rules

- **R1.** From now on every choice (cut-off, epochs, selection rules, anything) is made on the **dev set**. The test sets
  give final numbers only.
- **R2.** The training recipe is frozen: Ettin encoder, AdamW, learning rate 2e-5, 5 epochs (8 for the people's-labels
  runs), α = 0.5, max length 4,096, batch 8,192 tokens, best epoch by dev span F1. A change needs a written reason in
  the status log before it is run.
- **R3.** Every main number is the **mean of 3 training runs** (seeds 42, 2 and 3) with its spread. A difference counts
  only if all three runs point the same way **and** the 95% interval of the 3-run average stays above zero.
- **R4.** Every run is reported, including the ones that fail.
- **R5.** Free resources only: Colab and Kaggle free GPUs, open models, no paid APIs.
- **R6.** Nothing is ever trained on PsiloQA or on our pilot test set. They are test sets only.
- **R7.** Data licences are checked before use: PsiloQA is CC-BY-4.0; Mu-SHROOM is CC BY-NC-SA (not used); the pilot
  documents must allow sharing.

## Success criteria (fixed before the new runs)

| | Claim | Success if | Now (1 run) |
|---|---|---|---|
| S1 | Teacher + small-model answers help | ≥ +2 span F1 over people's labels only on RAGTruth, at 17M and 68M, and above zero on PsiloQA | +5.5 / +2.5; +1.4 / +1.9 |
| S2 | Best checker of its size | Our 68M beats TinyLettuce-68M on all three test sets | +6.0, +1.3, pilot not built |
| S3 | Close to the 6× larger teacher | Our 68M reaches ≥ 95% of the teacher's span F1 on RAGTruth and ≥ 100% on PsiloQA | 95.1%, 100.6% (borderline) |
| S4 | Small-model answers are the better source | Small ≥ 7B on PsiloQA in all 3 runs at 17M and in the 68M run, and step 10 or 11 explains why | +2.5 at 17M |
| S5 | Fits on a phone | 8-bit loses ≤ 1 point; file ≤ 150 MB; ≤ 2 s per answer on a mid-range phone (fallback: ≤ 1 s on one laptop CPU core) | not measured |
| S6 | Works on the real target | Annotators agree (word-level κ ≥ 0.6); our 68M beats TinyLettuce-68M and the self-check baseline | not built |

S1, S2 and S5 are enough for a solid paper. S3 and S6 make it strong. S4 with steps 10–11 is the new finding.

## Calendar

| Week | Dates | Steps |
|---|---|---|
| 1 | 20–26 Oct | 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| 2 | 27 Oct – 2 Nov | 10, 11, 12, 13, 14, (15), 16, 17, 18 |
| 3 | 3–9 Nov | 19, 20, start 21 and 22 |
| 4 | 10–19 Nov | 21, 22, 23; the last days are kept free for re-runs |

GPU time in total: about 16–18 hours. Use Colab and Kaggle's free weekly GPU hours side by side.

---

# The steps

## Phase A — Read, then fix the rules (week 1)

### Step 1 — Read the most important papers
- [ ] **Who:** Gemechu (and the colleague). **Needs:** nothing. **Time:** about 4 hours for 1a, 1.5 hours for 1b.
- **Do:** read 1a now, in this order. Skim 1b to know the closest competitors. Read 1c when you reach the step named.
- **Done when:** you can answer the five questions at the end of this step without looking.

**1a. Must read first**

| # | Paper | Time | Why it matters / what to look for |
|---|---|---|---|
| 1 | Niu et al. 2024, *RAGTruth* (ACL) — [PDF](https://aclanthology.org/2024.acl-long.585.pdf) | 60 min | Our main benchmark. How people marked unsupported words; the four error types; which answering models hallucinate most. |
| 2 | Kovács & Recski 2025, *LettuceDetect* — [arXiv](https://arxiv.org/abs/2502.17125) | 40 min | Our teacher. How a token classifier reads passages + answer, how token probabilities become spans, how span F1 is computed. |
| 3 | Kovács & Osmanaj 2025, *TinyLettuce* (blog) — [link](https://huggingface.co/blog/adaamko/tinylettuce) | 20 min | The baseline of our size (17–68M Ettin). Note: 3,000 made-up errors from GPT-OSS-120B; only whole-answer F1; no CPU numbers. |
| 4 | Hinton et al. 2015, *Distilling the Knowledge in a Neural Network* — [arXiv](https://arxiv.org/abs/1503.02531) | 30 min | Sections 1–3. What a soft target is and why a teacher's probabilities teach more than 0/1 labels. |
| 5 | He et al. 2022, *Generate, Annotate, and Learn* (TACL) — [DOI](https://doi.org/10.1162/tacl_a_00492) | 45 min | Our recipe in general NLP: write new text, let the teacher label it, train the student. |
| 6 | Rykov et al. 2025, *PsiloQA* (Findings of EMNLP) — [arXiv](https://arxiv.org/abs/2510.04849) | 45 min | Closest earlier work and our second test set. Real answers from 24 models, marked by GPT-4o (~535 USD), written without documents. |

**1b. The closest competitors (skim, so we do not over-claim)**

| # | Paper | Time | What it does that is close to us |
|---|---|---|---|
| 7 | Rykov et al. 2026, *Enoki* — [arXiv](https://arxiv.org/abs/2609.00581) | 30 min | A 120B model's labels trained into ModernBERT-large encoders (~400M); span scores on RAGTruth and PsiloQA. |
| 8 | Paudel et al. 2025, *HalluciNot (HDM-2)* — [arXiv](https://arxiv.org/abs/2504.07069) | 20 min | Answers from several models labelled by DeepSeek-R1; 3B checker with word scores. |
| 9 | Yang et al. 2026, *Hallucination Self-Play* — [arXiv](https://arxiv.org/abs/2607.07993) | 20 min | A generator makes training data for a 7–8B checker; never compares generator sizes. |
| 10 | Rathmayr, Kovács & Recski 2026, *Grounding Probes* — [arXiv](https://arxiv.org/abs/2610.04642) | 15 min | Whole-answer probe; leaving one answering model out of training costs only ~0.02 AUROC. Compare with our test A. |
| 11 | Algawiaz 2026, *HALL-OPT* (Scientific Reports) — [DOI](https://doi.org/10.1038/s41598-026-42981-3) | 10 min | Hallucination detection with distillation on edge hardware, whole outputs, SQuAD/CNN-DM. Know why it differs. |

**1c. Read when you reach the step**

| # | Paper | For step | Why |
|---|---|---|---|
| 12 | Weller et al. 2025, *Ettin (Seq vs Seq)* — [arXiv](https://arxiv.org/abs/2507.11412) | 4–7 | The encoders our students start from. |
| 13 | Hron et al. 2024, *Training LMs on the Knowledge Graph* — [arXiv](https://arxiv.org/abs/2408.07852) | 9–11 | Larger models hallucinate less, and their hallucinations are harder to detect. |
| 14 | Islam et al. 2025, *How Much Do LLMs Hallucinate across Languages?* (EMNLP) — [PDF](https://aclanthology.org/2025.emnlp-main.1481.pdf) | 9–11 | Smaller LLMs hallucinate more. |
| 15 | Bondarenko et al. 2021, *Efficient Transformer Quantization* (EMNLP) — [DOI](https://doi.org/10.18653/v1/2021.emnlp-main.627) | 13 | Why plain 8-bit breaks these models (outlier values). |
| 16 | Ren 2026, *MAM-AI* — [arXiv](https://arxiv.org/abs/2606.29580) | 15–16 | Offline RAG on an Android phone for midwives in Zanzibar; the setting we serve. |
| 17 | Yu et al. 2026, *Retromorphic Testing (RT4CHART)* — [arXiv](https://arxiv.org/abs/2603.27752) | 18–19 | RAGTruth's labels miss many errors (1.68× more after re-marking): mark carefully. |
| 18 | Sanh et al. 2019, *DistilBERT* — [arXiv](https://arxiv.org/abs/1910.01108) | 22 | How distillation results are usually reported ("97% of the teacher at 60% of the size"). |

All 17 papers with a DOI in one click: [import into Zotero](https://literature.researchfasttrack.com/zotero?ids=10.18653/v1/2024.acl-long.585,10.48550/arxiv.2502.17125,10.48550/arxiv.1503.02531,10.1162/tacl_a_00492,10.48550/arxiv.2510.04849,10.48550/arxiv.2609.00581,10.48550/arxiv.2504.07069,10.48550/arxiv.2607.07993,10.48550/arxiv.2610.04642,10.1038/s41598-026-42981-3,10.48550/arxiv.2507.11412,10.48550/arxiv.2408.07852,10.18653/v1/2025.emnlp-main.1481,10.18653/v1/2021.emnlp-main.627,10.48550/arxiv.2606.29580,10.48550/arxiv.2603.27752,10.48550/arxiv.1910.01108)
(TinyLettuce is a blog post and has no DOI.)

**Check yourself**
1. What are RAGTruth's four error types, and which answering models made the most errors?
2. How does LettuceDetect turn word probabilities into marked spans, and how is span F1 computed?
3. What does TinyLettuce report, and what does it not report?
4. Why can a teacher's probability (say 0.7) teach a student more than a plain 0/1 label?
5. In one sentence each: what do PsiloQA and Enoki do that is close to us, and what is different?

### Step 2 — Fix the evaluation in writing
- [ ] **Who:** Claude drafts; Gemechu and the colleague agree. **Needs:** step 1a read. **Time:** 1 hour.
- **Do:** Claude writes `EVALUATION.md` with the rules, the success criteria, the three test sets, the measures
  (span F1 at 0.5 and at the dev cut-off; answer-level F1 for comparison with published numbers; size, memory and time)
  and how intervals are computed. Commit it with the date **before** steps 6 and 7 start.
- **Output:** `EVALUATION.md` in the repository. **Done when:** committed and pushed.

### Step 3 — Choose the target venue
- [ ] **Who:** Gemechu and the colleague (Claude can look up current deadlines). **Needs:** nothing. **Time:** 30 min.
- **Do:** pick one venue (an African NLP workshop, an ACL-family venue, or a journal). Write its deadline and page limit here.
- **Output:** one line in the status log. **Done when:** the deadline and page limit are known.

## Phase B — Make the main result solid (week 1)

### Step 4 — Code for repeated runs and averaged tables
- [ ] **Who:** Claude. **Needs:** nothing. **Time:** half a day.
- **Do:**
  - Run names with a seed suffix: `<run>-s2`, `<run>-s3` (seed 42 keeps the plain name). `run.py train --seed` already exists.
  - A new `summary` command: for groups of runs, the mean and spread over seeds on RAGTruth (0.5 and dev cut-off) and
    PsiloQA (all, ≤2B, >4B), and paired differences whose 95% interval resamples test sources across all seeds.
    It writes `results/summary.md`.
  - A new notebook `colab/plan.ipynb` with one cell per step that needs the GPU (steps 5, 6, 7, 9–13, 17, 20).
  - Tests for the averaging.
- **Output:** pushed code and notebook. **Done when:** the tests pass and Gemechu has pushed.

### Step 5 — Test A at 68M (runs now)
- [ ] **Who:** Gemechu on Colab. **Needs:** nothing (the cell is ready). **Time:** about 2.5 GPU hours.
- **Do:** open `colab/plan.ipynb`
  ([Colab](https://colab.research.google.com/github/gbirhanu/ondevice-hallu/blob/main/colab/plan.ipynb)), run the two
  setup cells, then the step 5 cell. It trains `kd68m-n4509` and `kdlarge68m`, scores them on RAGTruth and PsiloQA and
  writes `results/compare_step5.md` (17M and 68M rows side by side).
- **Output:** `kd68m-n4509`, `kdlarge68m` and their scores on Drive. **Done when:** `compare_step5.md` has the 68M row.

### Step 6 — Two more seeds of the main runs
- [ ] **Who:** Gemechu. **Needs:** steps 2 and 4. **Time:** about 6 GPU hours (68M runs take about 75 minutes each).
- **Do:** run the step 6 cell: `hard17m-8ep`, `kd17m`, `hard68m-8ep` and `kd68m`, each with seeds 2 and 3 (8 runs).
- **Output:** 8 new runs scored on RAGTruth and PsiloQA. **Done when:** all 8 appear in `summary.md`.

### Step 7 — Two more seeds of test A at 17M
- [ ] **Who:** Gemechu. **Needs:** steps 2 and 4. **Time:** about 1 GPU hour.
- **Do:** run the step 7 cell: `kd17m-n4509` and `kdlarge17m`, each with seeds 2 and 3.
- **Output:** 4 new runs. **Done when:** they appear in `summary.md`.

### Step 8 — Read the averaged results and check S1–S4
- [ ] **Who:** Claude. **Needs:** steps 5–7. **Time:** 1 hour.
- **Do:** read `summary.md`; check S1, S2 (two of three test sets), S3 and the test A part of S4 against the rules;
  write the result in the status log; update the concept note's numbers.
- **Decision:** if S1 fails at 68M, report 17M and the PsiloQA gains, and consider more new answers (after the month).
- **Done when:** the status log says which criteria are met.

## Phase C — Why do small models' answers teach better? (weeks 1–2)

### Step 9 — Count the errors in each set of answers
- [ ] **Who:** Claude writes it; Gemechu runs one short cell (no GPU). **Needs:** step 4. **Time:** 15 minutes.
- **Do:** for each set of new answers (Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B, Qwen2.5-7B, Mistral-7B) report the
  share of answers with at least one word the teacher marks, the share of marked words, and the average answer length.
  Do the same for RAGTruth's six answering models using people's marks.
- **Output:** `results/density.md`. **Done when:** the table exists.
- **Decision:** if small models' answers clearly hold more marked words, steps 10 and 11 test whether that is the reason.

### Step 10 — Same number of errors, different answer writers
- [ ] **Who:** Claude writes the selection; Gemechu runs. **Needs:** step 9. **Time:** about 1.5 GPU hours.
- **Do:** build a small-model set and a 7B set with the same number of answers (4,509) **and** the same share of
  answers with marked errors (chosen from the counts in step 9). Train a 17M student on each, 3 seeds each.
- **Output:** runs `kdeqsmall17m(-s2,-s3)` and `kdeqlarge17m(-s2,-s3)`. **Done when:** in `summary.md`.
- **Reading the result:** if the gap disappears, "more mistakes to learn from" is the reason. If it stays, small
  models make a different kind of mistake; then compare by RAGTruth error type.

### Step 11 — Pick training answers on purpose (mistake-rich sets)
- [ ] **Who:** Claude writes the selection; Gemechu runs. **Needs:** step 9. **Time:** 1.5 GPU hours (+1.2 at 68M).
- **Do:** from the small-model answers, pick 4,509 so that about half contain a teacher-marked error. Compare with
  4,509 taken as they come (`kd17m-n4509`). 17M, 3 seeds. If it wins by ≥ 1 point on PsiloQA in all three runs,
  repeat once at 68M. If the pool is too small, first write one more round of small-model answers (about 2 hours).
- **Output:** runs `kdrich17m(-s2,-s3)` (and `kdrich68m`). **Done when:** in `summary.md`.
- **Why it matters:** if it wins, it is a new method, not only an observation.

### Step 12 — Baseline: the phone's own model checks itself
- [ ] **Who:** Gemechu runs (code exists: `llmlabel --local`). **Needs:** nothing (runs now). **Time:** about 1 GPU hour.
- **Do:** run the step 12 cell in `plan.ipynb`. It asks Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct to list the
  unsupported words in answers: first on 300 RAGTruth test answers (the same ones the 7B labeller had), later on the
  pilot set (step 20). Our teacher and students are scored on the same 300 answers; an unreadable reply counts as no marks.
- **Output:** `results/llmlabel_selfcheck05_test.json`, `results/llmlabel_selfcheck15_test.json` and a table in the
  notebook. **Done when:** both models are scored.

## Phase D — Size and speed (week 2)

### Step 13 — 8-bit files and their accuracy
- [ ] **Who:** Gemechu runs; Claude adds PsiloQA scoring for 8-bit files. **Needs:** step 8. **Time:** about 1 hour.
- **Do:** `export` and `check8bit --n 2700` for `kd17m` and `kd68m` (full RAGTruth test set), then the same on PsiloQA.
- **Output:** `results/kd*_int8.json` with the change in span F1 and its interval; file sizes.
- **Done when:** the change is known for both models (S5: ≤ 1 point).

### Step 14 — Time and memory on a laptop CPU
- [ ] **Who:** Gemechu (laptop), or Colab's CPU as a fallback. **Needs:** step 13. **Time:** 30 minutes.
- **Do:** `bench --name kd68m --where laptop --threads 1 2` (and `kd17m`); write down the CPU model.
- **Output:** median and 90th-percentile time per answer, memory added, file size. **Done when:** in the status log.

### Step 15 — Time and memory on an Android phone (optional)
- [ ] **Who:** Gemechu with a borrowed mid-range phone; Claude writes the instructions. **Needs:** step 13. **Time:** half a day.
- **Do:** run the 8-bit file with ONNX Runtime on the phone; time 100 answers; note the phone model, RAM and Android version.
- **Done when:** numbers for one phone exist, or we decide to leave it until after the month.

## Phase E — A small test set from the real use (weeks 2–3)

### Step 16 — Documents and questions
- [ ] **Who:** Gemechu and the colleague; Claude helps write questions. **Needs:** nothing. **Time:** 1 day.
- **Do:** pick documents whose licence allows sharing: your own lecture notes (for example Computer Architecture),
  since you hold the rights, or open textbooks; plus public health guidelines with an open licence. Write about 50
  questions (25 course, 25 health) that the documents can answer.
- **Output:** `pilot/documents/` and `pilot/questions.jsonl`, with each document's source and licence.
- **Done when:** 50 questions, each with its passage.

### Step 17 — Answers from phone-sized models
- [ ] **Who:** Claude adapts `generate`; Gemechu runs. **Needs:** step 16. **Time:** 30 GPU minutes.
- **Do:** Qwen2.5-0.5B, Qwen2.5-1.5B and SmolLM2-1.7B answer every question from its passage (same prompt style as
  RAGTruth's questions). About 150 answers.
- **Output:** `pilot/answers.jsonl`. **Done when:** every question has three answers.

### Step 18 — Annotation guide, tool and practice round
- [ ] **Who:** Claude drafts the guide and builds a simple marking page; both annotators practise. **Needs:** step 17. **Time:** half a day.
- **Do:** the guide says what counts as unsupported and how to mark the shortest span, with RAGTruth's four error
  types. Both people mark the same 10 answers, compare, and fix the guide.
- **Output:** `pilot/GUIDE.md` and the marking page. **Done when:** the practice round is discussed and the guide revised.

### Step 19 — Mark the answers and measure agreement
- [ ] **Who:** Gemechu and the colleague, separately (about one day each); Claude computes agreement. **Needs:** step 18.
- **Do:** each person marks all ~150 answers alone. Claude computes word-level Cohen's κ and span F1 between the two.
  If κ < 0.6, discuss the disagreements and revise. Then agree one final marking per answer.
- **Output:** `pilot/labels_a.jsonl`, `pilot/labels_b.jsonl`, `pilot/gold.jsonl`, `results/pilot_agreement.md`.
- **Done when:** the gold labels exist and the agreement is written down.

### Step 20 — Score every checker on the pilot set
- [ ] **Who:** Claude writes the cell; Gemechu runs. **Needs:** step 19. **Time:** 10 GPU minutes.
- **Do:** score the teacher, LettuceDetect-base, TinyLettuce 17M/68M, our 17M/68M (all seeds) and the self-check
  baseline from step 12.
- **Output:** `results/pilot.md`. **Done when:** S6 can be checked.

## Phase F — Write and release (weeks 3–4)

### Step 21 — Final tables, figures and the claims we can make
- [ ] **Who:** Claude. **Needs:** steps 8, 10–14, 20. **Time:** 1 day.
- **Do:** final tables (3-run means) and figures; check S1–S6 one by one; list the claims we can make and the ones we cannot.
- **Output:** `results/final.md` and figures. **Done when:** every criterion is marked met / not met.

### Step 22 — Write the paper
- [ ] **Who:** Claude drafts; Gemechu and the colleague edit. **Needs:** steps 3 and 21. **Time:** about 1 week.
- **Do:** introduction, related work (from the literature review), method (the architecture figure is ready),
  experiments, results, limitations, ethics (data licences, people's marking).
- **Done when:** a full draft within the venue's page limit.

### Step 23 — Release
- [ ] **Who:** Gemechu uploads; Claude prepares the files. **Needs:** step 22. **Time:** half a day.
- **Do:** code on GitHub; our 17M and 68M models on Hugging Face (16-bit and 8-bit, with model cards); the pilot test
  set with its document licences; the preprint on arXiv.
- **Done when:** all links work.

### Step 24 — After the month (only if the results hold)
More new answers and each small model alone; made-up errors added at 17M; a second teacher; a larger test set
(300–500 answers); full phone measurements; Amharic and Afaan Oromo.

---

# Reference for Claude

- **Repository:** github.com/gbirhanu/ondevice-hallu. Gemechu's copy: `C:\Users\cprfa\Documents\newpaper\ondevice_hallu`
  (Windows line endings; Gemechu commits and pushes himself). The old `tse` / `hallu_poc` work is never changed.
- **Drive:** `MyDrive/ondevice_hallu/work/` with `results/`, `preds/`, `models/`, `extra/` (new answers) and `llm/` (LLM replies).
- **Notebooks:** `size_ladder.ipynb` (published detectors, 8-bit, CPU timing), `signals_test.ipynb` (dropped idea),
  `distill.ipynb` (teacher–student), `novelty.ipynb` (tests A and B, PsiloQA), `plan.ipynb` (this plan, from step 4).
- **Run names:** people's labels only `hard17m`, `hard17m-8ep`, `hard68m-8ep`; teacher's marks on RAGTruth only
  `soft17m`; distilled `kd17m`, `kd68m`; test A `kd17m-n4509` vs `kdlarge17m` (and the 68M pair); test B `kdhard17m`
  vs `llm17m`; seeds add `-s2`, `-s3`; steps 10 and 11 use `kdeq*` and `kdrich*`.
- **Models:** teacher `ld_large` = KRLabsOrg/lettucedect-large-modernbert-en-v1; students start from
  jhu-clsp/ettin-encoder-17m / -68m; new answers from Qwen2.5-0.5B/1.5B-Instruct and SmolLM2-1.7B-Instruct; 7B arm
  Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (4-bit).
- **Scoring:** span F1 exactly as LettuceDetect, at 0.5 and at the dev cut-off; 95% intervals resample RAGTruth
  sources or PsiloQA Wikipedia articles; the dev set is 10% of the training sources.
- **Commands:** `run.py` with `train`, `score`, `teach`, `generate`, `llmlabel`, `psilo`, `compare`, `export`,
  `check8bit`, `bench` (and `summary` from step 4). Every command skips work whose result already exists.
- **Copying files to Gemechu's folder:** convert to Windows line endings, write through the device bridge, check the md5.

# Status log

| Date | Step | What happened | Where the result is |
|---|---|---|---|
| 16–17 Oct | — | Published detectors scored, 8-bit fixed (w8 recipe), first idea (generator signals) dropped | `results/table.md`, `fusion*.md` |
| 18 Oct | — | Distillation works at 17M and 68M (1 run each) | `results/distill.md` |
| 19 Oct | — | Tests A and B at 17M; PsiloQA scores; 68M checks | `results/compare.md`, `psilo.md`, `compare_checks.md` |
| 20 Oct | — | Literature review (FastTrack); gap reworded; this plan written | `PLAN.md` |
| | 1 | | |
