# Experiment plan: a compact word-level hallucination checker for offline RAG

**Period:** 20 October – 19 November 2026 (one month) · **People:** Gemechu Birhanu, one colleague, Claude (assistant)
**Last updated:** 28 November 2026

## How to use this plan

- The work is a list of **numbered steps**, done roughly in order. Say "do step 6" or "implement step 10" and Claude
  knows what is meant: every step says who does it, what it needs, what to do, what it produces, and when it is done.
- Steps marked **(runs now)** need no new code and can run in the background while other steps go on.
- When a step is finished, tick its box and add one line to the [status log](#status-log) at the end.
- The [rules](#rules) and the [success criteria](#success-criteria-fixed-before-the-new-runs) apply to every step.
  They are fixed now, before the new runs, so we cannot bend them to fit the results later.

## Where we stand (3 training runs each, 31 October 2026)

Span F1 (cut-off 0.5), mean ± spread over the training runs. RAGTruth: 2,700 test answers, people's labels.
PsiloQA: 4,000 English answers, GPT-4o labels, never used for training. All numbers: `results/summary.md`.

| Checker | Size | RAGTruth | PsiloQA |
|---|---|---|---|
| LettuceDetect-large (our teacher, published) | 396M | 58.92 | 70.22 |
| LettuceDetect-base (published) | 150M | 55.43 | 69.60 |
| TinyLettuce 68M / 17M (published) | 68M / 17M | 50.05 / 44.26 | 69.38 / 66.45 |
| **Ours 68M, distilled** | 68M | **56.22 ± 0.42** | **70.37 ± 0.24** |
| Ours 68M, people's labels only | 68M | 54.19 ± 1.49 | 69.43 ± 0.91 |
| **Ours 17M, distilled** | 17M | **45.64 ± 3.92** | **63.66 ± 2.45** |
| Ours 17M, people's labels only | 17M | 41.26 ± 3.78 | 61.79 ± 3.01 |
| Chat models asked to mark the errors: Qwen2.5-0.5B / 1.5B / 7B (300 test answers) | 0.5–7B | 5.4 / 3.5 / 11.1 | — |

- Distillation helps at 17M on both test sets (+4.38 and +1.87, all runs agree) and at 68M on RAGTruth (+2.03,
  all runs agree). At 68M on PsiloQA it is +0.94 [0.64, 1.24] on average, but one of the three runs was −0.19,
  so by rule R3 it does not count.
- Our 68M beats TinyLettuce-68M in all three runs on both test sets (+6.17 and +0.99), reaches 95.4% of the
  6× larger teacher on RAGTruth and ties with it on PsiloQA (+0.15, no clear difference).
- Test A: who writes the new answers makes no clear difference. At 17M, small-model answers give +0.97 on
  PsiloQA on average, but one of three runs went the other way; at 68M (one run) +0.29. The earlier +2.5 was one run.
- Small models' answers hold more errors: 52–61% of them have an error the teacher marks (8–12% of their
  characters), against 36–39% (3–4%) for the 7B models' answers. Even so, they did not teach clearly better.
- Test B: a free 7B chat model is a poor labeller (11.1 span F1 against people, our teacher 63.0).
- Phone-sized chat models cannot check answers: 3.5–5.4 span F1, against 59.7 for our 68M checker on the same
  300 answers. A separate small checker is needed.
- In 8-bit, our 68M is a 73.9 MB file (261 MB in full precision) and loses nothing: −0.03 [−0.16, 0.09] on
  RAGTruth, 0.00 on PsiloQA. The 17M file is 18.3 MB and loses −0.22 on RAGTruth (not clear) and −0.93 on PsiloQA.
- On one laptop core (Intel i7-1185G7), the 8-bit 68M needs 1.66 s per answer (median; 1.03 s with 2 threads),
  over the 1 s target; the 17M needs 0.15 s. Our 8-bit file is not faster than full precision (1.20 s): it keeps
  the arithmetic in float, so it saves size and memory, not time.
- The `mix` 8-bit recipe (8-bit arithmetic, tried for speed on 2 November) fails: 56.02 → 34.59 on RAGTruth
  (−21.43 [−24.52, −18.21]). That line is closed. Speed must come from a smaller model or shorter inputs.
- Of our 68M's +6.17 over TinyLettuce-68M, +4.14 comes from our training with people's labels alone
  (`hard68m-8ep` 54.19 against 50.05) and +2.03 from the teacher and the new answers.
- On PsiloQA every checker of 68M or more, the 396M teacher included, lies between 69.4 and 70.6, so that test set
  has little room left to show a gain at 68M.
- With learning rate 2e-5 the 17M students were still improving on the dev set at their last epoch (the best epoch
  of `kd17m` was the last one in all three runs). Steps 8b and 8c (added 13 November) test other learning rates on
  the dev set and add a 32M student.
- **Added runs (steps 8b–8e, 24–25 November; exploratory, all in `results/summary_added.md`):**
  - On the dev set, learning rate 1e-4 was chosen for both 17M students and for `kd32m`; `hard32m-8ep` keeps 2e-5.
  - 17M with 1e-4: `kd17m-lr1e4` 49.71 ± 1.73 on RAGTruth (2e-5: 45.64 ± 3.92) and 63.69 ± 1.10 on PsiloQA (no
    change). The teacher still adds +4.85 and +2.79 (all runs agree). It ties TinyLettuce-68M on RAGTruth (−0.34)
    but stays below TinyLettuce-17M on PsiloQA (−2.76, all runs).
  - 32M: `kd32m-lr1e4` 52.30 ± 1.60 / 67.32 ± 1.26; `hard32m-8ep` 51.12 ± 0.94 / 68.99 ± 0.47. It beats
    TinyLettuce-32M (+4.44) and TinyLettuce-68M (+2.25) on RAGTruth, but is below both on PsiloQA (−1.77, −2.06).
    At 32M the teacher adds +1.18 on RAGTruth (one run −0.27, so it does not count; +2.54 with the dev cut-off) and
    costs −1.67 on PsiloQA (−2.32 on answers by models up to 2B, which counts). The two 32M students use different
    learning rates. `kd68m` is 3.92 / 3.05 points above `kd32m-lr1e4`.
  - The higher learning rate raises RAGTruth but not PsiloQA, at both sizes; it was chosen on RAGTruth dev answers only.
  - `kd68m` beats the published 150M `ld_base` on PsiloQA (+0.77 [0.30, 1.29], all runs); on RAGTruth all three runs
    are higher (+0.51 to +1.28), but the interval includes 0.
  - 8-bit 32M: 122.4 → 34.6 MB with no loss (RAGTruth +0.10 [−0.06, 0.29], PsiloQA +0.08). Laptop, one core: median
    0.47 s per answer (90th percentile 1.15 s), 0.36 s on 2 threads. It meets the 1 s target; the 68M (1.66 s) does not.
  - 8-bit 17M with 1e-4: 18.3 MB, RAGTruth +0.09, PsiloQA −0.04 (the 2e-5 run lost 0.93 on PsiloQA).
- Weak points: at 17M, TinyLettuce-17M is better on PsiloQA (−2.79; on answers by models up to 2B −3.79, in all
  runs); and 17M training varies a lot between runs (± 3.8 points on RAGTruth, against ± 0.4 at 68M).

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

| | Claim | Success if | Now (3 runs, 31 Oct) |
|---|---|---|---|
| S1 | Teacher + small-model answers help | ≥ +2 span F1 over people's labels only on RAGTruth, at 17M and 68M, and above zero on PsiloQA | **not met**: 17M met (+4.38; +1.87); 68M RAGTruth +2.03 met, PsiloQA +0.94 but one run −0.19 |
| S2 | Best checker of its size | Our 68M beats TinyLettuce-68M on all three test sets | +6.17 and +0.99, all runs (met on 2 of 3); pilot not built |
| S3 | Close to the 6× larger teacher | Our 68M reaches ≥ 95% of the teacher's span F1 on RAGTruth and ≥ 100% on PsiloQA | **met**: 95.4%, 100.2% |
| S4 | Small-model answers are the better source | Small ≥ 7B on PsiloQA in all 3 runs at 17M and in the 68M run, and step 10 or 11 explains why | **not met**: 17M +0.97, one run −0.56; 68M +0.29 (one run) |
| S5 | Fits on a phone | 8-bit loses ≤ 1 point; file ≤ 150 MB; ≤ 2 s per answer on a mid-range phone (fallback: ≤ 1 s on one laptop CPU core) | **not met on time**: 8-bit loses 0.03 / 0.00 and the file is 73.9 MB, but one laptop core needs 1.66 s per answer (17M: 0.15 s). Added later (exploratory): the 32M meets it, 8-bit loses nothing, 34.6 MB, 0.47 s |
| S6 | Works on the real target | Annotators agree (word-level κ ≥ 0.6); our 68M beats TinyLettuce-68M and the self-check baseline | self-check baseline measured (3.5–5.4); pilot not built |

S1, S2 and S5 are enough for a solid paper. S3 and S6 make it strong. S4 with steps 10–11 is the new finding.

## Calendar

| Week | Dates | Steps |
|---|---|---|
| 1 | 20–26 Oct | 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| 2 | 27 Oct – 2 Nov | 10, 11, 12, 13, 14, (15), 16, 17, 18 |
| 3 | 3–9 Nov | 19, 20, start 21 and 22 |
| 4 | 10–19 Nov | 21, 22, 23; the last days are kept free for re-runs |
| added | 13–19 Nov | 8b, 8c, 8d, 8e (about 9 GPU hours and 1 CPU hour); 16–20 in parallel, since they need people, not the GPU |

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
- [x] **Who:** Claude drafts; Gemechu and the colleague agree. **Needs:** step 1a read. **Time:** 1 hour.
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
- [x] **Who:** Claude. **Needs:** nothing. **Time:** half a day.
- **Do:**
  - Run names with a seed suffix: `<run>-s2`, `<run>-s3` (seed 42 keeps the plain name). `run.py train --seed` already exists.
  - A new `summary` command: for groups of runs, the mean and spread over seeds on RAGTruth (0.5 and dev cut-off) and
    PsiloQA (all, ≤2B, >4B), and paired differences whose 95% interval resamples test sources across all seeds.
    It writes `results/summary.md`.
  - A new notebook `colab/plan.ipynb` with one cell per step that needs the GPU (steps 5, 6, 7, 9–13, 17, 20).
  - Tests for the averaging.
- **Output:** pushed code and notebook. **Done when:** the tests pass and Gemechu has pushed.

### Step 5 — Test A at 68M (runs now)
- [x] **Who:** Gemechu on Colab. **Needs:** nothing (the cell is ready). **Time:** about 2.5 GPU hours.
- **Do:** open `colab/plan.ipynb`
  ([Colab](https://colab.research.google.com/github/gbirhanu/ondevice-hallu/blob/main/colab/plan.ipynb)), run the two
  setup cells, then the step 5 cell. It trains `kd68m-n4509` and `kdlarge68m`, scores them on RAGTruth and PsiloQA and
  writes `results/compare_step5.md` (17M and 68M rows side by side).
- **Output:** `kd68m-n4509`, `kdlarge68m` and their scores on Drive. **Done when:** `compare_step5.md` has the 68M row.

### Step 6 — Two more seeds of the main runs
- [x] **Who:** Gemechu. **Needs:** steps 2 and 4. **Time:** about 6 GPU hours (68M runs take about 75 minutes each).
- **Do:** run the step 6 cell: `hard17m-8ep`, `kd17m`, `hard68m-8ep` and `kd68m`, each with seeds 2 and 3 (8 runs).
- **Output:** 8 new runs scored on RAGTruth and PsiloQA. **Done when:** all 8 appear in `summary.md`.

### Step 7 — Two more seeds of test A at 17M
- [x] **Who:** Gemechu. **Needs:** steps 2 and 4. **Time:** about 1 GPU hour.
- **Do:** run the step 7 cell: `kd17m-n4509` and `kdlarge17m`, each with seeds 2 and 3.
- **Output:** 4 new runs. **Done when:** they appear in `summary.md`.

### Step 8 — Read the averaged results and check S1–S4
- [x] **Who:** Claude. **Needs:** steps 5–7. **Time:** 1 hour.
- **Do:** read `summary.md`; check S1, S2 (two of three test sets), S3 and the test A part of S4 against the rules;
  write the result in the status log; update the concept note's numbers.
- **Decision:** if S1 fails at 68M, report 17M and the PsiloQA gains, and consider more new answers (after the month).
- **Done when:** the status log says which criteria are met.

## Phase B2 — A better 17M and a 32M student (added 13 November)

These steps were added **after** the results of 31 October were seen (EVALUATION.md, section 9). Their results are
reported as added later; S1–S4 stay judged on the runs with the original recipe. The rule for all of them was
written down before any run: for each student, seed 42 is trained once per learning rate; only the dev span F1 at
0.5 of the best epoch is compared (no test set is scored); the student keeps 2e-5 unless another value is at least
2.0 dev points better; the chosen run is repeated with seeds 2 and 3. `run.py lrpick` applies the rule and saves the
choice in `results/lrpick_<student>.json`.

### Step 8b — The 17M students with other learning rates
- [x] **Who:** Gemechu on Colab (GPU). **Needs:** nothing. **Time:** about 1.7 GPU hours (part 1) + 1.1 (part 2).
- **Why:** with 2e-5 the 17M students were still improving at their last epoch and vary by ± 3.8 points between runs.
- **Do:** in `colab/plan.ipynb`, run the two setup cells, then the cell "Step 8b, part 1": `hard17m-8ep` and `kd17m`,
  seed 42, with learning rates 5e-5, 1e-4 and 2e-4 (6 runs; run names end in `-lr5e5`, `-lr1e4`, `-lr2e4`). It ends
  with two small tables that say which learning rate each student keeps. Then run "Step 8b, part 2": the chosen runs
  with seeds 2 and 3 (nothing is trained if both students keep 2e-5).
- **Output:** `results/lrpick_hard17m-8ep.json`, `lrpick_kd17m.json` (and `.md`); up to 4 new scored runs.
- **Done when:** both choices are saved and the chosen runs have three seeds.

### Step 8c — A 32M student
- [x] **Who:** Gemechu on Colab (GPU). **Needs:** nothing. **Time:** about 3.5 GPU hours (part 1) + 2.5 (part 2);
  an estimate, the first run shows the real time. Each part can be spread over two sessions.
- **Why:** the 68M needs 1.66 s per answer on one laptop core and the 17M is weak; the 32M lies between them.
- **Do:** the cell "Step 8c, part 1": `hard32m-8ep` (8 epochs) and `kd32m` (5 epochs, the teacher, the same 6,719 new
  answers), seed 42, with learning rates 2e-5, 5e-5 and 1e-4 (6 runs), then the choice. Then "Step 8c, part 2": the
  chosen runs with seeds 2 and 3 (4 runs).
- **Output:** `results/lrpick_hard32m-8ep.json`, `lrpick_kd32m.json`; 6 scored runs. About 0.7 GB more on Drive.
- **Done when:** both chosen 32M students have three scored runs.

### Step 8d — Tables for the added runs
- [x] **Who:** Gemechu runs the cell; Claude reads the table. **Needs:** step 8b or 8c. **Time:** 10–20 minutes (GPU).
- **Do:** the cell "Step 8d". It scores the new runs (and the published TinyLettuce-32M) on PsiloQA and writes
  `results/summary_added.md`: what the teacher adds at 17M and 32M, our 17M and 32M against TinyLettuce 17M / 32M / 68M,
  what the new learning rate changed, and `kd68m` against the published 150M `ld_base`. `summary.md` is not touched.
  Run it after 8b and again after 8c. Then tell Claude "step 8d done".
- **Done when:** the status log says what the added runs show.

### Step 8e — 8-bit files, size and speed of the new students
- [x] **Who:** Gemechu (Colab CPU runtime, then the laptop). **Needs:** step 8d. **Time:** about 1 hour + 30 minutes.
- **Do:** on a **CPU** runtime, the cell "Step 8e": the 8-bit file of the chosen `kd32m` run (and of the chosen
  `kd17m` run if its learning rate changed), its accuracy on all RAGTruth test answers and on PsiloQA, and the time
  per answer on Colab's CPU. Then on the laptop, as in step 14: download the folder of the chosen run (its name is in
  the table of step 8c, for example `kd32m` or `kd32m-lr5e5`) from `work/models/` on Drive into `work\models\`, and run
  `python run.py bench --name <run> --model work\models\<run> --threads 1 2 --where laptop --data-dir ..\RAGTruth --work-dir work`.
- **Output:** `results/<run>_export.json`, `<run>_int8_n2700.json`, `<run>_int8_psilo.json`, `<run>_bench_colab.json`;
  on the laptop `work\results\<run>_bench_laptop.json`.
- **Done when:** size, 8-bit cost and laptop time of the 32M are in the status log.

## Phase C — Why do small models' answers teach better? (weeks 1–2)

### Step 9 — Count the errors in each set of answers
- [x] **Who:** Claude writes it; Gemechu runs one short cell (no GPU). **Needs:** step 4. **Time:** 15 minutes.
- **Do:** run the step 9 cell of `plan.ipynb` (`python run.py density`; no GPU, a few minutes). For each set of new
  answers (Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B, Qwen2.5-7B, Mistral-7B) it reports the share of answers with at
  least one error the teacher marks, the share of marked characters, and the average answer length; the same for
  RAGTruth's six answering models, marked by people and by the teacher.
- **Output:** `results/density.md`. **Done when:** the table exists.
- **Decision (1 November):** test A showed no clear difference, so the counts are reported as a description; step 10
  is dropped and step 11 is optional.

### Step 10 — Same number of errors, different answer writers (dropped)
- **Dropped on 1 November** (agreed by Gemechu): test A showed no clear difference between small-model and 7B
  answers (S4 not met), so there is no gap to explain. The original plan is kept below for the record.
- **Who:** Claude writes the selection; Gemechu runs. **Needs:** step 9. **Time:** about 1.5 GPU hours.
- **Do:** build a small-model set and a 7B set with the same number of answers (4,509) **and** the same share of
  answers with marked errors (chosen from the counts in step 9). Train a 17M student on each, 3 seeds each.
- **Output:** runs `kdeqsmall17m(-s2,-s3)` and `kdeqlarge17m(-s2,-s3)`. **Done when:** in `summary.md`.
- **Reading the result:** if the gap disappears, "more mistakes to learn from" is the reason. If it stays, small
  models make a different kind of mistake; then compare by RAGTruth error type.

### Step 11 — Pick training answers on purpose (mistake-rich sets) (optional)
- **Optional since 1 November:** only if GPU time is left at the end. 17M runs vary by ± 3.8 points between seeds,
  so a 1-point gain would be hard to see.
- [ ] **Who:** Claude writes the selection; Gemechu runs. **Needs:** step 9. **Time:** 1.5 GPU hours (+1.2 at 68M).
- **Do:** from the small-model answers, pick 4,509 so that about half contain a teacher-marked error. Compare with
  4,509 taken as they come (`kd17m-n4509`). 17M, 3 seeds. If it wins by ≥ 1 point on PsiloQA in all three runs,
  repeat once at 68M. If the pool is too small, first write one more round of small-model answers (about 2 hours).
- **Output:** runs `kdrich17m(-s2,-s3)` (and `kdrich68m`). **Done when:** in `summary.md`.
- **Why it matters:** if it wins, it is a new method, not only an observation.

### Step 12 — Baseline: the phone's own model checks itself
- [x] **Who:** Gemechu runs (code exists: `llmlabel --local`). **Needs:** nothing (runs now). **Time:** about 1 GPU hour.
- **Do:** run the step 12 cell in `plan.ipynb`. It asks Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct to list the
  unsupported words in answers: first on 300 RAGTruth test answers (the same ones the 7B labeller had), later on the
  pilot set (step 20). Our teacher and students are scored on the same 300 answers; an unreadable reply counts as no marks.
- **Output:** `results/llmlabel_selfcheck05_test.json`, `results/llmlabel_selfcheck15_test.json` and a table in the
  notebook. **Done when:** both models are scored.

## Phase D — Size and speed (week 2)

### Step 13 — 8-bit files and their accuracy
- [x] **Who:** Gemechu runs; Claude adds PsiloQA scoring for 8-bit files. **Needs:** step 8. **Time:** about 1 hour.
- **Do:** run the step 13 cell of `plan.ipynb` on a CPU runtime (no GPU needed, about 2 hours): `export` and
  `check8bit --n 2700 --psilo` for `kd17m` and `kd68m` (all RAGTruth test answers and the PsiloQA answers).
- **Output:** `results/kd*_int8_n2700.json` and `kd*_int8_psilo.json` with the change in span F1 and its interval;
  file sizes in `kd*_export.json`.
- **Done when:** the change is known for both models (S5: ≤ 1 point).

### Step 14 — Time and memory on a laptop CPU
- [x] **Who:** Gemechu (laptop), or Colab's CPU as a fallback. **Needs:** step 13. **Time:** about 1 hour, most of it installing.
- **Do:** on the laptop, in PowerShell, in `C:\Users\cprfa\Documents\newpaper\ondevice_hallu`:
  1. once: `python -m pip install -r requirements.txt`;
  2. download the folders `work/models/kd68m` and `work/models/kd17m` from Drive into `work\models\`;
  3. `python run.py bench --name kd68m --model work\models\kd68m --threads 1 2 --where laptop --data-dir ..\RAGTruth --work-dir work`,
     then the same with `kd17m`. The CPU name, memory and Windows version are recorded with the times.
  Fallback: the step 14 cell of `plan.ipynb` on a Colab CPU runtime.
- **Output:** median and 90th-percentile time per answer, memory added, file size. **Done when:** in the status log.

### Step 14b — A faster 68M without new training (added 1 November, exploratory)
- **13 November:** variant (a), the `mix` recipe, fails (−21.43 points on RAGTruth) and is closed; variant (c), which
  uses it, is dropped. Only the accuracy of variant (b), inputs cut to 1,024 tokens, is still to be measured: the
  step 14b cell of `plan.ipynb` now runs only that (CPU runtime, about 1.5 hours). Its laptop time is known (1.23 s).
- [ ] **Who:** Claude wrote the code; Gemechu runs. **Needs:** step 14. **Time:** about 4 hours on a Colab CPU runtime
  (accuracy) and 1 hour on the laptop (speed).
- **Why:** on one laptop core the 8-bit 68M needs 1.66 s per answer; the target was 1 s.
- **Do:** three variants of the 8-bit `kd68m`: (a) the `mix` recipe, which computes in 8 bits in the layers where
  that is safe; (b) inputs cut to 1,024 tokens (the passages are shortened, the answer is kept); (c) both.
  Accuracy: the step 14b cell of `plan.ipynb` on a Colab CPU runtime. Speed: on the laptop, in the `odh` environment,
  in `C:\Users\cprfa\Documents\newpaper\ondevice_hallu`:
  `python run.py bench --name kd68m --model work\models\kd68m --recipe mix --threads 1 2 --where laptop --data-dir ..\RAGTruth --work-dir work`,
  then the same with `--max-length 1024` instead of `--recipe mix`, then with both.
- **Rule:** these variants were chosen after seeing step 14, so they are reported as exploratory; S5 stays judged on
  the planned 8-bit file.
- **Output:** `results/kd68m_int8_mix_n2700.json`, `kd68m_int8_mix_psilo.json`, `kd68m_int8_len1024_n2700.json`,
  `kd68m_int8_mix_len1024_n2700.json`; on the laptop `work\results\kd68m_bench_laptop_mix.json`, `..._len1024.json`,
  `..._mix_len1024.json`. **Done when:** accuracy and speed are known for all three.

### Step 15 — Time and memory on an Android phone (optional)
- [ ] **Who:** Gemechu with a borrowed mid-range phone; Claude writes the instructions. **Needs:** step 13. **Time:** half a day.
- **Do:** run the 8-bit file with ONNX Runtime on the phone; time 100 answers; note the phone model, RAM and Android version.
- **Done when:** numbers for one phone exist, or we decide to leave it until after the month.

## Phase E — A small test set from the real use (weeks 2–3)

### Step 16 — Documents and questions
- **28 November:** done, with 67 questions instead of about 50, all kept after the review by Gemechu and the
  colleague. Course: 16 passages written in plain words from the topics of the CoSc4035 Computer Security slides
  (the slides contain textbook text, so none is copied; CC BY 4.0), 34 questions. Health: 10 excerpts copied word
  for word from CDC pages (public domain), 33 questions. 51 questions are answered fully by their passage, 8 in part
  and 8 not at all; the last two kinds check whether a model makes up an answer. Details: `pilot/README.md`.
- [x] **Who:** Gemechu and the colleague; Claude helps write questions. **Needs:** nothing. **Time:** 1 day.
- **Do:** pick documents whose licence allows sharing: your own lecture notes (for example Computer Architecture),
  since you hold the rights, or open textbooks; plus public health guidelines with an open licence. Write about 50
  questions (25 course, 25 health) that the documents can answer.
- **Output:** `pilot/documents/` and `pilot/questions.jsonl`, with each document's source and licence.
- **Done when:** 50 questions, each with its passage.

### Step 17 — Answers from phone-sized models
- [ ] **Who:** Claude wrote `run.py pilotgen` and the cell; Gemechu runs. **Needs:** step 16. **Time:** about 20–30 GPU minutes.
- **Do:** Qwen2.5-0.5B, Qwen2.5-1.5B and SmolLM2-1.7B answer every question from its passage, with RAGTruth's
  question-answering prompt (one passage instead of three, and RAGTruth's own sentence asking for "Unable to answer
  based on given passages." when the passage does not answer). Sampling as for the new training answers; an answer
  that stops at 450 tokens or is empty is written again with another seed (up to 3 tries). 201 answers. In
  `colab/plan.ipynb`, on a T4 GPU: the two setup cells, then the cell "Step 17".
- **Output:** `work/pilot/answers.jsonl` on Drive (Claude copies it into `pilot/answers.jsonl` with step 18) and
  `results/pilot_answers.md`. **Done when:** every question has three answers.

### Step 18 — Annotation guide, tool and practice round
- [ ] **Who:** Claude drafts the guide and builds a simple marking page; both annotators practise. **Needs:** step 17. **Time:** half a day.
- **Do:** the guide says what counts as unsupported and how to mark the shortest span, with RAGTruth's four error
  types. Both people mark the same 10 answers, compare, and fix the guide.
- **Output:** `pilot/GUIDE.md` and the marking page. **Done when:** the practice round is discussed and the guide revised.

### Step 19 — Mark the answers and measure agreement
- [ ] **Who:** Gemechu and the colleague, separately (about 1.5 days each); Claude computes agreement. **Needs:** step 18.
- **Do:** each person marks all 201 answers alone. Claude computes word-level Cohen's κ and span F1 between the two.
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
  Added 13 November: another learning rate adds `-lr5e5`, `-lr1e4` or `-lr2e4` before the seed (`kd17m-lr1e4-s2`);
  the 32M students are `hard32m-8ep` and `kd32m`; the chosen runs are in `results/lrpick_<student>.json`.
- **Models:** teacher `ld_large` = KRLabsOrg/lettucedect-large-modernbert-en-v1; students start from
  jhu-clsp/ettin-encoder-17m / -68m; new answers from Qwen2.5-0.5B/1.5B-Instruct and SmolLM2-1.7B-Instruct; 7B arm
  Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (4-bit).
- **Scoring:** span F1 exactly as LettuceDetect, at 0.5 and at the dev cut-off; 95% intervals resample RAGTruth
  sources or PsiloQA Wikipedia articles; the dev set is 10% of the training sources.
- **Commands:** `run.py` with `train`, `score`, `teach`, `generate`, `llmlabel`, `psilo`, `compare`, `export`,
  `check8bit`, `bench`, `summary` (step 4), `density` (step 9) and `lrpick` (steps 8b, 8c). Every command skips work whose result already exists.
- **Copying files to Gemechu's folder:** convert to Windows line endings, write through the device bridge, check the md5.

# Status log

| Date | Step | What happened | Where the result is |
|---|---|---|---|
| 16–17 Oct | — | Published detectors scored, 8-bit fixed (w8 recipe), first idea (generator signals) dropped | `results/table.md`, `fusion*.md` |
| 18 Oct | — | Distillation works at 17M and 68M (1 run each) | `results/distill.md` |
| 19 Oct | — | Tests A and B at 17M; PsiloQA scores; 68M checks | `results/compare_novelty.md` (tests A, B), `compare.md` (68M checks), `psilo.md` |
| 20 Oct | — | Literature review (FastTrack); gap reworded; this plan written | `PLAN.md` |
| | 1 | | |
| 28 Oct | 2 | `EVALUATION.md` drafted by Claude, accepted as written by Gemechu and the colleague, pushed | `EVALUATION.md` |
| 28 Oct | 4 | Seed runs (`<run>-s2`, `-s3`, recipe checked by `train`), `summary` command, `plan.ipynb` with steps 5, 6, 7, 12 and the averaged tables; 39 tests pass | `run.py`, `colab/plan.ipynb` |
| 28 Oct | 5 | Test A at 68M (one run): small-model vs 7B answers, no clear difference (RAGTruth +0.32 [−1.31, 1.79], PsiloQA +0.29 [−0.13, 0.69]; at 17M PsiloQA +2.50). With 4,509 instead of 6,719 new answers the 68M student loses 0.50 [0.18, 0.79] on PsiloQA. Each run 67 minutes | `results/compare_step5.md` |
| 28–30 Oct | 6 | 8 runs: seeds 2 and 3 of `hard17m-8ep`, `kd17m`, `hard68m-8ep`, `kd68m` | `results/summary.md` |
| 30 Oct | 7 | 4 runs: seeds 2 and 3 of `kd17m-n4509`, `kdlarge17m` | `results/summary.md` |
| 31 Oct | 8 | S1 not met (68M PsiloQA: +0.94 [0.64, 1.24] but one run −0.19); S2 met on RAGTruth and PsiloQA, pilot to come; S3 met (95.4%, 100.2%); S4 not met (17M +0.97 [0.53, 1.37] for small-model answers, one run −0.56). 17M runs vary ± 3.8 points | `results/summary.md` |
| 31 Oct | 12 | Chat models as the checker, 300 test answers: Qwen2.5-0.5B 5.4, 1.5B 3.5, 7B 11.1; on the same answers our `kd68m` 59.7, `kd17m` 42.8, the teacher 63.0 | `results/llmlabel_selfcheck*_test.json` |
| 31 Oct | 13 | 8-bit (w8) on all test answers: `kd68m` 261.4 → 73.9 MB, RAGTruth 56.02 → 55.99 (−0.03 [−0.16, 0.09]), PsiloQA 70.64 → 70.64 (0.00); `kd17m` 64.5 → 18.3 MB, RAGTruth −0.22 [−0.53, 0.07], PsiloQA −0.93 [−1.07, −0.80]. On Colab's CPU, `kd68m` took 1.9 s per RAGTruth answer on average | `results/kd*_int8_n2700.json`, `kd*_int8_psilo.json`, `kd*_export.json` |
| 1 Nov | 14 | Laptop: Intel i7-1185G7 (4 cores), 16 GB, Windows 11, ONNX Runtime 1.29; 300 test answers, 772 tokens on average. `kd68m` 8-bit: median 1.66 s on 1 thread (90th percentile 3.72 s), 1.03 s on 2; full precision 1.20 / 0.99 s; memory added 726 MB (8-bit), 905 MB (full). `kd17m` 8-bit: 0.15 s (90th percentile 0.38 s), 0.11 s on 2; 386 MB. S5 time not met for 68M | `work\results\kd*_bench_laptop.json` on the laptop |
| 1 Nov | 10, 11 | Step 10 dropped and step 11 made optional (agreed by Gemechu): test A showed no clear difference | `PLAN.md`, `EVALUATION.md` |
| 1 Nov | 14b | Speed tests added after step 14, as exploratory: `mix` 8-bit recipe and inputs cut to 1,024 tokens; code for them and for step 9 (`density`) written | `run.py`, `colab/plan.ipynb` |
| 1 Nov | 9 | Teacher-marked errors in the new answers: Qwen2.5-0.5B 51.7% of answers (12.3% of characters), Qwen2.5-1.5B 61.0% (8.1%), SmolLM2-1.7B 57.1% (9.0%), Qwen2.5-7B 35.9% (3.2%), Mistral-7B 38.5% (3.7%). On RAGTruth's training answers the teacher's counts are close to people's (it was trained on them) | `results/density.md` |
| 2 Nov | 14b | The `mix` recipe stopped with an error on the real 68M ("Unable to find data type"): after a layer kept in the slower 8-bit form, ONNX could not tell the data type of the next layers. Fixed (they are float) and tested; quantizer logs made quiet. Laptop, inputs cut to 1,024 tokens (planned recipe): median 1.23 s on 1 thread, 0.99 s on 2, 90th percentile 1.99 s, memory added 283 MB (726 MB with full inputs). To run again: the `mix` timings and the step 14b cell on Colab | `work\results\kd68m_bench_laptop_len1024.json` on the laptop |
| 2 Nov | 14b | Accuracy of the `mix` recipe on all 2,700 RAGTruth test answers: 56.02 → 34.59 (−21.43 [−24.52, −18.21]); the export check already showed 18% of word decisions flipped. `mix` fails; the cell stopped there, so variants (b) and (c) were not scored | `results/kd68m_int8_mix_n2700.json`, `kd68m_export_mix.json` |
| 13 Nov | — | Review of the design and the results by Claude: one solid result (the 68M checker), S4 failed, S5 time not met, pilot set not started. `mix` and variant (c) closed. Steps 8b–8e added (other learning rates for 17M on the dev set, a 32M student, their tables, 8-bit and speed), with the rule written in EVALUATION.md section 9 before any run. Code: `lrpick` command, new cells in `plan.ipynb`, TinyLettuce-32M scored on PsiloQA; the 12 command tests pass (the tests that need PyTorch were not run by Claude: run `python -m pytest tests -q` on the laptop) | `EVALUATION.md`, `PLAN.md`, `run.py`, `colab/plan.ipynb` |
| 24 Nov | 8b | Part 1 (seed 42, dev set only): `hard17m-8ep` dev span F1 37.52 → 46.81 with 1e-4 (5e-5 45.13, 2e-4 39.06); `kd17m` 40.89 → 50.76 with 1e-4 (5e-5 49.87, 2e-4 47.91); both take 1e-4. Part 2: seeds 2 and 3 of `hard17m-8ep-lr1e4` and `kd17m-lr1e4`. The 13 November changes pushed (fa674cd); the laptop now uses a Python 3.12 venv instead of conda; 45 tests pass | `results/lrpick_hard17m-8ep.md`, `lrpick_kd17m.md` |
| 25 Nov | 8c | Part 1 (seed 42, dev set only): `hard32m-8ep` keeps 2e-5 (48.13; 5e-5 +0.67, 1e-4 +0.20); `kd32m` takes 1e-4 (50.68 → 53.11; 5e-5 52.06). Part 2: seeds 2 and 3 of `hard32m-8ep` and `kd32m-lr1e4`. About 35 minutes per run on a T4 | `results/lrpick_hard32m-8ep.md`, `lrpick_kd32m.md` |
| 25 Nov | 8d | 17M with 1e-4: `kd17m-lr1e4` 49.71 ± 1.73 / 63.69 ± 1.10 (RAGTruth / PsiloQA); teacher adds +4.85 / +2.79 (all runs); against `kd17m` +4.07 / +0.03 (one run −2.15, does not count); against TinyLettuce-17M +5.45 / −2.76; against TinyLettuce-68M −0.34 / −5.69. 32M: `kd32m-lr1e4` 52.30 ± 1.60 / 67.32 ± 1.26, `hard32m-8ep` 51.12 ± 0.94 / 68.99 ± 0.47; teacher +1.18 (one run −0.27) / −1.67 (−2.32 up to 2B, counts); against TinyLettuce-32M +4.44 / −1.77; against TinyLettuce-68M +2.25 / −2.06; `kd68m` is +3.92 / +3.05 above it. `kd68m` against `ld_base` (150M): +0.79 (all runs higher, interval includes 0) / +0.77 (counts) | `results/summary_added.md` |
| 25 Nov | 8e | 8-bit (w8), seed-42 runs, all test answers: `kd32m-lr1e4` (32.0M parameters) 122.4 → 34.6 MB, RAGTruth 52.98 → 53.08 (+0.10 [−0.06, 0.29]), PsiloQA 68.65 → 68.73 (+0.08 [0.05, 0.12]); `kd17m-lr1e4` 64.5 → 18.3 MB, RAGTruth 51.10 → 51.19 (+0.09 [−0.03, 0.23]), PsiloQA 63.02 → 62.98 (−0.04 [−0.09, 0.00]). Colab CPU, 8-bit, 1 thread: 0.65 s (32M), 0.25 s (17M). Laptop as in step 14, `kd32m-lr1e4` 8-bit: median 0.47 s on 1 thread (90th percentile 1.15 s), 0.36 s on 2; full precision 0.42 / 0.35 s; memory added 572 MB (8-bit), 652 MB (full). The 32M meets the S5 fallback (exploratory, added later); the 68M does not | `results/kd*-lr1e4_export.json`, `_int8_n2700.json`, `_int8_psilo.json`, `_bench_colab.json`; `work\results\kd32m-lr1e4_bench_laptop.json` on the laptop |
| 28 Nov | 16 | Pilot set: 26 passages (16 course, written from the CoSc4035 slides' topics; 10 health, CDC excerpts checked line by line against the live pages) and 67 questions (34 course, 33 health; 51 answered fully by the passage, 8 in part, 8 not at all), all kept after review by Gemechu and the colleague | `pilot/` |
| 28 Nov | 17 | `run.py pilotgen` and the step 17 cell written; tests for the pilot file and the command pass (with a made-up answering model) | `run.py`, `odh/pilot.py`, `colab/plan.ipynb`, `tests/test_pilot.py` |
