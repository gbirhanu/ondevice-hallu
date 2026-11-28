# Pilot test set (steps 16-20 of PLAN.md)

A small test set from the real use: questions over course notes and health guidelines, answered by phone-sized
models (step 17) and marked word by word by two people (steps 18-19). It is a test set only: nothing is ever
trained on it (rule R6).

## Files
- `documents/course/c01.md` ... `c16.md`: 16 course passages, one topic each.
- `documents/health/h01.md` ... `h10.md`: 10 health passages, one topic each.
- `questions.jsonl`: one question per line, with its passage. Fields: `id`, `domain` (course or health), `doc_id`,
  `topic`, `question`, `answerable` (yes, partly or no: whether the passage answers it), `question_type`,
  `reference_answer` (a short answer taken from the passage, for the annotators), `status`, `source`, `licence`,
  `passage`.
- `answers.jsonl` (step 17): the answers of the phone-sized models.

## Sources and licences
- Course passages: written for this project, in plain words, from the topics of the CoSc4035 Computer Security
  course slides (Jimma University), chapters 1-7. No slide text is copied, because the slides contain text from
  textbooks and other courses. Licence: CC BY 4.0.
- Health passages: excerpts copied word for word from pages of the US Centers for Disease Control and Prevention
  (CDC), accessed on 2026-11-28. CDC website material is in the public domain unless noted otherwise; the
  material is available on the CDC website at no charge, and its use here does not imply endorsement by CDC.
  Each document lists the sections that were left out. The only other changes: line breaks caused by links were
  joined, and footnote numbers were removed. Every line of 30 or more characters was checked against the live page.
  - `h01` Malaria: [About Malaria](https://www.cdc.gov/malaria/about/index.html), page dated 2024-03-12
  - `h02` Signs and symptoms of tuberculosis: [Signs and Symptoms of Tuberculosis](https://www.cdc.gov/tb/signs-symptoms/index.html), page dated 2025-01-17
  - `h03` Treating tuberculosis: treatment plans and side effects: [Treating Tuberculosis](https://www.cdc.gov/tb/treatment/index.html), page dated 2025-04-17
  - `h04` Treating tuberculosis: completing treatment: [Treating Tuberculosis](https://www.cdc.gov/tb/treatment/index.html), page dated 2025-04-17
  - `h05` Cholera: [Cholera](https://www.cdc.gov/cholera/about/index.html), page dated 2025-05-29
  - `h06` Treating cholera: [Treating Cholera](https://www.cdc.gov/cholera/treatment/index.html), page dated 2025-05-29
  - `h07` Measles symptoms and complications: [Measles Symptoms and Complications](https://www.cdc.gov/measles/signs-symptoms/index.html), page dated 2026-08-05
  - `h08` Hand hygiene for healthcare workers: [Clinical Safety: Hand Hygiene for Healthcare Workers](https://www.cdc.gov/clean-hands/hcp/clinical-safety/index.html), page dated 2024-02-27
  - `h09` Rabies: [About Rabies](https://www.cdc.gov/rabies/about/index.html), page dated 2026-01-27
  - `h10` Typhoid fever and paratyphoid fever: [About Typhoid Fever and Paratyphoid Fever](https://www.cdc.gov/typhoid-fever/about/index.html), page dated 2024-04-25

## Questions
67 questions, all kept after the review by Gemechu Birhanu and the colleague on 28 November 2026:
- Course: 34 questions over 16 passages (28 answered fully by the passage, 3 in part, 3 not at all).
- Health: 33 questions over 10 passages (23 fully, 5 in part, 5 not at all).
Questions that the passage answers only in part or not at all check whether a model makes up an answer. The set
was fixed before any model answered it.
