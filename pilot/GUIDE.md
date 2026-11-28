# Marking guide for the pilot answers (steps 18-19)

You mark, word by word, the parts of each answer that **the passage does not support**. The passage is the only
source of truth: what you know about computer security or health does not count. Your marks become the answer key
that every checker (ours and the published ones) is scored against in step 20.

Both annotators mark **alone**. Do not look at each other's marks and do not discuss single answers until both have
finished a round.

## What you see

For each answer: the question, the passage (the "document" the answering model was given) and the answer. Answers
are grouped by question, so you read a passage once and then mark up to four answers about it. The models that wrote
the answers are not shown. Answers that are only "Unable to answer based on given passages." are left out: they make
no claim.

## What to mark

Mark a part of the answer when it is one of these four types (the types used by RAGTruth):

| Type | Meaning | Example (unsupported part in bold) |
|---|---|---|
| **Evident conflict** | Says something the passage clearly contradicts: a wrong number, name, step or the opposite claim | Passage: dogs cause "more than 95% of the estimated 70,000 human deaths each year". Answer: "**about 60%** of human rabies deaths worldwide are caused by dogs, while **approximately 40,000 to 50,000** people die of rabies each year" |
| **Subtle conflict** | Changes the meaning of the passage slightly: a weaker or stronger word, a changed condition or scope | Passage: "more than 95%". Answer: "**approximately** 95% of the estimated 70,000 human deaths" |
| **Evident baseless information** | Adds a clear claim the passage does not contain | Question: which kind of vulnerability causes the most incidents? (The passage lists kinds but never says which causes the most.) Answer: "the kind of vulnerability causing the most security incidents in organizations would be **physical vulnerabilities**" |
| **Subtle baseless information** | Adds a small detail, reason, generalisation or inference that goes beyond the passage | Passage: risk is "the chance that a given threat will exploit a given vulnerability and lead to a given harmful result". Answer: "it quantifies the potential negative impact ..., taking into account both the probability of such an incident occurring **and the severity of its consequences**" |

More examples from the pilot answers:
- "Encephalitis ... **can also be a leading cause of fatality in young measles patients**": the passage gives
  encephalitis numbers but never calls it a leading cause of death. Evident baseless.
- "an adult with inactive TB **typically takes a daily dose** of either isoniazid, rifampin, or rifapentine": the
  passage gives no doses and no daily schedule. Subtle baseless (the medicine names themselves are supported).
- "isoniazid ... **often administered alongside other antibiotics**": not in the passage. Subtle baseless.

If you cannot decide between evident and subtle, choose the one that seems closer and write a short note. The
difference between conflict and baseless matters more than the difference between evident and subtle.

## How much to mark

- Mark the **shortest part** that carries the unsupported claim: usually a few words or a phrase, not the whole
  sentence. Mark the whole sentence only when all of it is unsupported.
- One mark per separate claim. In a list, mark only the wrong or added items.
- The page snaps a selection to whole words; you do not need to be precise to the letter.

## What not to mark

- Wording that keeps the meaning: paraphrases, synonyms, a different order, short summaries.
- Phrases about the source: "Based on the provided passage", "According to the text", "This is according to the
  passage".
- Repeating the question.
- The sentence "Unable to answer based on given passages." It is not an error, even when the passage does answer
  the question. But mark any unsupported claim that comes before or after it.
- Correct statements that the passage does not answer something ("The passage does not say how long this takes")
  when that is true. If the passage does say it, mark the statement as a conflict.
- Missing information. Only what is written can be wrong; an incomplete answer is not an error.
- Generic advice or hedges that make no claim about the topic: "consult your health care provider", "it is
  important to follow the guidelines". But mark advice that adds a specific claim the passage does not make (a dose,
  a duration, a cause, a recommendation).
- Formatting: bold, numbering, line breaks.

## How to work

The page shows three texts. The **passage** is the truth; you only read it. The **question** is what the model
was asked; you only read it. The **answer** (the box marked ANSWER) is what the model wrote; this is the only text
you mark.

1. Read the passage, then the question, then the answer.
2. Check every claim in the answer against the passage. If some words in the answer say something the passage does
   not say, or say it differently, select those words in the answer with the mouse (or by touch). The page then
   asks what is wrong with them: press one of the four types. Click a mark to change its type or remove it.
3. When you have checked the whole answer, press the green **Finish** button. With marks it reads
   "Finish this answer (2 marks)"; with no marks it reads "No errors: finish this answer", which records that the
   whole answer is supported. The page then shows the next answer. An answer where Finish was not pressed counts
   as not yet marked.
4. **Previous answer** and **Next answer** move one answer back or forward without finishing anything.
   **Go to the next unfinished answer** jumps to the next answer you have not finished. The numbered squares show
   every answer in the round and its state; click one to go to it.
5. Use the note box for doubts ("not sure if 'about' is a conflict"). Notes help when you compare.
6. Your marks save as you work. You can stop and come back.

## Rounds

1. **Practice round (10 answers).** Both mark the same 10 answers. Claude computes how far you agree; you then
   discuss the differences and this guide is corrected where it was unclear.
2. **Full round (179 answers, about 1.5 days each).** Both mark every answer alone (the practice answers again,
   with the corrected guide). Claude computes word-level agreement (Cohen's kappa) and span F1 between you. If
   kappa is below 0.6, you discuss the disagreements and revise.
3. **Final marks.** You agree one marking per answer. These are the answer key for step 20.
