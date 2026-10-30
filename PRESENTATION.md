# Presentation runbook

This demo exercises APP's printed question-scoring formula using a real hosted
LLM, a synthetic patient case, and an illustrative 11-disease knowledge base.
It is not a full reproduction of the paper's evaluation.

## Verified demonstration (2026-10-07)

The corrected run is saved at `outputs/presentation/validated_run.json`, with
an offline report at `outputs/presentation/report.html`. It completed three
rounds with Groq's `openai/gpt-oss-20b` in 11.51 minutes, using 47 HTTP attempts
and 31,275 reported tokens. The profile was reused from its existing cache;
runtime and token counts exclude that earlier profile-generation work.

The final label was Lactose Intolerance, matching synthetic case SYN-001's
reference, with model weight 0.75. Entropy was 0.9986 → 0.8979 → 0.6874 → 0.6874
nats. This single-case match is not an accuracy measurement.

Observed limitations worth showing: the first unknown answer still shifted the
doctor's estimates; the third question repeated the first question's topic; and
the patient supplied a recorded dairy trigger while answering a stool question.
The extractive simulator prevents new clinical wording but does not guarantee
ideal fact relevance or conversational realism. Repetition control and belief
stability after unknown answers are useful future experiments.

Verification: 54 automated tests passed, as did 24 offline mock runs across
eight cases and three methods. The live trace's probability sums and question
selection were checked, and the HTML report was generated and content-checked.
Visual browser preview was unavailable because the automation browser blocks
local file URLs; open the report locally before presenting.

## Before presenting

From PowerShell in `D:\askpatients`:

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py --config configs/free_demo.yaml
.\.venv\Scripts\python.exe -m pytest -q
```

The setup check never displays the API key. Keep the key only in `.env`, and
keep the Groq account on its Free plan. Do not show `.env` on the projector.

## Generate a real-model run

```powershell
.\.venv\Scripts\python.exe -u scripts/run_app.py --config configs/free_demo.yaml --case data/remedi/synthetic_sample.jsonl --case-id SYN-001 --turns 3 --out outputs/presentation/validated_run.json
.\.venv\Scripts\python.exe scripts/presentation_report.py --input outputs/presentation/validated_run.json --out outputs/presentation/report.html
```

Allow several minutes for a new run: requests are paced for the free quota.
Subsequent runs may reuse cached model responses; explicitly describe these as
replays, not independent experiments. Completed turns are checkpointed. If the
command fails, inspect the saved status and rerun after resolving the cause;
cached completions reduce repeated API calls.

## Present without internet

Open `outputs/presentation/report.html` in a browser. It contains the conversation,
the candidate-question scores, imagined answers and likelihoods, and the disease
probabilities before and after the actual answers. Opening it calls no API.
Keep the corresponding JSON as the machine-readable evidence.

Suggested five-minute explanation:

1. Explain the problem: the opening patient complaint may omit diagnostic clues.
2. Show the neutral opening and initial disease distribution.
3. Expand a round's evidence: candidate questions, imagined answers and scores.
4. Show the selected question and its patient-friendly wording.
5. Show how actual answers change the disease distribution and entropy.
6. State the limitations and next research experiments.

## Explain the method in plain language

APP treats a consultation as a sequence of information-gathering decisions.
The initial complaint is usually incomplete. The doctor first maps it to symptoms,
retrieves relevant medical descriptions, and asks the LLM to assign relative
weights to possible diseases. The code normalizes these weights to sum to one.

For each proposed follow-up question, another model call imagines possible patient
answers. Further calls estimate how plausible each answer would be under each
candidate disease. The program multiplies those likelihoods by the current disease
weights, combines the imagined-answer evidence according to the paper's printed
formula, and calculates its entropy. It selects the question with the lowest score.

The selected clinical question is rewritten for the patient. The actual answer
then joins the conversation, and the doctor estimates disease weights again.
The simulator's private profile is kept separate from what the doctor can see.
Only answers and the opening complaint enter the doctor's dialogue.

Distinguish three things on screen: imagined answers used to score a question,
the actual simulated patient's answer, and the posterior estimated after that
actual answer. The question's score is not the same quantity as observed entropy
after the answer. Neither a smaller score nor a more confident final distribution
proves a correct diagnosis. A run ends at the configured round limit, not because
the system has established clinical certainty.

## Claims to make accurately

- This is a working demonstration with an LLM, retrieval and explicit arithmetic.
- The free demo uses three candidate questions, two hypothetical answers and three
  follow-up rounds, rather than the paper's full settings.
- Retrieval is BM25 keyword matching, not semantic embeddings.
- The KB entries are illustrative, not an ingested MSD corpus.
- Model confidence is not measured diagnostic accuracy or calibration.
- The printed aggregate-entropy formula is retained; it has a known degeneracy
  for complete normalized response sets. Standard expected information gain is a
  future comparison, not an already established improvement.
- ReMeDi-base was obtained publicly, but the paper's exact subset remains
  unverified. The provisional Chinese subset is not used in this English demo.
- The patient, doctor and profile builder share the demo backend. Independent
  evaluation, corpus expansion, and real benchmark alignment remain future work.
- The simulator selects recorded profile facts and returns their exact statements.
  Missing details remain unknown. This conservative implementation departs from
  the paper's free-form persona role-play; it prevents new answer wording from
  inventing findings, but profile extraction and fact selection still need review.
- Empty symptom extraction falls back to retrieval from the opening complaint.
  Lexical retrieval discards zero-score matches; five diseases is an upper bound.

The earlier `outputs/presentation/live_run.json` is retained as a debugging
artifact. It exposed invented patient denials and irrelevant fallback candidates.
Use the corrected `validated_run.json` and its report for the presentation.

## Optional interactive demonstration

```powershell
.\.venv\Scripts\python.exe scripts/chat.py --config configs/free_demo.yaml --turns 1
```

Use a fictional case. Keep the saved report ready in case the provider quota or
internet connection prevents a live interaction.
