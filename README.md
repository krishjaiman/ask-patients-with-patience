# AskPatients

A from-scratch **research reproduction** of:

> Jiayuan Zhu, Jiazhen Pan, Yuyuan Liu, Fenglin Liu, Junde Wu. **"Ask
> Patients with Patience: Enabling LLMs for Human-Centric Medical
> Dialogue with Grounded Reasoning."** EMNLP 2025, pages 2846-2857.

This codebase implements **Dr.APP**: a multi-turn LLM medical dialogue
agent that (1) grounds its reasoning in clinical guideline text, (2)
represents its diagnosis as an explicit, Bayesian-updated probability
distribution over candidate diseases, (3) selects each follow-up
question via **entropy-minimizing Bayesian active learning** rather
than LLM "vibes", and (4) rewrites the selected question into
empathetic, jargon-free language before asking the (simulated)
patient.

> **⚠️ Research / educational use only.** This is not a medical device,
> not clinically validated, and not a substitute for professional
> medical advice. See [Disclaimer](#disclaimer).

---

## Table of contents

- [Paper overview](#paper-overview)
- [Architecture](#architecture)
- [Paper → code mapping](#paper--code-mapping)
- [Installation](#installation)
- [Dataset setup (ReMeDi)](#dataset-setup-remedi)
- [Medical knowledge setup](#medical-knowledge-setup)
- [API / provider configuration](#api--provider-configuration)
- [Running the system](#running-the-system)
- [Running baselines](#running-baselines)
- [Evaluation](#evaluation)
- [Expected outputs](#expected-outputs)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reproduction deviations](#reproduction-deviations--implementation-decisions)
- [Disclaimer](#disclaimer)

---

## Paper overview

LLMs perform well on static medical QA benchmarks but struggle in real
consultations because (a) they tend to diagnose from a single message
without follow-up questions, (b) naive multi-turn prompting produces
mechanical, un-grounded, un-prioritized question sequences, and (c)
their reasoning is a black box with no notion of *how confident* a
diagnosis is or *why* a particular follow-up question was asked.

Dr.APP addresses this with an explicit probabilistic model:

1. Map the dialogue so far to a normalized symptom set, grounded via
   RAG retrieval over clinical-guideline text (MSD Manual in the
   paper).
2. Maintain a disease probability distribution `P_t(D)` over a
   candidate disease set, grounded in each disease's retrieved
   causes/pathophysiology.
3. Generate a pool of candidate follow-up questions, grounded in each
   disease's "Diagnosis" / "What a doctor does" guideline text.
4. For each candidate question, simulate several plausible patient
   responses and estimate `P(response | disease)` for each candidate
   disease.
5. Use Bayes' rule to compute the *expected* posterior distribution
   for each candidate question, and its Shannon entropy.
6. Ask the question with the **lowest expected entropy** (maximum
   expected information gain) — this is the Bayesian active-learning
   core of the method.
7. Rewrite the selected question in warm, jargon-free, example-driven
   language before presenting it to the patient.
8. Once the real patient response arrives, Bayesian-update `P_t(D)`
   and repeat.

## Architecture

```
Patient dialogue
    │
    ▼
Symptom extraction (RAG-grounded)              askpatients.medical.mapping
    │
    ▼
Medical knowledge retrieval                     askpatients.rag
    │
    ▼
Disease probability distribution P_t(D)        askpatients.medical.diagnosis
    │
    ▼
Generate candidate questions Q_{t+1}           askpatients.questions.generator
    │
    ▼
Generate possible responses per question        askpatients.questions.generator
    │
    ▼
Estimate P(response | disease)                  askpatients.questions.generator
    │
    ▼
Bayesian joint / posterior (Eq. 2-3)            askpatients.bayesian.probability
    │
    ▼
Expected entropy per question (Eq. 4)           askpatients.bayesian.entropy
    │
    ▼
Select min-entropy question (Eq. 5)             askpatients.bayesian.entropy
    │
    ▼
Human-centric rewrite                           askpatients.dialogue.humanize
    │
    ▼
Ask patient (simulated)                          askpatients.patient.simulator
    │
    ▼
Receive response → Bayesian update → repeat     askpatients.dialogue.app_agent
```

The whole per-turn loop is orchestrated by `askpatients.dialogue.app_agent.DrAPP`,
driven turn-by-turn by `askpatients.dialogue.orchestrator.DialogueOrchestrator`.

**Critically, the Bayesian math (`askpatients/bayesian/`) contains no
LLM calls.** LLM calls only ever *supply inputs* — a symptom list, a
proposed prior distribution, candidate questions, plausible responses,
per-disease response likelihoods. Every probability computation (joint
probability, posterior, entropy, argmin selection) is deterministic
Python, independently unit-tested in `tests/`.

## Paper → code mapping

| Paper section / equation | Code |
|---|---|
| §2.2 Mapping into MSD | `askpatients/medical/mapping.py` (`SymptomMapper`) |
| §2.3 Diagnosis Probability Prediction, **Eq. 1** | `askpatients/medical/diagnosis.py` (`DiagnosisEngine`), `askpatients/bayesian/probability.py` (`normalize_distribution`) |
| §2.4 Question Generation | `askpatients/questions/generator.py` (`QuestionGenerator.generate_candidate_questions`) |
| §2.5 One-more Step Thinking & Conditional Probability, **Eq. 2** | `askpatients/questions/generator.py` (`generate_possible_responses`, `estimate_conditional_probabilities`), `askpatients/bayesian/probability.py` (`joint_probability`) |
| §2.6 Question Selection, **Eq. 3-5** | `askpatients/bayesian/probability.py` (`posterior_given_question`), `askpatients/bayesian/entropy.py` (`shannon_entropy`, `expected_entropy_for_question`, `select_min_entropy_question`) |
| §2.7 Human-Centric Communication | `askpatients/dialogue/humanize.py` (`HumanCentricRewriter`) |
| §3.1 Dataset (ReMeDi filtering/sampling) | `askpatients/data/remedi_loader.py` |
| §3.2 Patient Simulator | `askpatients/patient/profile.py`, `askpatients/patient/simulator.py` |
| §3.3 Experimental Setup (K, L, turns) | `askpatients/config.py` (`AppAlgoConfig`), `configs/default.yaml` |
| §3.4.1 Accuracy | `askpatients/evaluation/metrics.py` (`judge_diagnosis_accuracy`) — **LLM-judge approximation**, see below |
| §3.4.2 Entropy | `askpatients/bayesian/entropy.py` (reused directly — faithfully reproduced) |
| §3.4.3 Human-Centric | `askpatients/evaluation/metrics.py` (`judge_human_centric_quality`) — **LLM-judge approximation** |
| Fig. 1(a) one-shot baseline | `askpatients/dialogue/baselines.py` (`OneShotBaseline`) |
| Fig. 1(b) naive multi-turn baseline | `askpatients/dialogue/baselines.py` (`NaiveMultiTurnBaseline`) |

## Installation

Requires Python 3.10+.

```bash
git clone <this-repo>
cd askpatients
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional extras for a real (non-mock) run:
pip install openai anthropic faiss-cpu sentence-transformers matplotlib
```

No installation is required to explore the code offline: every script
supports `--mock`, which uses `MockLLM` and `MockEmbedder` (no network,
no API key) so you can run the entire pipeline immediately.

## Dataset setup (ReMeDi)

This repository does **not** bundle the ReMeDi dataset (Yan et al.,
2022, SIGIR) — see [`data/remedi/README.md`](data/remedi/README.md)
for how to obtain it and the exact expected schema. Once you have a
raw export:

```bash
python scripts/prepare_dataset.py \
  --raw data/remedi/raw/remedi_base.jsonl \
  --out data/processed/remedi_diagnosis_only.jsonl \
  --eval-out data/processed/remedi_eval_100.jsonl \
  --eval-n 100 --seed 42
```

This reproduces §3.1's filtering exactly: keep only dialogues whose
doctor turns are *exclusively* labeled `"Diagnosis"`, then randomly
sample 100 (seeded).

For offline experimentation without the real dataset, a small
**synthetic, hand-written** sample in the correct schema is provided
at `data/remedi/synthetic_sample.jsonl` (clearly not real ReMeDi data
— see the file's header comment in the README).

## Medical knowledge setup

Also not bundled: the MSD Manual content itself is copyrighted. A
small **original, hand-written** 11-disease sample knowledge base
ships at `data/knowledge_base/sample_diseases.json` so the pipeline
runs out of the box. To reproduce the paper's actual grounding source,
supply your own legally-obtained MSD-Manual-derived (or other
guideline) corpus in the same schema — see
[`data/knowledge_base/README.md`](data/knowledge_base/README.md).

```bash
python scripts/build_index.py --mock            # sample KB, mock embeddings
python scripts/build_index.py --kb data/knowledge_base/your_file.json  # real KB
```

## API / provider configuration

Copy `.env.example` to `.env` and fill in whichever provider you use:

```bash
cp .env.example .env
# then edit .env: OPENAI_API_KEY=... or ANTHROPIC_API_KEY=...
```

Then set `llm.provider` (and, for real retrieval quality,
`rag.embedding_provider`) in `configs/default.yaml` or your own config
file to `openai` or `anthropic` / `openai` / `sentence-transformers`.
Every script also accepts `--mock` to force everything to run offline
regardless of the config file.

All hyperparameters from §3.3 are in `configs/default.yaml`:
`K=5` candidate questions, `2<=L<=5` simulated responses per question,
6 dialogue turns.

## Running the system

Run Dr.APP on a single case against the patient simulator:

```bash
# fully offline
python scripts/run_app.py --mock \
  --case data/remedi/synthetic_sample.jsonl --case-id SYN-001 --turns 3

# with a real LLM (after configuring .env and configs/default.yaml)
python scripts/run_app.py \
  --case data/processed/remedi_eval_100.jsonl --case-id <some_id>
```

This prints, per turn: the clinically-selected question, its
human-centric rewrite, the simulated patient's answer, and the entropy
after the Bayesian update — plus the final diagnosis and full
probability distribution.

## Running baselines

```bash
python scripts/run_baseline.py --mock --method one_shot \
  --case data/remedi/synthetic_sample.jsonl --case-id SYN-001

python scripts/run_baseline.py --mock --method naive_multi_turn \
  --case data/remedi/synthetic_sample.jsonl --case-id SYN-001
```

## Evaluation

```bash
python scripts/evaluate.py --mock \
  --cases data/remedi/synthetic_sample.jsonl \
  --methods one_shot naive_multi_turn app \
  --out-dir outputs/smoke_eval

# real run over the paper's 100-case subset
python scripts/evaluate.py \
  --cases data/processed/remedi_eval_100.jsonl \
  --methods one_shot naive_multi_turn app \
  --out-dir outputs/full_eval
```

Produces `results.csv`, `results.json`, `summary.json` (per-method
means), and — for the `app` method — `entropy_across_iterations.png`,
a Figure-4-style plot of average entropy per turn (only meaningful
across multiple evaluated cases).

## Expected outputs

- `outputs/<run>/results.json` — one record per (case, method): predicted
  diagnosis, accuracy score, entropy trajectory (APP only), human-centric
  sub-scores.
- `outputs/<run>/summary.json` — per-method means matching the shape of
  the paper's Table 1 / Table 2 (not the same numbers — see deviations
  below).
- `outputs/<run>/entropy_across_iterations.png` — average entropy vs.
  turn index for Dr.APP, analogous to Figure 4.

## Testing

```bash
pytest tests/ -q
```

Includes:
- `test_probability.py` — Eq. 1-3 (normalization, joint probability,
  posterior marginalization, Bayesian update) with hand-computed
  expected values.
- `test_entropy.py` — Eq. 4-5 (Shannon entropy properties, expected
  entropy, argmin selection, tie-breaking).
- `test_question_selection.py` — end-to-end check that a more
  informative candidate question is actually selected over a less
  informative one (see the note in that file about how Eq. 3's
  unweighted marginalization behaves on symmetric hypothetical
  responses).
- `test_retriever.py` — chunking, mock embeddings, FAISS/numpy
  retrieval relevance and save/load round-trip.
- `test_smoke_e2e.py` — full offline pipeline run (LLM → RAG →
  Bayesian math → entropy selection → human-centric rewrite → patient
  response → Bayesian update), plus both baselines and the dataset
  preprocessing pipeline.

All 33 tests pass with `--mock` (no network / API key required), and
every command in this README's "Running the system" / "Running
baselines" / "Evaluation" sections has been executed against
`--mock` as part of preparing this repository.

## Troubleshooting

- **`ImportError: openai package required`** — install the optional
  provider dependency (`pip install openai`) or use `--mock`.
- **`OPENAI_API_KEY is not set`** — copy `.env.example` to `.env` and
  fill it in, or pass `--mock`.
- **`FileNotFoundError: ReMeDi raw data not found`** — you need to
  supply your own ReMeDi export; see `data/remedi/README.md`. Use
  `data/remedi/synthetic_sample.jsonl` for offline testing.
- **`faiss` not installed** — the retriever automatically falls back
  to a pure-numpy brute-force cosine search; correctness is
  unaffected, only large-scale performance.
- **Mock-mode output looks semantically odd** (e.g. a follow-up
  question about the wrong symptom) — expected: `MockLLM` and
  `MockEmbedder` are deterministic placeholders for *data shape*, not
  a stand-in for real reasoning quality. Swap `llm.provider` /
  `rag.embedding_provider` to a real backend for meaningful output.

## Reproduction deviations / implementation decisions

Read this section before citing any numbers produced by this codebase
as a comparison to the paper's reported results.

**1. Faithfully reproduced (the core claims of the paper):**
- The Bayesian joint-probability, posterior-marginalization, and
  Shannon-entropy computations (Eq. 1-5) are implemented exactly as
  specified, in pure Python/numpy, independent of any LLM call, and
  unit-tested against hand-computed values.
- Minimum-expected-entropy question selection (Eq. 5) is a genuine
  `argmin` over computed entropies — never an LLM ranking call.
- The K=5 / 2≤L≤5 / 6-turn hyperparameters from §3.3.
- The §3.1 dataset filtering/sampling logic (Diagnosis-only filter,
  seeded 100-case sample).
- The overall pipeline shape (symptom mapping → grounded diagnosis
  distribution → grounded question generation → response simulation →
  Bayesian update → entropy selection → human-centric rewrite).

**2. Implementation choices / approximations (paper under-specifies
these, or they require resources this repo cannot legally
redistribute):**
- **MSD Manual content**: not redistributed; a small original sample
  knowledge base stands in for it. Grounding *quality* in a real run
  depends entirely on the corpus you supply.
- **ReMeDi's exact raw JSON schema**: not published in the paper; we
  define an explicit, documented schema and a single adapter function
  (`_coerce_case`) to convert from whatever the real export uses.
- **Candidate disease set D**: fixed at dialogue initialization from
  the opening symptoms (rather than recomputed from scratch each
  turn), so the entropy trajectory is comparable across turns. The
  paper does not specify whether D is static or dynamic.
- **Entropy log base**: natural log; the paper doesn't specify a base.
  This only rescales entropy values, never changes the Eq. 5 argmin.
- **Numerical floor** (`1e-6` by default) applied when normalizing
  distributions, to avoid `log(0)` / division-by-zero and to keep
  every disease Bayesian-recoverable. Not specified in the paper.
- **Diagnosis-accuracy and human-centric evaluation (§3.4.1, §3.4.3)**:
  the paper used 3 human biomedical experts and 5 human non-medical
  raters respectively. This repository substitutes an **LLM-as-judge**
  using the same rubrics (Appendix A.1/A.2) as a scalable
  approximation — this is explicitly *not* a reproduction of the
  paper's human evaluation, and resulting scores should not be
  directly compared to Table 1 / Table 2.
- **Patient-profile construction model**: the paper uses DeepSeek-v3
  specifically; this repo is provider-agnostic (whatever
  `llm.provider` is configured).

**3. Components requiring external data/API access to run for real**
(all have documented, functioning offline substitutes for
development/testing):
- The real MSD Manual (or equivalent licensed clinical guideline
  corpus) — offline substitute: `data/knowledge_base/sample_diseases.json`.
- The real ReMeDi dataset — offline substitute:
  `data/remedi/synthetic_sample.jsonl`.
- A real LLM provider (OpenAI/Anthropic) for genuine reasoning quality
  — offline substitute: `MockLLM`.
- A real embedding provider for genuine semantic retrieval — offline
  substitute: `MockEmbedder` (deterministic hashed bag-of-words).

## Disclaimer

This repository is provided for **research and educational purposes
only**. It is not a certified medical device, has not been clinically
validated, and must not be used to make real medical decisions for
yourself or anyone else. If you have a medical concern, consult a
qualified healthcare professional.
