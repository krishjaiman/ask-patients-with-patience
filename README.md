# AskPatients

Research implementation of **Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning** (EMNLP 2025).

This is an educational prototype. Mock runs verify software behavior, not diagnostic accuracy. A successful real-provider demo and a reproduction of the published experiments are separate milestones.

## Quick start (Windows PowerShell)

Use the existing virtual environment from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_app.py --mock --case data/remedi/synthetic_sample.jsonl --case-id SYN-001 --turns 3
.\.venv\Scripts\python.exe scripts/chat.py --mock --turns 2
```

For a fresh environment, create a Python 3.10+ virtual environment and install `requirements.txt`. Entry-point scripts add `src` to the import path automatically.

## Free hosted demo (no GPU required)

`configs/free_demo.yaml` uses Groq's chat-completions endpoint and a model listed in its Free-plan documentation as checked on 2026-10-06. It does not use a paid OpenAI API key. Provider availability and quotas must be confirmed for your account.

1. Create a Free account at https://console.groq.com and create an API key.
2. Add `GROQ_API_KEY=...` to your local `.env`. Keep existing entries. Never put the key in chat or commit it.
3. Stay on the Free plan; do not enable paid billing for this setup.
4. Check prerequisites, then run one synthetic case:

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py --config configs/free_demo.yaml
.\.venv\Scripts\python.exe scripts/run_app.py --config configs/free_demo.yaml --case data/remedi/synthetic_sample.jsonl --case-id SYN-001 --turns 1
.\.venv\Scripts\python.exe scripts/chat.py --config configs/free_demo.yaml
```

The compatible adapter uses Python's standard library, so no extra SDK is needed. The free configuration uses three candidate questions, two hypothetical responses each, three follow-up rounds, and up to five candidate diseases. These are **reduced demo settings**, not the paper's full experiment. See [PRESENTATION.md](PRESENTATION.md) for the saved-report workflow and presentation notes.

Requests are paced 15 seconds apart across roles sharing an endpoint/model. Successful completions are cached under `outputs/llm_cache`, and uncached HTTP attempts are capped at 80 per endpoint/model per process. Quota/authentication errors stop rather than invoking a paid fallback. The cap is not a currency limit: remain on the provider's Free plan. Cache hits replay prior responses and must not be treated as independent experimental repeats.

Retrieval in this configuration is local BM25 keyword matching (`lexical`), not semantic embeddings. It requires no model downloads or GPU. Semantic retrieval can later use a CPU sentence-transformer with the optional `sentence-transformers` dependency and a matching `rag.embedding_model`. Mock embeddings remain available for tests only.

For a local chat-completions server, configure `llm.provider: compatible`, its `base_url`, and its installed `model` name. Loopback endpoints do not require a key. No local model has been installed or benchmarked on this laptop.

Official provider references:
- https://console.groq.com/docs/rate-limits
- https://console.groq.com/docs/model/openai/gpt-oss-20b
- https://console.groq.com/docs/openai

## Methodology and implementation choices

The pipeline maps dialogue to symptoms, retrieves clinical context, estimates disease weights, proposes follow-up questions, imagines answers, estimates per-disease response likelihoods, scores each question with the printed paper formula, and rewrites the selected question for the patient.

**Question selection:** Equations 2-5 are retained. The printed formula takes entropy after marginalizing hypothetical answers. It is not standard expected posterior entropy. With a complete normalized response partition it returns the prior, even for a perfectly discriminating question. A regression test preserves this distinction. Standard expected information gain is a future experiment, not silently substituted here.

**Observed-answer update:** `algo.belief_update: dialogue` estimates disease probabilities from the complete updated conversation and guidelines, consistent with the conditioning described in Section 2.3. `bayesian` retains the previous recursive likelihood multiplication as an explicitly labeled approximation, now with prior dialogue in the likelihood prompt. Neither mode establishes calibrated probabilities. Candidate diseases remain fixed at initialization; missing diseases cannot enter later.

**Patient profiles:** the profile builder extracts symptoms, timing, triggers, negative findings and history with exact source quotes. Unknown age remains unknown. Quotes are checked against the source and literal diagnosis-label leakage is rejected. These checks do not prove semantic entailment; profiles still require review. The simulator receives the profile, not the original transcript. Profiles are cached by source case and model configuration under `data/processed/profiles` and reused across methods.

**Opening:** all methods use the initial patient complaint; multi-turn methods prepend a neutral greeting. Original doctor utterances are never replayed as the opening question.

**Patient answers:** the simulator selects numbered profile facts and code returns their exact statements, with unknown details acknowledged explicitly. This is a conservative departure from the paper's free-form patient persona. It prevents generated paraphrases from adding clinical findings; it does not validate the extracted profile or guarantee the relevance of selected facts.

**Retrieval fallback:** when symptom extraction is empty, candidate retrieval uses the opening patient complaint instead of the first entries in the KB. BM25 ignores common English stopwords, and candidate retrieval excludes nonpositive matches. Missing relevant candidates now fail explicitly rather than using arbitrary diseases.

**Model roles:** `llm` is the doctor. Optional `profile_llm`, `patient_llm`, and `judge_llm` blocks accept the same configuration; null inherits the doctor configuration. Sharing a model is convenient for a demo but creates evaluation bias. The old unused `patient` settings have been removed.

**Validation:** likelihoods must be finite and in [0, 1]; all candidate diseases must be present. Initial weights must be nonnegative with positive mass. Hypothetical answers must be distinct and meet the configured count; invalid outputs trigger structured-output retries, never synthetic answer padding. The judge rubric excludes score 1.

**Grounding:** the bundled KB is 11 original illustrative entries, not MSD content. Source title, URL and section fields are supported, but the bundled entries have no source citations. Disease-centric records approximate the paper's professional/consumer symptom-page representation. Retrieval index files are reused only when KB and retrieval configuration fingerprints match.

## ReMeDi is publicly available

The authors publish ReMeDi-base and state that all resources use the MIT license:
- https://github.com/yanguojun123/Medical-Dialogue
- https://github.com/yanguojun123/Medical-Dialogue/blob/main/data/ReMeDi-base.json
- https://github.com/yanguojun123/Medical-Dialogue/blob/main/MIT-license.txt

The downloaded original and license are in `data/remedi/raw` (gitignored). The release contains **1,557 Chinese conversations**. Its actual schema is `dialogue`, `information`, `sentence`, and per-turn `actions`; the adapter preserves all action labels.

Inspection found:
- 386 conversations containing a doctor Diagnosis action.
- 0 conversations whose annotated doctor actions are exclusively Diagnosis.
- 280 conversations with exactly one distinct nonempty Diagnosis/disease `value1` annotation.

The paper reports 329 filtered cases. Its exact selection cannot be recovered from the stated rule alone. `single_diagnosis` is a documented provisional policy, not a reproduction claim:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --raw data/remedi/raw/ReMeDi-base.json --out data/processed/remedi_single_diagnosis.jsonl --eval-out data/processed/remedi_eval_100_provisional.jsonl --filter single_diagnosis --eval-n 100 --seed 42
```

This writes a SHA-256 audit, policy, counts and selected IDs. Disease labels are provisional annotation-derived references. Multiple-diagnosis cases are excluded rather than arbitrarily choosing a label. No specialty or translation is invented. The Chinese labels need review and terminology alignment with a larger guideline KB before meaningful evaluation. The eight bundled English synthetic cases remain the immediate demo input.

## Outputs and evaluation

`run_app.py --out outputs/demo/result.json` saves configuration, patient profile and full dialogue trace. `chat.py --out outputs/chat/session.json` saves interactive sessions. Traces retain clinical and actual rewritten questions, imagined responses, per-disease likelihoods, question scores, grounding records and disease distributions.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --mock --cases data/remedi/synthetic_sample.jsonl --methods one_shot naive_multi_turn app --out-dir outputs/smoke_eval
```

Evaluation produces CSV/JSON, summary and a manifest recording configuration, input hash, KB and selected cases. Failed runs are explicit rows excluded from score means, counted in the summary, and cause a nonzero command exit. Unavailable metrics are null rather than zero. Optional `matplotlib` generates APP's entropy plot; baseline entropy is not implemented. Accuracy is an LLM-judged 0/2/3/4/5 similarity score, not percent correct. Mock scores have no scientific meaning.

## Remaining work before research claims

- Extend real-model conversation checks beyond the single synthetic demonstration.
- Supply a sufficiently complete, source-linked guideline corpus and verify retrieval coverage.
- Review the provisional ReMeDi labels, language handling and case selection.
- Add independent/human evaluation and the missing paper experiment conditions.
- Measure accuracy and calibration alongside entropy; low entropy can be confidently wrong.
- Only then compare improvements such as expected information gain, candidate expansion and source-linked explanations.

This software is not clinically validated and must not be used to make real medical decisions.
