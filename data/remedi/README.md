# ReMeDi dataset

This repository does **not** bundle the ReMeDi dataset. ReMeDi
("Resources for multi-domain, multi-service, medical dialogues") was
introduced by Yan, Pei, Ren, Ren, Xin, Liang, de Rijke, and Chen at
SIGIR 2022:

> Guojun Yan, Jiahuan Pei, Pengjie Ren, Zhaochun Ren, Xin Xin, Huasheng
> Liang, Maarten de Rijke, Zhumin Chen. "ReMeDi: Resources for
> Multi-domain, Multi-service, Medical Dialogues." SIGIR 2022.

## Obtaining the data

Obtain ReMeDi-base from its original release (check the paper / the
authors' repository for the current distribution link and license
terms) and place the raw export under `data/remedi/raw/`.

## Expected schema

`askpatients.data.remedi_loader.load_remedi_raw` expects either a
`.jsonl` file (one case per line) or a single `.json` array file, with
records shaped like:

```json
{
  "case_id": "12345",
  "specialty": "Gastroenterology",
  "ground_truth_diagnosis": "Gastroesophageal Reflux Disease",
  "turns": [
    {"role": "patient", "text": "I've had heartburn for a week."},
    {"role": "doctor", "text": "Does it get worse after eating?", "action": "Diagnosis"},
    {"role": "patient", "text": "Yes, especially at night."}
  ]
}
```

If your actual ReMeDi export uses different field names (e.g.
`dialogue` instead of `turns`, `department` instead of `specialty`),
either rename them to match this schema, or adapt the single adapter
function `_coerce_case` in
`src/askpatients/data/remedi_loader.py` -- that is the one place the
rest of the pipeline depends on for field names.

## Preprocessing pipeline (Section 3.1 of the paper)

```bash
python scripts/prepare_dataset.py \
  --raw data/remedi/raw/remedi_base.jsonl \
  --out data/processed/remedi_diagnosis_only.jsonl \
  --eval-out data/processed/remedi_eval_100.jsonl \
  --eval-n 100 --seed 42
```

This filters ReMeDi-base down to cases whose doctor turns are
exclusively labeled `"Diagnosis"` (329 cases in the paper) and then
randomly samples 100 of them (seeded, so the same 100 cases are
selected every run with the default seed) for the evaluation subset,
exactly as Section 3.1 describes.

## Synthetic smoke-test data

`data/remedi/synthetic_sample.jsonl` contains a handful of **entirely
synthetic, hand-written** mini-cases (NOT real ReMeDi data) in the
expected schema, used only so `scripts/prepare_dataset.py`,
`scripts/run_app.py --mock`, and the test suite can run end-to-end
without requiring the real dataset.
