# Knowledge base data

`sample_diseases.json` is a small, **hand-written, original** set of
illustrative disease entries (12 conditions loosely matching the ones
that appear as examples in the paper's Figure 1/Figure 3 case studies)
used so that the pipeline is runnable out of the box, in mock mode, and
in unit tests. **It is not copied or scraped from the MSD Manual and is
not a substitute for it** -- it is far too small and too coarse to be
clinically useful.

## Using the real MSD Manual (or another guideline corpus)

The paper grounds Dr.APP in the MSD Manual (professional + consumer
editions). The MSD Manual's content is copyrighted by Merck & Co. and
is **not redistributed in this repository**. To reproduce the paper's
grounding faithfully you must obtain the relevant content yourself
(e.g. via an institutional license, the publicly browsable MSD Manual
website subject to its terms of use, or another authoritative clinical
guideline source you are licensed to use), and convert it into the
schema below, one JSON object per disease/symptom entry:

```json
{
  "disease": "string, required",
  "specialty": "string, optional",
  "symptoms": ["list", "of", "associated", "symptom", "strings"],
  "causes": "free text, the 'Causes' section",
  "pathophysiology": "free text, 'Pathophysiology/Etiology' section",
  "diagnosis": "free text, the 'Diagnosis' section",
  "what_a_doctor_does": "free text, 'What a doctor does' section",
  "consumer_summary": "free text, simplified consumer-facing summary"
}
```

Save the resulting list of entries as a `.json` array (see
`sample_diseases.json` for the exact shape) and point
`paths.knowledge_base_dir` / the `--kb` CLI flag at your file. Then run:

```bash
python scripts/build_index.py --kb data/knowledge_base/your_file.json
```

to (re)build the RAG index used for symptom mapping and grounding
retrieval (`askpatients.rag`).
