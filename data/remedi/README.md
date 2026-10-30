# ReMeDi data and provenance

The authors publicly release ReMeDi-base under their repository's MIT license statement:
https://github.com/yanguojun123/Medical-Dialogue

Verified and downloaded on 2026-10-06:
- data/ReMeDi-base.json: https://github.com/yanguojun123/Medical-Dialogue/blob/main/data/ReMeDi-base.json
- License: https://github.com/yanguojun123/Medical-Dialogue/blob/main/MIT-license.txt

The raw JSON and license are stored locally under `raw/`, excluded from Git. The release contains 1,557 Chinese conversations. Downloaded bytes are fingerprinted in the preprocessing audit output.

The official adapter supports `dialogue`, `information`, `sentence`, and each turn's `actions`. All doctor intents are retained and normalized. One distinct annotated disease becomes a provisional reference label; ambiguous/missing diagnoses are not guessed. Specialties are not present in the inspected records and are left empty.

The strict reading of APP's filter returns zero records on this release, not 329. A separate `single_diagnosis` policy yields 280 candidates (386 contain a Diagnosis action). Do not present the generated 100-case provisional subset as the authors' exact benchmark. It still requires label review, language handling and guideline coverage.

Run the command documented in the root README to regenerate data and the audit JSON. The eight cases in `synthetic_sample.jsonl` are original English examples, not ReMeDi data.
