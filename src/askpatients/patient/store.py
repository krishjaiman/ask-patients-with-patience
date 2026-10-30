"""Reuse identical, source-keyed personas across methods and repeated runs."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .profile import PatientProfile, build_profile_from_case


def get_profile(llm, case, cache_dir='data/processed/profiles'):
    identity = {'version': 2, 'case': case.to_dict(), 'backend': type(llm).__name__,
                'model': llm.model, 'temperature': llm.temperature,
                'base_url': getattr(llm, 'base_url', None),
                'max_tokens': llm.max_tokens,
                'reasoning_effort': getattr(llm, 'reasoning_effort', None)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f'{key}.json'
    if path.exists():
        return PatientProfile(**json.loads(path.read_text(encoding='utf-8')))
    profile = build_profile_from_case(
        llm, case.case_id, case.full_transcript_text(), case.opening_patient_turn(),
        [], case.ground_truth_diagnosis)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding='utf-8')
    return profile
