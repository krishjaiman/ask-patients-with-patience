"""Load the public ReMeDi-base release and the project's synthetic schema.

The official release uses dialogue/information/sentence/actions. Retain all doctor
labels. A unique Diagnosis/disease annotation is a provisional reference label.
The strict action filter does not recover the paper's 329 cases; the separately
named single_diagnosis policy is a documented approximation requiring review.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

VALID_ACTIONS = {"Informing", "Inquiring", "Chitchat", "QA", "Recommendation", "Diagnosis", "Others"}


@dataclass
class ReMeDiTurn:
    role: str            # "doctor" | "patient"
    text: str
    action: Optional[str] = None  # only present/meaningful for doctor turns
    actions: List[str] = field(default_factory=list)


@dataclass
class ReMeDiCase:
    case_id: str
    specialty: str
    ground_truth_diagnosis: str
    turns: List[ReMeDiTurn] = field(default_factory=list)
    diagnosis_candidates: List[str] = field(default_factory=list)
    source_format: str = 'project'

    def doctor_actions(self) -> List[str]:
        return [a for t in self.turns if t.role == 'doctor'
                for a in (t.actions or ([t.action] if t.action else []))]

    def opening_patient_turn(self) -> str:
        for t in self.turns:
            if t.role == "patient":
                return t.text
        return ""

    def opening_doctor_turn(self) -> str:
        """Use the same neutral opening in every method; never replay a diagnosis."""
        return "Hi, could you please tell me what's been bothering you today?"

    def full_transcript_text(self) -> str:
        lines = []
        for t in self.turns:
            speaker = "Doctor" if t.role == "doctor" else "Patient"
            lines.append(f"{speaker}: {t.text}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "specialty": self.specialty,
            "ground_truth_diagnosis": self.ground_truth_diagnosis,
            "turns": [t.__dict__ for t in self.turns],
            'diagnosis_candidates': self.diagnosis_candidates,
            'source_format': self.source_format,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReMeDiCase":
        return cls(
            case_id=d["case_id"], specialty=d.get("specialty", ""),
            ground_truth_diagnosis=d["ground_truth_diagnosis"],
            turns=[ReMeDiTurn(**t) for t in d.get("turns", [])],
            diagnosis_candidates=d.get('diagnosis_candidates', []),
            source_format=d.get('source_format', 'project'),
        )


def _coerce_case(raw: dict) -> ReMeDiCase:
    """
    Adapter from a raw ReMeDi-style JSON record to ReMeDiCase.

    ADAPT THIS FUNCTION if your ReMeDi export uses different field names.
    The expected raw shape (documented as our reconstruction, see module
    docstring) is:
        {
          "id" or "case_id": ...,
          "department" or "specialty": ...,
          "diagnosis" or "ground_truth_diagnosis": ...,
          "dialogue" or "turns": [
             {"role": "doctor"|"patient", "text"|"utterance": "...",
              "action" or "action_type": "Diagnosis"|... (doctor turns only)}
             ...
          ]
        }
    """
    if 'information' in raw:
        return _coerce_official_case(raw)
    case_id = str(raw.get("case_id", raw.get("id", "")))
    specialty = raw.get("specialty", raw.get("department", ""))
    diagnosis = raw.get("ground_truth_diagnosis", raw.get("diagnosis", ""))
    raw_turns = raw.get("turns", raw.get("dialogue", []))
    turns = []
    for rt in raw_turns:
        role = rt.get("role") or rt.get("speaker") or "patient"
        text = rt.get("text") or rt.get("utterance") or ""
        action = rt.get("action") or rt.get("action_type")
        turns.append(ReMeDiTurn(role=role, text=text, action=action, actions=rt.get('actions', [])))
    return ReMeDiCase(case_id=case_id, specialty=specialty,
                       ground_truth_diagnosis=diagnosis, turns=turns,
                       diagnosis_candidates=raw.get('diagnosis_candidates', []),
                       source_format=raw.get('source_format', 'project'))


def _coerce_official_case(raw: dict) -> ReMeDiCase:
    """Adapt the authors' public release, retaining all doctor action labels.

    A unique disease annotation is a provisional reference label, not a
    clinically adjudicated ground truth. Multiple labels require review.
    """
    aliases = {'Inform': 'Informing', 'Inquire': 'Inquiring',
               'Recommend': 'Recommendation', 'QuestionAnswering': 'QA', 'Other': 'Others'}
    turns, candidates = [], []
    for turn in raw['information']:
        labels = turn.get('actions', [])
        actions = [aliases.get(a.get('intent'), a.get('intent', 'Unknown')) for a in labels]
        turns.append(ReMeDiTurn(turn['role'], turn.get('sentence', ''), actions=actions))
        if turn['role'] == 'doctor':
            for label in labels:
                value = label.get('value1')
                if label.get('intent') == 'Diagnosis' and label.get('slot') == 'disease' and value:
                    if value.strip() and value.strip() not in candidates:
                        candidates.append(value.strip())
    return ReMeDiCase(str(raw.get('dialogue', raw.get('dialogue_id', ''))), '',
        candidates[0] if len(candidates) == 1 else '', turns,
        diagnosis_candidates=candidates, source_format='official-remedi-base')


def load_remedi_raw(path: str | Path) -> List[ReMeDiCase]:
    """Loads a JSONL (one case per line) or JSON-array file of raw ReMeDi cases."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ReMeDi raw data not found at {path}. See data/remedi/README.md for how "
            "to obtain the dataset from its original source; this repository does not "
            "redistribute it."
        )
    cases: List[ReMeDiCase] = []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                cases.append(_coerce_case(json.loads(line)))
    else:
        data = json.loads(text)
        for raw in data:
            cases.append(_coerce_case(raw))
    return cases


def filter_diagnosis_only(cases: List[ReMeDiCase], policy: str = 'strict') -> List[ReMeDiCase]:
    """
    Section 3.1: "We extract ... real-world conversations that exclusively
    contain the 'Diagnosis' label" among doctor-turn action annotations.
    """
    out = []
    if policy not in {'strict', 'single_diagnosis'}:
        raise ValueError('Unknown diagnosis filter policy')
    for c in cases:
        actions = c.doctor_actions()
        keep = bool(actions) and all(a == 'Diagnosis' for a in actions)
        if policy == 'single_diagnosis':
            keep = 'Diagnosis' in actions and bool(c.ground_truth_diagnosis)
        if keep:
            out.append(c)
    return out


def sample_eval_subset(cases: List[ReMeDiCase], n: int = 100, seed: int = 42) -> List[ReMeDiCase]:
    """Section 3.1: "randomly select 100 for our study."""
    rnd = random.Random(seed)
    pool = list(cases)
    rnd.shuffle(pool)
    return pool[:n]
