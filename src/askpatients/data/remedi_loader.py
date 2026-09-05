"""
Section 3.1 "Dataset": ReMeDi (Yan et al., 2022) preprocessing.

  ReMeDi-base (1,557 labeled dialogues)
    -> keep only dialogues whose doctor turns are exclusively labeled
       "Diagnosis" among the seven action types
       {Informing, Inquiring, Chitchat, QA, Recommendation, Diagnosis, Others}
    -> 329 cases
    -> random subset of 100 for the study, across 18 specialties / 72 diseases

We do NOT bundle the ReMeDi dataset (obtain it from the original
authors / SIGIR'22 resource release; see data/remedi/README.md). This
module defines the case schema this codebase expects and the exact
filtering/sampling logic described above, operating on whatever raw
file the user supplies.

IMPLEMENTATION DECISION: the paper does not publish ReMeDi's raw JSON
schema in-text. We define a minimal, explicit `ReMeDiCase` schema
(id, specialty, ground_truth_diagnosis, presenting_complaint, turns
with per-turn action labels) that captures exactly the fields Sections
3.1-3.2 describe using, and provide `load_remedi_raw` as an adapter
point: if your actual ReMeDi export uses different field names, adapt
them in `_coerce_case` (clearly marked below) rather than elsewhere in
the pipeline.
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


@dataclass
class ReMeDiCase:
    case_id: str
    specialty: str
    ground_truth_diagnosis: str
    turns: List[ReMeDiTurn] = field(default_factory=list)

    def doctor_actions(self) -> List[str]:
        return [t.action for t in self.turns if t.role == "doctor" and t.action]

    def opening_patient_turn(self) -> str:
        for t in self.turns:
            if t.role == "patient":
                return t.text
        return ""

    def opening_doctor_turn(self) -> str:
        for t in self.turns:
            if t.role == "doctor":
                return t.text
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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReMeDiCase":
        return cls(
            case_id=d["case_id"], specialty=d.get("specialty", ""),
            ground_truth_diagnosis=d["ground_truth_diagnosis"],
            turns=[ReMeDiTurn(**t) for t in d.get("turns", [])],
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
    case_id = str(raw.get("case_id", raw.get("id", "")))
    specialty = raw.get("specialty", raw.get("department", ""))
    diagnosis = raw.get("ground_truth_diagnosis", raw.get("diagnosis", ""))
    raw_turns = raw.get("turns", raw.get("dialogue", []))
    turns = []
    for rt in raw_turns:
        role = rt.get("role") or rt.get("speaker") or "patient"
        text = rt.get("text") or rt.get("utterance") or ""
        action = rt.get("action") or rt.get("action_type")
        turns.append(ReMeDiTurn(role=role, text=text, action=action))
    return ReMeDiCase(case_id=case_id, specialty=specialty,
                       ground_truth_diagnosis=diagnosis, turns=turns)


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


def filter_diagnosis_only(cases: List[ReMeDiCase]) -> List[ReMeDiCase]:
    """
    Section 3.1: "We extract ... real-world conversations that exclusively
    contain the 'Diagnosis' label" among doctor-turn action annotations.
    """
    out = []
    for c in cases:
        actions = c.doctor_actions()
        if actions and all(a == "Diagnosis" for a in actions):
            out.append(c)
    return out


def sample_eval_subset(cases: List[ReMeDiCase], n: int = 100, seed: int = 42) -> List[ReMeDiCase]:
    """Section 3.1: "randomly select 100 for our study."""
    rnd = random.Random(seed)
    pool = list(cases)
    rnd.shuffle(pool)
    return pool[:n]
