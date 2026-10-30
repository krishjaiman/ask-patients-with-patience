"""
Knowledge base interface for the "grounded" part of Dr.APP.

The paper grounds every stage of reasoning (disease-probability
generation, question generation, response-likelihood estimation) in
sections retrieved from the MSD Manual -- specifically the *causes*,
*pathophysiology/etiology*, *diagnosis*, and *what a doctor does*
sections of each symptom/disease page (Sections 2.2-2.4).

We CANNOT redistribute the MSD Manual itself (it is copyrighted and not
freely redistributable). Instead this module defines:

  1. A generic `DiseaseEntry` schema mirroring the sections the paper
     actually retrieves and uses (causes, diagnosis criteria/what a
     doctor does, associated symptoms).
  2. A `KnowledgeBase` loader that reads any JSON/JSONL file matching
     that schema from disk -- so a user who has legally obtained MSD
     Manual text (or any other clinical guideline corpus, e.g. their
     institution's licensed content) can plug it in directly.
  3. A small, clearly-labeled SAMPLE knowledge base
     (data/knowledge_base/sample_diseases.json) containing a handful of
     short, original, hand-written illustrative entries (NOT copied
     from the MSD Manual) so the pipeline is runnable out of the box in
     mock mode and for tests.

IMPLEMENTATION DECISION: the paper does not fully specify the on-disk
schema of its MSD ingestion, so this schema is our reconstruction of
the fields the paper's Sections 2.2-2.4 describe using.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DiseaseEntry:
    disease: str
    specialty: str = ""
    symptoms: List[str] = field(default_factory=list)          # associated symptom vocabulary
    causes: str = ""                                            # Gamma(S_MSD) / Gamma(d_i) "causes" text
    pathophysiology: str = ""                                   # Gamma(S_MSD) "pathophysiology/etiology"
    diagnosis: str = ""                                         # Upsilon(S_MSD) "Diagnosis" section
    what_a_doctor_does: str = ""                                # Upsilon(S_MSD) "What a doctor does"
    consumer_summary: str = ""                                  # simplified/consumer-facing text

    source_url: str = ""
    source_title: str = ""
    source_section: str = ""

    def as_grounding_text(self) -> str:
        """Concatenated text used as retrieval-chunk content and as
        Gamma(d_i)/Upsilon(d_i) context passed into LLM prompts."""
        parts = [
            f"Disease: {self.disease}",
            f"Source: {self.source_title} {self.source_url} ({self.source_section})" if self.source_url else "",
            f"Specialty: {self.specialty}" if self.specialty else "",
            f"Associated symptoms: {', '.join(self.symptoms)}" if self.symptoms else "",
            f"Causes: {self.causes}" if self.causes else "",
            f"Pathophysiology/Etiology: {self.pathophysiology}" if self.pathophysiology else "",
            f"Diagnosis: {self.diagnosis}" if self.diagnosis else "",
            f"What a doctor does: {self.what_a_doctor_does}" if self.what_a_doctor_does else "",
            f"Patient-friendly summary: {self.consumer_summary}" if self.consumer_summary else "",
        ]
        return "\n".join(p for p in parts if p)


class KnowledgeBase:
    """In-memory clinical knowledge base loaded from a JSON file of DiseaseEntry records."""

    def __init__(self, entries: List[DiseaseEntry]):
        self.entries: List[DiseaseEntry] = entries
        self._by_name: Dict[str, DiseaseEntry] = {e.disease: e for e in entries}

    @classmethod
    def from_json_file(cls, path: str | Path) -> "KnowledgeBase":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge base file not found: {path}. See data/knowledge_base/README.md "
                "for how to supply your own legally-obtained clinical guideline corpus."
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        entries = [DiseaseEntry(**item) for item in raw]
        return cls(entries)

    def get(self, disease: str) -> Optional[DiseaseEntry]:
        return self._by_name.get(disease)

    def all_disease_names(self) -> List[str]:
        return [e.disease for e in self.entries]

    def symptom_vocabulary(self) -> List[str]:
        seen: Dict[str, None] = {}
        for e in self.entries:
            for s in e.symptoms:
                seen.setdefault(s, None)
        return list(seen.keys())

    def diseases_for_symptom(self, symptom: str) -> List[DiseaseEntry]:
        symptom_l = symptom.lower()
        return [e for e in self.entries if any(symptom_l in s.lower() or s.lower() in symptom_l
                                                 for s in e.symptoms)]
