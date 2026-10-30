"""
Central configuration for the AskPatients reproduction.

All hyperparameters called out in the paper's Experimental Setup
(Section 3.3) are exposed here and are overridable via a YAML config
file (see configs/default.yaml) or CLI flags in the scripts/ entry
points. Nothing that affects the paper's core algorithm is hard-coded
deeper in the pipeline -- it all flows through this object so
experiments are reproducible and auditable.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LLMConfig:
    # "openai", "anthropic", or "mock". Mock requires no network/API key
    # and is fully deterministic -- used for tests and offline demos.
    provider: str = "mock"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024
    max_retries: int = 5
    retry_base_delay: float = 1.5
    request_timeout: float = 60.0
    base_url: Optional[str] = None
    api_key_env: str = 'LLM_API_KEY'
    min_request_interval: float = 0.0
    max_api_calls: int = 100
    response_cache: Optional[str] = None
    reasoning_effort: Optional[str] = None


@dataclass
class RAGConfig:
    embedding_provider: str = "mock"  # "openai" | "sentence-transformers" | "mock"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 400          # characters per chunk
    chunk_overlap: int = 60
    top_k: int = 5
    index_path: str = "data/processed/faiss.index"
    metadata_path: str = "data/processed/chunks.jsonl"


@dataclass
class AppAlgoConfig:
    """
    Hyperparameters from Section 3.3 "Experimental Setup":
      - K = 5 candidate questions per iteration
      - 2 <= L <= 5 plausible patient responses per candidate question
      - multi-turn setting uses six rounds of iterative questioning
    """
    num_candidate_questions: int = 5          # K
    min_responses_per_question: int = 2       # L lower bound
    max_responses_per_question: int = 5       # L upper bound
    max_dialogue_turns: int = 6               # iterative questioning rounds
    top_n_diseases_for_questions: int = 8      # diseases considered when building Q_{t+1}
    min_prob_floor: float = 1e-6              # numerical floor to avoid log(0) / div0
    belief_update: str = "dialogue"  # dialogue | bayesian (legacy approximation)

    def __post_init__(self):
        if self.belief_update not in {'dialogue', 'bayesian'}:
            raise ValueError('belief_update must be dialogue or bayesian')
        if not 2 <= self.min_responses_per_question <= self.max_responses_per_question <= 5:
            raise ValueError('Require 2 <= min_responses <= max_responses <= 5')
        if self.num_candidate_questions < 1 or self.max_dialogue_turns < 0 or self.top_n_diseases_for_questions < 1:
            raise ValueError('Invalid question, turn, or candidate disease count')


@dataclass
class EvalConfig:
    dataset_diagnosis_only_size: int = 329   # cases with only "Diagnosis" label
    eval_subset_size: int = 100              # randomly selected for study
    seed: int = 42
    accuracy_scale_max: int = 5
    output_dir: str = "outputs"


@dataclass
class PathsConfig:
    remedi_raw: str = "data/remedi/raw"
    remedi_processed: str = "data/processed/remedi_diagnosis_only.jsonl"
    remedi_eval_subset: str = "data/processed/remedi_eval_100.jsonl"
    knowledge_base_dir: str = "data/knowledge_base"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    profile_llm: Optional[LLMConfig] = None
    patient_llm: Optional[LLMConfig] = None
    judge_llm: Optional[LLMConfig] = None
    rag: RAGConfig = field(default_factory=RAGConfig)
    algo: AppAlgoConfig = field(default_factory=AppAlgoConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 42

    def apply_seed(self) -> None:
        random.seed(self.seed)
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass

    @staticmethod
    def _merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Config._merge(out[k], v)
            else:
                out[k] = v
        return out

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[dict] = None) -> "Config":
        base = asdict(cls())
        if path and not Path(path).exists():
            raise FileNotFoundError(f'Configuration not found: {path}')
        if path:
            with open(path, "r") as f:
                file_cfg = yaml.safe_load(f) or {}
            base = cls._merge(base, file_cfg)
        if overrides:
            base = cls._merge(base, overrides)
        unknown = set(base) - set(asdict(cls()))
        if unknown:
            raise ValueError(f'Unknown configuration sections: {sorted(unknown)}. Use profile_llm/patient_llm/judge_llm for model roles.')
        return cls(
            llm=LLMConfig(**base["llm"]),
            profile_llm=LLMConfig(**base['profile_llm']) if base['profile_llm'] else None,
            patient_llm=LLMConfig(**base['patient_llm']) if base['patient_llm'] else None,
            judge_llm=LLMConfig(**base['judge_llm']) if base['judge_llm'] else None,
            rag=RAGConfig(**base["rag"]),
            algo=AppAlgoConfig(**base["algo"]),
            eval=EvalConfig(**base["eval"]),
            paths=PathsConfig(**base["paths"]),
            seed=base["seed"],
        )


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
