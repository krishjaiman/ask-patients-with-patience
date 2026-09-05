"""Shared setup helpers used by every scripts/*.py entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running scripts/*.py directly (python scripts/run_app.py) without
# having pip-installed the package first.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from askpatients.config import Config  # noqa: E402
from askpatients.llm.factory import build_llm  # noqa: E402
from askpatients.llm.base import BaseLLM  # noqa: E402
from askpatients.medical.knowledge_base import KnowledgeBase  # noqa: E402
from askpatients.rag.embeddings import build_embedder  # noqa: E402
from askpatients.rag.retriever import Retriever  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                         help="Path to a YAML config overriding configs/default.yaml")
    parser.add_argument("--mock", action="store_true",
                         help="Force llm.provider=mock and rag.embedding_provider=mock "
                              "(no network / API key needed).")
    parser.add_argument("--kb", type=str, default=None,
                         help="Path to a knowledge-base JSON file "
                              "(default: data/knowledge_base/sample_diseases.json)")
    parser.add_argument("--log-level", type=str, default="INFO")


def load_config(args: argparse.Namespace) -> Config:
    overrides = {}
    if args.mock:
        overrides = {"llm": {"provider": "mock"}, "rag": {"embedding_provider": "mock"},
                      "patient": {"persona_model_provider": "mock"}}
    cfg = Config.load(args.config, overrides=overrides)
    cfg.apply_seed()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return cfg


def build_knowledge_base(cfg: Config, kb_path: str | None) -> KnowledgeBase:
    path = kb_path or str(Path(cfg.paths.knowledge_base_dir) / "sample_diseases.json")
    return KnowledgeBase.from_json_file(path)


def build_retriever(cfg: Config, kb: KnowledgeBase) -> Retriever:
    embedder = build_embedder(cfg.rag)
    return Retriever.build_from_knowledge_base(
        kb, embedder, chunk_size=cfg.rag.chunk_size, overlap=cfg.rag.chunk_overlap,
    )


def build_llm_from_config(cfg: Config) -> BaseLLM:
    return build_llm(cfg.llm)
