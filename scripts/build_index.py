#!/usr/bin/env python
"""
Builds (and persists) the RAG index over a knowledge-base JSON file:
  documents -> chunking -> embeddings -> FAISS -> (saved to disk)

Usage:
    python scripts/build_index.py --mock
    python scripts/build_index.py --kb data/knowledge_base/your_file.json \
        --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging

from _common import add_common_args, build_knowledge_base, build_retriever, load_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args)
    kb = build_knowledge_base(cfg, args.kb)
    logger.info("Loaded knowledge base with %d disease entries", len(kb.entries))
    retriever = build_retriever(cfg, kb)
    logger.info("Built retriever with %d chunks", len(retriever.chunks))
    if cfg.rag.embedding_provider == 'lexical':
        logger.info('Lexical retrieval needs no persisted embedding index.')
        return
    retriever.save(cfg.rag.index_path, cfg.rag.metadata_path)
    logger.info("Saved index to %s / metadata to %s", cfg.rag.index_path, cfg.rag.metadata_path)


if __name__ == "__main__":
    main()
