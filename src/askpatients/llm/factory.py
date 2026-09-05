from __future__ import annotations

from askpatients.config import LLMConfig
from .base import BaseLLM


def build_llm(cfg: LLMConfig) -> BaseLLM:
    kwargs = dict(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        retry_base_delay=cfg.retry_base_delay,
        request_timeout=cfg.request_timeout,
    )
    if cfg.provider == "mock":
        from .mock_llm import MockLLM
        return MockLLM(**kwargs)
    if cfg.provider == "openai":
        from .openai_llm import OpenAILLM
        return OpenAILLM(**kwargs)
    if cfg.provider == "anthropic":
        from .anthropic_llm import AnthropicLLM
        return AnthropicLLM(**kwargs)
    raise ValueError(f"Unknown llm.provider: {cfg.provider!r}")
