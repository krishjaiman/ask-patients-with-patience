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
    if cfg.provider == 'compatible':
        from .compatible_llm import CompatibleLLM
        return CompatibleLLM(base_url=cfg.base_url, api_key_env=cfg.api_key_env,
            min_request_interval=cfg.min_request_interval, max_api_calls=cfg.max_api_calls,
            response_cache=cfg.response_cache, reasoning_effort=cfg.reasoning_effort, **kwargs)
    if cfg.provider == "openai":
        from .openai_llm import OpenAILLM
        return OpenAILLM(**kwargs)
    if cfg.provider == "anthropic":
        from .anthropic_llm import AnthropicLLM
        return AnthropicLLM(**kwargs)
    raise ValueError(f"Unknown llm.provider: {cfg.provider!r}")
