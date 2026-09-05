"""
Anthropic Messages API backend. Requires ANTHROPIC_API_KEY in the
environment (see .env.example). Only imported/instantiated when
config.llm.provider == "anthropic".
"""
from __future__ import annotations

import os
from typing import List

from .base import BaseLLM, LLMMessage, LLMResponse


class AnthropicLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for llm.provider=anthropic. "
                "Install with `pip install anthropic`."
            ) from exc
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export it in your shell."
            )
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.request_timeout)

    def _complete(self, messages: List[LLMMessage]) -> LLMResponse:
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        chat_msgs = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=chat_msgs,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return LLMResponse(text=text, raw=None)
