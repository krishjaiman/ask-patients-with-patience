"""
OpenAI chat-completions backend. Requires OPENAI_API_KEY in the environment
(see .env.example). Only imported/instantiated when config.llm.provider ==
"openai", so the `openai` package is an optional dependency.
"""
from __future__ import annotations

import os
from typing import List

from .base import BaseLLM, LLMMessage, LLMResponse


class OpenAILLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for llm.provider=openai. "
                "Install with `pip install openai`."
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export it in your shell."
            )
        self._client = OpenAI(api_key=api_key, timeout=self.request_timeout)

    def _complete(self, messages: List[LLMMessage]) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, raw=resp.model_dump() if hasattr(resp, "model_dump") else None)
