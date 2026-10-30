"""
Abstract LLM interface used throughout AskPatients.

Every place in the pipeline that needs an LLM call (symptom mapping,
disease-probability estimation, question generation, response
simulation, conditional-probability estimation, human-centric
rewriting, patient simulation) goes through this interface. This keeps
the provider swappable via config and keeps the *actual* Bayesian
math (askpatients.bayesian) completely independent of any particular
LLM vendor.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class NonRetryableError(RuntimeError):
    """Configuration, budget or quota failure that retries cannot repair now."""


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    raw: Optional[dict] = None


class BaseLLM(ABC):
    """Provider-agnostic chat completion interface with structured-output helper."""

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 1024,
                 max_retries: int = 5, retry_base_delay: float = 1.5,
                 request_timeout: float = 60.0):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.request_timeout = request_timeout

    @abstractmethod
    def _complete(self, messages: List[LLMMessage]) -> LLMResponse:
        """Provider-specific single completion call (no retry logic)."""
        raise NotImplementedError

    def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        """Completion call wrapped with exponential-backoff retries."""
        import time

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._complete(messages)
            except NonRetryableError:
                raise
            except Exception as exc:  # noqa: BLE001 - broad on purpose, network calls
                last_exc = exc
                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, self.max_retries, exc, delay,
                )
                time.sleep(min(delay, 20.0))
        assert last_exc is not None
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_exc

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences and locate the outermost JSON object/array."""
        text = text.strip()
        text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()
        # Find first '{' or '[' and matching last '}' or ']'
        start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if not start_candidates:
            return text
        start = min(start_candidates)
        end_curly = text.rfind("}")
        end_square = text.rfind("]")
        end = max(end_curly, end_square)
        if end == -1 or end < start:
            return text
        return text[start:end + 1]

    def complete_structured(self, messages: List[LLMMessage], schema: Type[T],
                             retries: int = 2, mock_context: Optional[dict] = None,
                             validation_context: Optional[dict] = None) -> T:
        """
        Ask the LLM to produce JSON conforming to `schema`, parse and validate it.
        On parse/validation failure, retries with an explicit correction message.

        `mock_context` is ignored by real providers; MockLLM uses it to build
        deterministic, schema-conformant dummy data without having to parse
        the full natural-language prompt (see llm/mock_llm.py).
        """
        working_messages = list(messages)
        last_err: Optional[Exception] = None
        for _ in range(retries + 1):
            resp = self.complete(working_messages)
            json_text = self._extract_json(resp.text)
            try:
                data = json.loads(json_text)
                return schema.model_validate(data, context=validation_context)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = exc
                working_messages = working_messages + [
                    LLMMessage(role="assistant", content=resp.text),
                    LLMMessage(
                        role="user",
                        content=(
                            "Your previous output was not valid JSON matching the "
                            f"required schema. Error: {exc}. Respond again with ONLY "
                            "valid JSON, no prose, no markdown fences."
                        ),
                    ),
                ]
        raise RuntimeError(f"Failed to obtain structured output after retries: {last_err}")
