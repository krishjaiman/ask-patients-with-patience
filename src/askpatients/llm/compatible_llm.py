"""Chat-completions HTTP adapter for hosted free tiers or local servers.

Uses the Python standard library. No paid SDK subscription or GPU is needed.
Cache and request budget are shared across roles using the same endpoint/model.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import BaseLLM, LLMResponse, NonRetryableError

_USAGE = {}


class CompatibleLLM(BaseLLM):
    def __init__(self, base_url, api_key_env='LLM_API_KEY', min_request_interval=0,
                 max_api_calls=100, response_cache=None, reasoning_effort=None, **kwargs):
        super().__init__(**kwargs)
        if not base_url or urlparse(base_url).scheme not in {'http', 'https'}:
            raise ValueError('compatible provider requires an http(s) base_url')
        self.base_url = base_url.rstrip('/')
        self.api_key = os.environ.get(api_key_env, '')
        if urlparse(base_url).hostname not in {'localhost', '127.0.0.1', '::1'} and not self.api_key:
            raise ValueError(f'Set {api_key_env} in the local .env file')
        self.interval = max(0, min_request_interval)
        self.limit = max_api_calls
        self.cache_dir = Path(response_cache) if response_cache else None
        self.reasoning_effort = reasoning_effort
        self.usage = _USAGE.setdefault((self.base_url, self.model), {'calls': 0, 'last': 0.0})

    def _complete(self, messages):
        payload = {'model': self.model, 'temperature': self.temperature,
                   'max_tokens': self.max_tokens,
                   'messages': [{'role': m.role, 'content': m.content} for m in messages]}
        if self.reasoning_effort:
            payload['reasoning_effort'] = self.reasoning_effort
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        key = hashlib.sha256(self.base_url.encode() + body).hexdigest()
        cache = self.cache_dir / f'{key}.json' if self.cache_dir else None
        if cache and cache.exists():
            self.usage['cache_hits'] = self.usage.get('cache_hits', 0) + 1
            return LLMResponse(text=json.loads(cache.read_text(encoding='utf-8'))['text'], raw={'cached': True})
        if self.usage['calls'] >= self.limit:
            raise NonRetryableError(f'Local API request budget exhausted ({self.limit}); rerun to resume from cache')
        wait = self.interval - (time.monotonic() - self.usage['last'])
        if wait > 0:
            time.sleep(wait)
        self.usage['calls'] += 1
        self.usage['last'] = time.monotonic()
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json',
                   'User-Agent': 'AskPatients/0.1'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        request = Request(self.base_url + '/chat/completions', data=body, headers=headers)
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            # Do not print request headers, credentials or the provider body.
            if exc.code in {400, 401, 403, 404, 429}:
                raise NonRetryableError(
                    f'Provider HTTP {exc.code}. Check model/key or wait for the free quota to reset; '
                    'successful responses are cached. No paid fallback is used.') from None
            raise RuntimeError(f'Provider HTTP {exc.code}') from None
        choice = result['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise NonRetryableError('Model output was truncated; increase max_tokens or shorten the input')
        text = choice['message'].get('content') or ''
        if not text.strip():
            raise RuntimeError('Provider returned an empty completion')
        for token_type in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            count = (result.get('usage') or {}).get(token_type, 0)
            if isinstance(count, int):
                self.usage[token_type] = self.usage.get(token_type, 0) + count
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({'text': text}), encoding='utf-8')
        return LLMResponse(text=text, raw={'usage': result.get('usage'), 'cached': False})
