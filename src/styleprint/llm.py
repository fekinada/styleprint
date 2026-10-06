"""Minimal client for OpenAI-compatible chat APIs: OpenAI itself, or a local server
(llama.cpp, vLLM, Ollama, LM Studio...).

Settings come from `styleprint.toml` in the working directory (see
styleprint.example.toml), overridable by environment variables and then by
explicit arguments:

    [llm]
    url = "https://api.openai.com"    # or "http://localhost:8080" for a local server
    model = "gpt-4.1"                 # required for OpenAI; empty = whatever a local server serves
    api_key_env = "OPENAI_API_KEY"    # name of the env var holding the key (never the key itself)
    kind = "auto"                     # "openai", "local", or "auto" (detect from the url)
    thinking = false                  # local Qwen3-style models: reason before answering
"""

from __future__ import annotations

import json
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class LLMError(Exception):
    pass


# Parameters a server may reject; dropped and retried once if it does.
OPTIONAL_PARAMS = ("temperature", "chat_template_kwargs", "max_tokens", "max_completion_tokens")


@dataclass
class LLMConfig:
    url: str = "http://localhost:8080"
    model: str = ""
    api_key_env: str = ""
    kind: str = "auto"
    thinking: bool = False
    timeout_s: float = 1800
    max_tokens: int = 4096
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = "styleprint.toml", **overrides) -> "LLMConfig":
        data: dict = {}
        p = Path(path)
        if p.exists():
            data = tomllib.loads(p.read_text()).get("llm", {})
        env = {"url": os.environ.get("STYLEPRINT_LLM_URL"), "model": os.environ.get("STYLEPRINT_LLM_MODEL")}
        data.update({k: v for k, v in env.items() if v})
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)

    @property
    def is_openai(self) -> bool:
        if self.kind != "auto":
            return self.kind == "openai"
        return (urlparse(self.url).hostname or "").endswith("openai.com")

    @property
    def api_key(self) -> str | None:
        """The key comes from an env var, and only one named for this server (or STYLEPRINT_LLM_API_KEY):
        a key exported for OpenAI is never sent to a local server by accident."""
        if os.environ.get("STYLEPRINT_LLM_API_KEY"):
            return os.environ["STYLEPRINT_LLM_API_KEY"]
        name = self.api_key_env or ("OPENAI_API_KEY" if self.is_openai else "")
        return os.environ.get(name) if name else None


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.url = re.sub(r"/v1/?$", "", config.url.rstrip("/"))
        self._model = config.model or None
        self._dropped: set[str] = set()
        if config.is_openai and not config.api_key:
            raise LLMError(f"no API key for {self.url}: export {config.api_key_env or 'OPENAI_API_KEY'}")

    def _request(self, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:800]
            raise LLMError(f"{self.url}{path}: HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"cannot reach {self.url}{path}: {e}") from None

    @property
    def model(self) -> str:
        if self._model is None:
            if self.config.is_openai:
                raise LLMError("set `model` in styleprint.toml (or --model) to use OpenAI")
            models = self._request("/v1/models").get("data", [])
            if not models:
                raise LLMError(f"{self.url} lists no models")
            self._model = models[0]["id"]
        return self._model

    def _body(self, messages: list[dict], temperature: float, response_format: dict | None) -> dict:
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.config.is_openai:
            body["max_completion_tokens"] = self.config.max_tokens
        else:
            body["max_tokens"] = self.config.max_tokens
            body["chat_template_kwargs"] = {"enable_thinking": self.config.thinking}
        if response_format:
            body["response_format"] = response_format
        body.update(self.config.extra)
        return {k: v for k, v in body.items() if k not in self._dropped}

    def _complete(self, body: dict) -> str:
        """POST a completion; if the server rejects an optional parameter, drop it (for good) and retry."""
        while True:
            try:
                resp = self._request("/v1/chat/completions", body)
                break
            except LLMError as e:
                msg = str(e)
                named = lambda p: f"'{p}'" in msg or re.search(rf'"param":\s*"{p}"', msg)  # noqa: E731
                bad = next((p for p in OPTIONAL_PARAMS if p in body and named(p)), None) if "HTTP 400" in msg else None
                if bad is None:
                    raise
                self._dropped.add(bad)
                body = {k: v for k, v in body.items() if k != bad}
        content = resp["choices"][0]["message"].get("content") or ""
        return re.sub(r"(?s)<think>.*?</think>", "", content).strip()

    def chat(self, messages: list[dict], temperature: float = 0.8, retries: int = 2) -> str:
        """One chat completion; returns the answer text (reasoning, if any, is dropped)."""
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                content = self._complete(self._body(messages, temperature, None))
                if content:
                    return content
                last = LLMError("empty answer")
            except (LLMError, KeyError) as e:
                last = e
            time.sleep(2 * (attempt + 1))
        raise LLMError(f"no answer after {retries + 1} attempts: {last}")

    def json(self, prompt: str, schema: dict, retries: int = 2) -> dict:
        """One chat turn constrained to a JSON schema; returns the parsed object."""
        fmt = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": False}}
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return json.loads(self._complete(self._body([{"role": "user", "content": prompt}], 0.2, fmt)))
            except (LLMError, json.JSONDecodeError, KeyError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise LLMError(f"no valid JSON after {retries + 1} attempts: {last}")
