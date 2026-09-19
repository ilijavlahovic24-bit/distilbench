import hashlib
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

class TeacherError(Exception):
    """Generic teacher failure (non-retryable at the chain level)."""
class RateLimitError(TeacherError):
    """Provider signalled a rate limit (HTTP 429 / RESOURCE_EXHAUSTED)."""

@dataclass(frozen=True)
class TeacherResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached: bool = False


class TeacherCache:
    """SQLite-backed cache: (provider, model, system, prompt) -> response."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key        TEXT PRIMARY KEY,
                provider   TEXT NOT NULL,
                model      TEXT NOT NULL,
                prompt     TEXT NOT NULL,
                response   TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _key(provider: str, model: str, prompt: str, system: str | None) -> str:
        h = hashlib.sha256()
        for part in (provider, model, system or "", prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(
        self, provider: str, model: str, prompt: str, system: str | None
    ) -> TeacherResponse | None:
        key = self._key(provider, model, prompt, system)
        row = self._conn.execute(
            "SELECT response FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return TeacherResponse(**data, cached=True)

    def put(
        self,
        provider: str,
        model: str,
        prompt: str,
        system: str | None,
        response: TeacherResponse,
    ) -> None:
        key = self._key(provider, model, prompt, system)
        payload = json.dumps(
            {
                "text": response.text,
                "provider": response.provider,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            }
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO responses "
            "(key, provider, model, prompt, response, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, provider, model, prompt, payload, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

class TeacherClient(ABC):
    provider: str
    model: str

    @abstractmethod
    def _call(
        self, prompt: str, system: str | None, **kwargs: Any
    ) -> TeacherResponse: ...

    def generate(
        self, prompt: str, system: str | None = None, **kwargs: Any
    ) -> TeacherResponse:
        return self._call(prompt, system, **kwargs)


class OpenAITeacherClient(TeacherClient):
    """OpenAI-compatible client (used for Groq and Cerebras)."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
    ):
        from openai import OpenAI  # local import keeps import-time deps minimal

        self.provider = provider
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _call(
        self, prompt: str, system: str | None, **kwargs: Any
    ) -> TeacherResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        except Exception as exc:  # SDK-specific types vary; string match is safest
            msg = str(exc)
            if "429" in msg or "rate" in msg.lower():
                raise RateLimitError(msg) from exc
            raise TeacherError(msg) from exc

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return TeacherResponse(
            text=choice.message.content or "",
            provider=self.provider,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )


class GeminiTeacherClient(TeacherClient):
    """Google Gemini client (not OpenAI-compatible; uses google-genai SDK)."""

    def __init__(self, model: str, api_key: str):
        from google import genai

        self.provider = "gemini"
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def _call(
        self, prompt: str, system: str | None, **kwargs: Any
    ) -> TeacherResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=kwargs.get("temperature", 0.2),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                raise RateLimitError(msg) from exc
            raise TeacherError(msg) from exc

        return TeacherResponse(
            text=resp.text or "",
            provider=self.provider,
            model=self.model,
        )

class TeacherChain:
    """Try providers in order; fall back on rate-limit and transient errors."""

    def __init__(
        self,
        clients: list[TeacherClient],
        cache: TeacherCache | None = None,
        max_retries_per_client: int = 2,
    ):
        if not clients:
            raise ValueError("TeacherChain requires at least one client")
        self.clients = clients
        self.cache = cache
        self.max_retries_per_client = max_retries_per_client

    def generate(
        self, prompt: str, system: str | None = None, **kwargs: Any
    ) -> TeacherResponse:
        # 1) Cache probe across all providers (so we don't regenerate
        #    something we already have, regardless of which provider made it).
        if self.cache is not None:
            for c in self.clients:
                hit = self.cache.get(c.provider, c.model, prompt, system)
                if hit is not None:
                    LOG.debug("cache hit via %s/%s", c.provider, c.model)
                    return hit

        errors: list[tuple[str, str, str]] = []
        for client in self.clients:
            for attempt in range(self.max_retries_per_client):
                try:
                    resp = client.generate(prompt, system=system, **kwargs)
                    if self.cache is not None:
                        self.cache.put(
                            client.provider, client.model, prompt, system, resp
                        )
                    return resp
                except RateLimitError as exc:
                    LOG.warning(
                        "%s/%s rate-limited, falling through: %s",
                        client.provider, client.model, exc,
                    )
                    errors.append((client.provider, "rate-limit", str(exc)))
                    break  # move to next provider immediately
                except TeacherError as exc:
                    LOG.warning(
                        "%s/%s error (attempt %d/%d): %s",
                        client.provider, client.model, attempt + 1,
                        self.max_retries_per_client, exc,
                    )
                    errors.append((client.provider, "error", str(exc)))
                    if attempt + 1 < self.max_retries_per_client:
                        time.sleep(min(2 ** attempt, 8))

        raise TeacherError(f"All providers failed. Errors: {errors}")
def build_default_chain(
    cache_path: Path | str = "data/cache/teacher.sqlite",
    max_retries_per_client: int = 2,
) -> TeacherChain:
    """Construct the standard Groq -> Gemini -> Cerebras fallback chain.

    Providers are added only if their API key is present in the environment,
    so a partial setup still works (e.g. only GROQ_API_KEY set).
    """
    cache = TeacherCache(Path(cache_path))
    clients: list[TeacherClient] = []

    if groq_key := os.getenv("GROQ_API_KEY"):
        clients.append(
            OpenAITeacherClient(
                provider="groq",
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
        )

    if gemini_key := (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        clients.append(
            GeminiTeacherClient(model="gemini-2.5-flash", api_key=gemini_key)
        )

    if cerebras_key := os.getenv("CEREBRAS_API_KEY"):
        clients.append(
            OpenAITeacherClient(
                provider="cerebras",
                model="llama-3.3-70b",
                api_key=cerebras_key,
                base_url="https://api.cerebras.ai/v1",
            )
        )

    if not clients:
        raise TeacherError(
            "No teacher API keys found. Set at least one of: "
            "GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY."
        )

    return TeacherChain(
        clients=clients,
        cache=cache,
        max_retries_per_client=max_retries_per_client,
    )