"""
Groq backend for the playground: same interface as the local backends, but the
tokens come from Groq's OpenAI-compatible API instead of this machine.

Groq serves open models (Llama, Qwen, gpt-oss, ...) on their own hardware, so
this exists for the case where you want good answers on a phone and are happy to
be online: run

    export GROQ_API_KEY=...          # never commit this
    python serve.py --backend groq   # then open the page from your phone

Configuration, all from the environment:

    GROQ_API_KEY    required; the request fails with a clear message without it
    GROQ_MODEL      default "openai/gpt-oss-120b"; any id from
                    `curl -H "Authorization: Bearer $GROQ_API_KEY" \
                          https://api.groq.com/openai/v1/models`
    GROQ_BASE_URL   default "https://api.groq.com/openai/v1" (override for a
                    proxy, or for tests against a local mock)

Deliberately explicit: `--backend auto` never picks this, so a machine that has
local models keeps working offline and never silently starts billing an API.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BASE = "https://api.groq.com/openai/v1"
# Groq retired llama-3.1-8b-instant and llama-3.3-70b-versatile on 2026-08-16;
# gpt-oss-120b is the replacement Groq points people at.
DEFAULT_MODEL = "openai/gpt-oss-120b"


class MissingKey(RuntimeError):
    """GROQ_API_KEY is not set -- the caller should explain how to set it."""


class GroqBackend:
    name = "groq"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None):
        self.base = (base_url or os.environ.get("GROQ_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        self.last_stats: dict = {}

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def key_hint() -> str:
        return ("GROQ_API_KEY is not set. Get a key at https://console.groq.com/keys "
                "and set it where the server runs:\n"
                "    export GROQ_API_KEY=gsk_...\n"
                "or put it in a .env file next to the repo (gitignored).")

    def _require_key(self) -> str:
        if not self.api_key:
            raise MissingKey(self.key_hint())
        return self.api_key

    def require_key(self) -> str:
        """Public form, so start-up can refuse to boot without a key."""
        return self._require_key()

    def info(self) -> dict:
        return {
            "backend": "groq",
            "model": self.model,
            "endpoint": self.base,
            "key": "set" if self.api_key else "missing",
            "checkpoint": "-",
            "params": "cloud",
            "context": "provider-defined",
            "note": f"Groq API ({self.model})" if self.api_key
                    else f"set $GROQ_API_KEY to use {self.model}",
        }

    # -- generation ------------------------------------------------------- #
    def stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
               top_k: int, seed: int):
        """Yield text deltas from Groq's streaming chat endpoint."""
        key = self._require_key()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
        )
        emitted = 0
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        # Groq reports stream errors as a top-level "error"
                        err = chunk.get("error")
                        if err:
                            raise RuntimeError(str(err))
                        continue
                    text = choices[0].get("delta", {}).get("content") or ""
                    if text:
                        emitted += 1
                        yield text
        except urllib.error.HTTPError as exc:  # surface the provider's reason
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"Groq returned HTTP {exc.code}: {detail}") from exc
        self.last_stats = {"tokens": emitted}

    def stop(self) -> None:
        return
