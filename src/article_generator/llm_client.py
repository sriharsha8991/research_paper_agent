"""LLM client wrapper with retry, caching, and model routing."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from article_generator.config import AppConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM interface with semantic caching and retry logic.

    Implements:
    - Model routing (§5.3 MoA)
    - Semantic cache (§10.2)
    - Retry with exponential backoff (§10.3)
    - Token budget awareness (§4.2)
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client: genai.Client | None = None
        self._cache_dir = config.paths.cache_dir / "llm_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._call_count = 0
        self._cached_count = 0

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.config.google_api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            self._client = genai.Client(api_key=self.config.google_api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        use_cache: bool = True,
    ) -> str:
        """Generate text from an LLM.

        Args:
            prompt: The user prompt.
            system: System prompt for agent instructions.
            model: Override model selection. Defaults to primary model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            use_cache: Whether to check the disk cache first.

        Returns:
            Generated text string.
        """
        model = model or self.config.models.primary

        # Check cache
        if use_cache:
            cache_key = self._cache_key(prompt, system, model, temperature)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._cached_count += 1
                logger.debug("Cache hit for LLM call (total cached: %d)", self._cached_count)
                return cached

        # Call LLM with retry
        self._call_count += 1
        response_text = self._call_with_retry(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Cache the response
        if use_cache:
            self._cache_set(cache_key, response_text)

        return response_text

    def generate_reasoning(self, prompt: str, *, system: str = "", **kwargs) -> str:
        """Use the reasoning model for deep analysis tasks."""
        return self.generate(
            prompt,
            system=system,
            model=self.config.models.reasoning,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 8192),
        )

    def generate_fast(self, prompt: str, *, system: str = "", **kwargs) -> str:
        """Use the fast model for metadata/formatting tasks."""
        return self.generate(
            prompt,
            system=system,
            model=self.config.models.fast,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
        )

    def _call_with_retry(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int = 3,
    ) -> str:
        """Call the Gemini API with exponential backoff."""
        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                if system:
                    config.system_instruction = system

                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                text = response.text

                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                logger.info(
                    "LLM call #%d: model=%s, input_tokens=%d, output_tokens=%d",
                    self._call_count,
                    model,
                    input_tokens,
                    output_tokens,
                )
                return text

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error("LLM call failed after %d retries: %s", max_retries, e)
                    raise
                wait = (2**attempt) + (time.time() % 1)
                logger.warning(
                    "LLM call attempt %d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    e,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError("Unreachable")

    # ── Cache helpers ────────────────────────────────────────────────

    def _cache_key(
        self, prompt: str, system: str, model: str, temperature: float
    ) -> str:
        content = f"{model}|{temperature}|{system}|{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("response")
        return None

    def _cache_set(self, key: str, response: str) -> None:
        path = self._cache_dir / f"{key}.json"
        path.write_text(
            json.dumps({"response": response}, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_calls": self._call_count,
            "cached_calls": self._cached_count,
        }
