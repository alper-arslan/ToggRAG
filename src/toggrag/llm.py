from typing import Protocol

import httpx

from toggrag.config import Settings

_ANTHROPIC_URL = "https://api.anthropic.com"
_OLLAMA_URL = "http://localhost:11434"
_OPENAI_URL = "https://api.openai.com/v1"


class LLM(Protocol):
    """A chat model: one system prompt and one user message in, text out."""

    def complete(self, system: str, user: str) -> str: ...


class OllamaLLM:
    """Ollama's native chat endpoint.

    Used instead of its OpenAI-compatible one because only the native API lets
    each request set the context window (num_ctx). Ollama's default window is
    2048 tokens, and a longer prompt is cut at the start without any error.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        context_window: int,
        temperature: float,
        max_tokens: int,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._options = {
            "num_ctx": context_window,
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        self._client = client or httpx.Client(base_url=base_url, timeout=300)

    def complete(self, system: str, user: str) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "options": self._options,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAICompatibleLLM:
    """Chat completions endpoint shared by Ollama, OpenAI and many other servers."""

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url, headers=headers, timeout=300
        )

    def complete(self, system: str, user: str) -> str:
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicLLM:
    """Claude through the Messages API."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(
            base_url=_ANTHROPIC_URL,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=300,
        )

    def complete(self, system: str, user: str) -> str:
        response = self._client.post(
            "/v1/messages",
            json={
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        response.raise_for_status()
        return "".join(
            part["text"]
            for part in response.json()["content"]
            if part["type"] == "text"
        )


def make_llm(settings: Settings) -> LLM:
    """Build the LLM selected in config; API keys come from the environment."""
    config = settings.llm
    if config.provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(
            config.model,
            config.temperature,
            config.max_tokens,
            settings.anthropic_api_key.get_secret_value(),
        )
    if config.provider == "ollama":
        return OllamaLLM(
            config.model,
            config.base_url or _OLLAMA_URL,
            config.context_window,
            config.temperature,
            config.max_tokens,
        )
    key = settings.openai_api_key
    if key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAICompatibleLLM(
        config.model,
        config.base_url or _OPENAI_URL,
        config.temperature,
        config.max_tokens,
        key.get_secret_value(),
    )
