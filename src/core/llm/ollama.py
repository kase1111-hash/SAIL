"""
SAIL Ollama LLM Provider

Integration with Ollama for local LLM inference.
https://ollama.ai/
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from .base import (
    GenerationConfig,
    GenerationResult,
    LLMConnectionError,
    LLMGenerationError,
    LLMProvider,
    LLMProviderType,
    Message,
    StreamChunk,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider implementation.

    Ollama provides a simple API for running large language models locally.
    This provider supports both synchronous and streaming generation.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama3", "mistral", "codellama")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        super().__init__(model=model, base_url=base_url)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def provider_type(self) -> LLMProviderType:
        """Return the provider type."""
        return LLMProviderType.OLLAMA

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client (thread-safe)."""
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.timeout, connect=10.0),
                )
            return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_options(self, config: GenerationConfig | None) -> dict:
        """Build Ollama options from generation config."""
        if config is None:
            return {}

        options = {
            "num_predict": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repeat_penalty": config.repeat_penalty,
        }

        if config.stop_sequences:
            options["stop"] = config.stop_sequences

        return options

    def _messages_to_ollama_format(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Ollama chat format."""
        return [msg.to_dict() for msg in messages]

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """
        Generate a response using Ollama.

        Args:
            messages: List of conversation messages
            config: Optional generation configuration

        Returns:
            GenerationResult with the model's response
        """
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": self._messages_to_ollama_format(messages),
            "stream": False,
            "options": self._build_options(config),
        }

        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            return GenerationResult(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                finish_reason=data.get("done_reason"),
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_tokens=(
                    (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
                    if data.get("prompt_eval_count") or data.get("eval_count")
                    else None
                ),
            )

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Ollama request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMGenerationError(f"Ollama returned error: {e.response.status_code}")
        except Exception as e:
            raise LLMGenerationError(f"Generation failed: {e}")

    async def stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a response from Ollama.

        Args:
            messages: List of conversation messages
            config: Optional generation configuration

        Yields:
            StreamChunk objects as they arrive
        """
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": self._messages_to_ollama_format(messages),
            "stream": True,
            "options": self._build_options(config),
        }

        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)

                    yield StreamChunk(
                        content=content,
                        done=done,
                        finish_reason=data.get("done_reason") if done else None,
                    )

                    if done:
                        break

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Ollama request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMGenerationError(f"Ollama returned error: {e.response.status_code}")
        except Exception as e:
            raise LLMGenerationError(f"Streaming failed: {e}")

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]

        except Exception as e:
            raise LLMConnectionError(f"Failed to list models: {e}")

    async def pull_model(self, model_name: str | None = None) -> AsyncIterator[dict]:
        """
        Pull a model from the Ollama library.

        Args:
            model_name: Model to pull (defaults to self.model)

        Yields:
            Progress updates as dictionaries
        """
        client = await self._get_client()
        model = model_name or self.model

        try:
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model},
                timeout=httpx.Timeout(600.0, connect=10.0),  # Long timeout for downloads
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            raise LLMConnectionError(f"Failed to pull model {model}: {e}")

    async def model_info(self, model_name: str | None = None) -> dict:
        """
        Get information about a model.

        Args:
            model_name: Model name (defaults to self.model)

        Returns:
            Model information dictionary
        """
        client = await self._get_client()
        model = model_name or self.model

        try:
            response = await client.post("/api/show", json={"name": model})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise LLMConnectionError(f"Failed to get model info for {model}: {e}")

    async def __aenter__(self) -> "OllamaProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
