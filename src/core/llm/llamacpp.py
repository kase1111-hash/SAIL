"""
SAIL llama.cpp LLM Provider

Integration with llama.cpp server for local LLM inference.
https://github.com/ggerganov/llama.cpp

llama.cpp server provides an OpenAI-compatible API endpoint.
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

DEFAULT_LLAMACPP_URL = "http://localhost:8080"


class LlamaCppProvider(LLMProvider):
    """
    llama.cpp server LLM provider implementation.

    llama.cpp server provides an OpenAI-compatible chat completions API.
    Start the server with: ./server -m model.gguf --port 8080
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = DEFAULT_LLAMACPP_URL,
        timeout: float = 60.0,
    ):
        """
        Initialize llama.cpp provider.

        Args:
            model: Model identifier (for logging, actual model is loaded by server)
            base_url: llama.cpp server URL
            timeout: Request timeout in seconds
        """
        super().__init__(model=model, base_url=base_url)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def provider_type(self) -> LLMProviderType:
        """Return the provider type."""
        return LLMProviderType.LLAMACPP

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

    def _messages_to_openai_format(self, messages: list[Message]) -> list[dict]:
        """Convert messages to OpenAI chat format."""
        return [msg.to_dict() for msg in messages]

    def _build_payload(
        self,
        messages: list[Message],
        config: GenerationConfig | None,
        stream: bool = False,
    ) -> dict:
        """Build request payload."""
        payload = {
            "messages": self._messages_to_openai_format(messages),
            "stream": stream,
        }

        if config:
            payload["max_tokens"] = config.max_tokens
            payload["temperature"] = config.temperature
            payload["top_p"] = config.top_p
            payload["top_k"] = config.top_k  # Added missing top_k parameter
            payload["repeat_penalty"] = config.repeat_penalty

            if config.stop_sequences:
                payload["stop"] = config.stop_sequences

        return payload

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """
        Generate a response using llama.cpp server.

        Args:
            messages: List of conversation messages
            config: Optional generation configuration

        Returns:
            GenerationResult with the model's response
        """
        client = await self._get_client()
        payload = self._build_payload(messages, config, stream=False)

        try:
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse OpenAI-compatible response
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})

            return GenerationResult(
                content=message.get("content", ""),
                model=data.get("model", self.model),
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Failed to connect to llama.cpp server at {self.base_url}: {e}"
            )
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"llama.cpp request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMGenerationError(
                f"llama.cpp server returned error: {e.response.status_code}"
            )
        except Exception as e:
            raise LLMGenerationError(f"Generation failed: {e}")

    async def stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a response from llama.cpp server.

        Args:
            messages: List of conversation messages
            config: Optional generation configuration

        Yields:
            StreamChunk objects as they arrive
        """
        client = await self._get_client()
        payload = self._build_payload(messages, config, stream=True)

        try:
            async with client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # SSE format: "data: {...}"
                    if line.startswith("data: "):
                        line = line[6:]

                    if line == "[DONE]":
                        yield StreamChunk(content="", done=True)
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choice.get("finish_reason")

                    yield StreamChunk(
                        content=content,
                        done=finish_reason is not None,
                        finish_reason=finish_reason,
                    )

        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Failed to connect to llama.cpp server at {self.base_url}: {e}"
            )
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"llama.cpp request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMGenerationError(
                f"llama.cpp server returned error: {e.response.status_code}"
            )
        except Exception as e:
            raise LLMGenerationError(f"Streaming failed: {e}")

    async def is_available(self) -> bool:
        """Check if llama.cpp server is available."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """
        List available models.

        Note: llama.cpp server only loads one model at a time,
        so this returns the currently loaded model.
        """
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                return [m.get("id", "") for m in models if m.get("id")]

            # Fallback: return configured model
            return [self.model]

        except Exception:
            return [self.model]

    async def get_server_props(self) -> dict:
        """
        Get llama.cpp server properties.

        Returns:
            Server configuration and status
        """
        try:
            client = await self._get_client()
            response = await client.get("/props")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise LLMConnectionError(f"Failed to get server properties: {e}")

    async def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text using the loaded model's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        try:
            client = await self._get_client()
            response = await client.post("/tokenize", json={"content": text})
            response.raise_for_status()
            data = response.json()
            return data.get("tokens", [])
        except Exception as e:
            raise LLMGenerationError(f"Tokenization failed: {e}")

    async def detokenize(self, tokens: list[int]) -> str:
        """
        Detokenize token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        try:
            client = await self._get_client()
            response = await client.post("/detokenize", json={"tokens": tokens})
            response.raise_for_status()
            data = response.json()
            return data.get("content", "")
        except Exception as e:
            raise LLMGenerationError(f"Detokenization failed: {e}")

    async def __aenter__(self) -> "LlamaCppProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
