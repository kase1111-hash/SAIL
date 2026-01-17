"""
SAIL LLM Provider Factory

Factory for creating LLM provider instances based on configuration.
"""

from typing import TYPE_CHECKING

from .base import LLMProvider, LLMProviderType

if TYPE_CHECKING:
    from src.config import LLMConfig


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Supports creating providers from configuration or directly.
    """

    @staticmethod
    def create(
        provider_type: str | LLMProviderType,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> LLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Provider type ("ollama" or "llamacpp")
            model: Model name/identifier
            base_url: Optional API base URL
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider type is unknown
        """
        # Normalize provider type
        if isinstance(provider_type, str):
            provider_type = provider_type.lower()
        else:
            provider_type = provider_type.value

        if provider_type == "ollama":
            from .ollama import DEFAULT_OLLAMA_URL, OllamaProvider

            return OllamaProvider(
                model=model,
                base_url=base_url or DEFAULT_OLLAMA_URL,
                timeout=timeout,
            )

        elif provider_type in ("llamacpp", "llama.cpp", "llama-cpp"):
            from .llamacpp import DEFAULT_LLAMACPP_URL, LlamaCppProvider

            return LlamaCppProvider(
                model=model,
                base_url=base_url or DEFAULT_LLAMACPP_URL,
                timeout=timeout,
            )

        else:
            raise ValueError(f"Unknown LLM provider type: {provider_type}")

    @staticmethod
    def from_config(config: "LLMConfig") -> LLMProvider:
        """
        Create an LLM provider from configuration.

        Args:
            config: LLMConfig object

        Returns:
            LLMProvider instance
        """
        return LLMProviderFactory.create(
            provider_type=config.provider,
            model=config.model,
            base_url=config.base_url,
            timeout=float(config.timeout_seconds),
        )


def create_provider(
    provider_type: str = "ollama",
    model: str = "llama3",
    base_url: str | None = None,
    timeout: float = 60.0,
) -> LLMProvider:
    """
    Convenience function to create an LLM provider.

    Args:
        provider_type: Provider type ("ollama" or "llamacpp")
        model: Model name
        base_url: Optional API base URL
        timeout: Request timeout

    Returns:
        LLMProvider instance
    """
    return LLMProviderFactory.create(
        provider_type=provider_type,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )


def create_provider_from_config() -> LLMProvider:
    """
    Create an LLM provider from the global configuration.

    Returns:
        LLMProvider instance based on current config
    """
    from src.config import get_config

    config = get_config()
    return LLMProviderFactory.from_config(config.llm)
