from typing import Any

from .client_gemini import generate_image_gemini
from .client_openai import generate_image_openai
from .errors import ConfigurationError, ImageGenerationError
from .models import Provider

__all__ = [
    "ConfigurationError",
    "ImageGenerationError",
    "Provider",
    "generate_image",
]


def _normalize_provider(provider: Provider | str) -> Provider:
    if isinstance(provider, Provider):
        return provider
    try:
        return Provider(str(provider).lower())
    except ValueError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc


def generate_image(
    prompt: str,
    provider: Provider | str,
    model_version: str,
    generation_params: dict[str, Any] | None = None,
) -> bytes:
    normalized_provider = _normalize_provider(provider)
    params = generation_params or {}

    if normalized_provider is Provider.GOOGLE:
        return generate_image_gemini(prompt, model_version, params)
    if normalized_provider is Provider.OPENAI:
        return generate_image_openai(prompt, model_version, params)

    raise ValueError(f"Unsupported provider: {normalized_provider}")
