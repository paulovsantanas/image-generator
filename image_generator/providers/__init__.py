from image_generator.providers.base import ImageProvider
from image_generator.providers.enums import Provider
from image_generator.providers.factory import get_provider

__all__ = [
    "GeminiProvider",
    "get_provider",
    "HuggingFaceProvider",
    "ImageProvider",
    "OpenAIProvider",
    "Provider",
    "VLLMProvider",
]


def __getattr__(name: str):
    if name == "GeminiProvider":
        from image_generator.providers.gemini import GeminiProvider

        return GeminiProvider
    if name == "HuggingFaceProvider":
        from image_generator.providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider
    if name == "OpenAIProvider":
        from image_generator.providers.openai import OpenAIProvider

        return OpenAIProvider
    if name == "VLLMProvider":
        from image_generator.providers.vllm import VLLMProvider

        return VLLMProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
