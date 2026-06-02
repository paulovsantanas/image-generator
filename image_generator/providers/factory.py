from image_generator.providers.base import ImageProvider
from image_generator.providers.enums import Provider
from image_generator.providers.gemini import GeminiProvider
from image_generator.providers.huggingface import HuggingFaceProvider
from image_generator.providers.openai import OpenAIProvider


def get_provider(provider: Provider, model: str) -> ImageProvider:
    if provider is Provider.OPENAI:
        return OpenAIProvider(model)
    if provider is Provider.GOOGLE:
        return GeminiProvider(model)
    if provider is Provider.HUGGINGFACE:
        return HuggingFaceProvider(model)
    raise ValueError(f"Unknown provider: {provider}")
