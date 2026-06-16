from image_generator.providers.base import ImageProvider
from image_generator.providers.enums import Provider


def get_provider(provider: Provider, model: str) -> ImageProvider:
    if provider is Provider.OPENAI:
        from image_generator.providers.openai import OpenAIProvider

        return OpenAIProvider(model)
    if provider is Provider.GOOGLE:
        from image_generator.providers.gemini import GeminiProvider

        return GeminiProvider(model)
    if provider is Provider.HUGGINGFACE:
        from image_generator.providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider(model)
    if provider is Provider.VLLM:
        from image_generator.providers.vllm import VLLMProvider

        return VLLMProvider(model)
    raise ValueError(f"Unknown provider: {provider}")
