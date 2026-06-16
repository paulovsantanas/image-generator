import base64
import os
from typing import Any

from image_generator.errors import ConfigurationError, ImageGenerationError
from image_generator.providers.base import ImageProvider


class OpenAIProvider(ImageProvider):
    def generate(self, prompt: str, generation_params: dict[str, Any] | None = None) -> bytes:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError(
                'OpenAI dependencies are not installed. Install them with '
                '`pip install "image-generator[openai]"` before using Provider.OPENAI.'
            ) from exc

        api_key = self._get_api_key()
        params = dict(generation_params or {})
        params.pop("prompt", None)

        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model=self.model,
            prompt=prompt,
            **params,
        )

        image_data = None
        if response.data:
            image_data = response.data[0].b64_json

        if not image_data:
            raise ImageGenerationError("OpenAI response did not include image data")

        return base64.b64decode(image_data)

    @staticmethod
    def _get_api_key() -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set")
        return api_key
