import base64
import os
from typing import Any

from openai import OpenAI

from .errors import ConfigurationError, ImageGenerationError


def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set")
    return api_key


def generate_image_openai(
    prompt: str,
    model_version: str,
    generation_params: dict[str, Any],
) -> bytes:
    api_key = _get_api_key()
    params = dict(generation_params or {})
    params.pop("prompt", None)

    client = OpenAI(api_key=api_key)

    response = client.images.generate(
        model=model_version,
        prompt=prompt,
        **params,
    )

    image_data = None
    if response.data:
        image_data = response.data[0].b64_json

    if not image_data:
        raise ImageGenerationError("OpenAI response did not include image data")

    return base64.b64decode(image_data)
