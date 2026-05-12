import base64
import os
from typing import Any

from google import genai

from .errors import ConfigurationError, ImageGenerationError


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    return api_key


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if "responseModalities" in normalized and "response_modalities" not in normalized:
        normalized["response_modalities"] = normalized.pop("responseModalities")
    if "imageConfig" in normalized and "image_config" not in normalized:
        normalized["image_config"] = normalized.pop("imageConfig")
    return normalized


def _extract_image_bytes(response: Any) -> bytes:
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            inline_data = part.inline_data
            if inline_data and inline_data.data:
                data = inline_data.data
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
                if isinstance(data, str):
                    return base64.b64decode(data)
    raise ImageGenerationError("Gemini response did not include image data")


def generate_image_gemini(
    prompt: str,
    model_version: str,
    generation_params: dict[str, Any],
) -> bytes:
    api_key = _get_api_key()
    client = genai.Client(api_key=api_key)

    params = dict(generation_params or {})
    contents = params.pop("contents", [prompt])
    config = params.pop("config", None) or params.pop("generationConfig", None)
    if config is None:
        config = {"response_modalities": ["IMAGE"]}
    else:
        config = _normalize_config(config)
        config.setdefault("response_modalities", ["IMAGE"])

    print(config)
    try:
        response = client.models.generate_content(
            model=model_version,
            contents=contents,
            config=config,
            **params,
        )
    except Exception as exc:
        raise ImageGenerationError(f"Gemini image generation failed: {exc}") from exc

    return _extract_image_bytes(response)
