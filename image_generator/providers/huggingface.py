from io import BytesIO
from typing import Any

import torch
from diffusers import DiffusionPipeline

from image_generator.errors import ImageGenerationError
from image_generator.providers.base import ImageProvider


class HuggingFaceProvider(ImageProvider):
    def __init__(self, model: str):
        super().__init__(model)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=self._dtype,
        ).to(self._device)

    def generate(self, prompt: str, generation_params: dict[str, Any] | None = None) -> bytes:
        params = dict(generation_params or {})
        params.pop("prompt", None)
        params.pop("model", None)

        width = params.pop("width", 1024)
        height = params.pop("height", 1024)
        # num_inference_steps = params.pop("num_inference_steps", 50)
        guidance_scale = params.pop("guidance_scale", None)
        negative_prompt = params.pop("negative_prompt", None)
        seed = params.pop("seed", None)

        pipe_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            # "num_inference_steps": num_inference_steps,
        }

        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = negative_prompt
        if guidance_scale is not None:
            pipe_kwargs["guidance_scale"] = guidance_scale
        if seed is not None:
            pipe_kwargs["generator"] = torch.Generator(device=self._device).manual_seed(seed)

        pipe_kwargs.update(params)

        try:
            result = self._pipeline(**pipe_kwargs)
        except Exception as exc:
            raise ImageGenerationError(f"HuggingFace image generation failed: {exc}") from exc

        image = result.images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
