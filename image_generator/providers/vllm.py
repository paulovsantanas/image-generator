from io import BytesIO
from typing import Any

from image_generator.errors import ConfigurationError, ImageGenerationError
from image_generator.providers.base import ImageProvider


class VLLMProvider(ImageProvider):
    _EXTRA_ARG_KEYS = {
        "use_system_prompt",
        "system_prompt",
        "timesteps_shift",
        "cfg_schedule",
        "use_norm",
    }

    def __init__(self, model: str):
        super().__init__(model)
        Omni, SamplingParams, current_omni_platform = self._load_vllm_omni()
        self._sampling_params_class = SamplingParams
        self._current_omni_platform = current_omni_platform

        try:
            self._omni = Omni(
                model=self.model,
                mode="text-to-image",
                tensor_parallel_size=1,
                cfg_parallel_size=1,
                ulysses_degree=1,
                ring_degree=1,
            )
        except Exception as exc:
            raise ImageGenerationError(f"vLLM-Omni initialization failed: {exc}") from exc

    def generate(self, prompt: str, generation_params: dict[str, Any] | None = None) -> bytes:
        params = dict(generation_params or {})
        params.pop("prompt", None)
        params.pop("model", None)

        width = params.pop("width", 1024)
        height = params.pop("height", 1024)
        num_inference_steps = params.pop("num_inference_steps", 50)
        guidance_scale = params.pop("guidance_scale", 4.0)
        cfg_scale = params.pop("cfg_scale", 4.0)
        negative_prompt = params.pop("negative_prompt", None)
        seed = params.pop("seed", None)
        num_images_per_prompt = params.pop("num_images_per_prompt", 1)
        guidance_scale_2 = params.pop("guidance_scale_2", None)

        extra_args = {
            key: params.pop(key)
            for key in list(params)
            if key in self._EXTRA_ARG_KEYS
        }
        extra_args.update(params)

        generator = None
        if seed is not None:
            try:
                import torch
            except ImportError as exc:
                raise ConfigurationError(
                    'Torch is required for seeded vLLM-Omni generation. Install the '
                    '`vllm` extra with `pip install "image-generator[vllm]"`.'
                ) from exc
            generator = torch.Generator(
                device=self._current_omni_platform.device_type,
            ).manual_seed(seed)

        sampling_kwargs: dict[str, Any] = {
            "height": height,
            "width": width,
            "generator": generator,
            "true_cfg_scale": cfg_scale,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_outputs_per_prompt": num_images_per_prompt,
            "extra_args": extra_args,
        }
        if guidance_scale_2 is not None:
            sampling_kwargs["guidance_scale_2"] = guidance_scale_2

        try:
            outputs = self._omni.generate(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                },
                self._sampling_params_class(**sampling_kwargs),
            )
        except Exception as exc:
            raise ImageGenerationError(f"vLLM-Omni image generation failed: {exc}") from exc

        image = self._extract_first_image(outputs)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _load_vllm_omni() -> tuple[Any, Any, Any]:
        try:
            from vllm_omni.entrypoints.omni import Omni
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
            from vllm_omni.platforms import current_omni_platform
        except ImportError as exc:
            raise ConfigurationError(
                'vLLM-Omni dependencies are not installed. Install them with '
                '`pip install "image-generator[vllm]"` before using Provider.VLLM.'
            ) from exc
        return Omni, OmniDiffusionSamplingParams, current_omni_platform

    @staticmethod
    def _extract_first_image(outputs: Any) -> Any:
        if not outputs:
            raise ImageGenerationError("vLLM-Omni response did not include outputs")

        first_output = outputs[0]
        request_output = getattr(first_output, "request_output", None)
        if request_output is None:
            raise ImageGenerationError("vLLM-Omni response did not include request_output")

        images = getattr(request_output, "images", None)
        if not images:
            raise ImageGenerationError("vLLM-Omni response did not include image data")

        return images[0]
