from io import BytesIO
from types import ModuleType, SimpleNamespace
import builtins
import sys
import unittest
from unittest.mock import patch

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _install_import_stubs():
    modules = {}

    openai = ModuleType("openai")
    openai.OpenAI = object
    modules["openai"] = openai

    google = ModuleType("google")
    genai = ModuleType("google.genai")
    google.genai = genai
    modules["google"] = google
    modules["google.genai"] = genai

    diffusers = ModuleType("diffusers")
    diffusers.DiffusionPipeline = object
    modules["diffusers"] = diffusers

    torch = ModuleType("torch")
    torch.bfloat16 = object()
    torch.float32 = object()
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    class FakeGenerator:
        def __init__(self, device=None):
            self.device = device
            self._seed = None

        def manual_seed(self, seed):
            self._seed = seed
            return self

        def initial_seed(self):
            return self._seed

    torch.Generator = FakeGenerator
    modules["torch"] = torch

    patcher = patch.dict(sys.modules, modules)
    patcher.start()
    return patcher


_IMPORT_STUBS = _install_import_stubs()

from image_generator.errors import ConfigurationError, ImageGenerationError
from image_generator.providers.enums import Provider
from image_generator.providers.factory import get_provider
from image_generator.providers.vllm import VLLMProvider


class FakeImage:
    def save(self, buffer, format):
        if format != "PNG":
            raise AssertionError(f"unexpected format: {format}")
        buffer.write(PNG_BYTES)


class VLLMProviderTest(unittest.TestCase):
    def _install_fake_vllm_omni(self, outputs=None):
        captured = {}

        class FakeOmni:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def generate(self, request, sampling_params):
                captured["request"] = request
                captured["sampling_params"] = sampling_params
                if outputs is not None:
                    return outputs
                return [
                    SimpleNamespace(
                        request_output=SimpleNamespace(images=[FakeImage()]),
                    )
                ]

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                captured["sampling_kwargs"] = kwargs

        modules = {
            "vllm_omni": ModuleType("vllm_omni"),
            "vllm_omni.entrypoints": ModuleType("vllm_omni.entrypoints"),
            "vllm_omni.entrypoints.omni": ModuleType("vllm_omni.entrypoints.omni"),
            "vllm_omni.inputs": ModuleType("vllm_omni.inputs"),
            "vllm_omni.inputs.data": ModuleType("vllm_omni.inputs.data"),
            "vllm_omni.platforms": ModuleType("vllm_omni.platforms"),
        }
        modules["vllm_omni.entrypoints.omni"].Omni = FakeOmni
        modules["vllm_omni.inputs.data"].OmniDiffusionSamplingParams = FakeSamplingParams
        modules["vllm_omni.platforms"].current_omni_platform = SimpleNamespace(device_type="cpu")

        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        return captured

    def test_factory_returns_vllm_provider(self):
        captured = self._install_fake_vllm_omni()

        provider = get_provider(Provider.VLLM, model="Qwen/Qwen-Image-2512")

        self.assertIsInstance(provider, VLLMProvider)
        self.assertEqual(
            captured["init_kwargs"],
            {
                "model": "Qwen/Qwen-Image-2512",
                "mode": "text-to-image",
                "tensor_parallel_size": 1,
                "cfg_parallel_size": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
            },
        )

    def test_missing_dependency_raises_configuration_error(self):
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name.startswith("vllm_omni"):
                raise ImportError(name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaises(ConfigurationError):
                VLLMProvider("Qwen/Qwen-Image-2512")

    def test_default_parameters_are_applied(self):
        captured = self._install_fake_vllm_omni()
        provider = VLLMProvider("Qwen/Qwen-Image-2512")

        provider.generate("A cat on the moon")

        self.assertEqual(
            captured["request"],
            {
                "prompt": "A cat on the moon",
                "negative_prompt": None,
            },
        )
        self.assertEqual(
            captured["sampling_kwargs"],
            {
                "height": 1024,
                "width": 1024,
                "generator": None,
                "true_cfg_scale": 4.0,
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "num_outputs_per_prompt": 1,
                "extra_args": {},
            },
        )

    def test_generation_parameters_are_forwarded(self):
        captured = self._install_fake_vllm_omni()
        provider = VLLMProvider("Qwen/Qwen-Image-2512")

        provider.generate(
            "A gallery poster",
            generation_params={
                "width": 512,
                "height": 768,
                "num_inference_steps": 20,
                "guidance_scale": 3.5,
                "cfg_scale": 4.5,
                "negative_prompt": "low quality",
                "seed": 123,
                "num_images_per_prompt": 2,
                "guidance_scale_2": 1.25,
                "use_system_prompt": "en_unified",
                "system_prompt": "You are precise.",
                "timesteps_shift": 1.1,
                "cfg_schedule": "linear",
                "use_norm": True,
                "custom_extra": "value",
            },
        )

        self.assertEqual(
            captured["request"],
            {
                "prompt": "A gallery poster",
                "negative_prompt": "low quality",
            },
        )
        sampling_kwargs = captured["sampling_kwargs"]
        self.assertEqual(sampling_kwargs["height"], 768)
        self.assertEqual(sampling_kwargs["width"], 512)
        self.assertEqual(sampling_kwargs["true_cfg_scale"], 4.5)
        self.assertEqual(sampling_kwargs["guidance_scale"], 3.5)
        self.assertEqual(sampling_kwargs["guidance_scale_2"], 1.25)
        self.assertEqual(sampling_kwargs["num_inference_steps"], 20)
        self.assertEqual(sampling_kwargs["num_outputs_per_prompt"], 2)
        self.assertEqual(sampling_kwargs["generator"].initial_seed(), 123)
        self.assertEqual(
            sampling_kwargs["extra_args"],
            {
                "use_system_prompt": "en_unified",
                "system_prompt": "You are precise.",
                "timesteps_shift": 1.1,
                "cfg_schedule": "linear",
                "use_norm": True,
                "custom_extra": "value",
            },
        )

    def test_returns_png_bytes(self):
        self._install_fake_vllm_omni()
        provider = VLLMProvider("Qwen/Qwen-Image-2512")

        image_bytes = provider.generate("A red square")

        self.assertIsInstance(image_bytes, bytes)
        self.assertEqual(BytesIO(image_bytes).read(8), b"\x89PNG\r\n\x1a\n")

    def test_response_without_images_raises_generation_error(self):
        self._install_fake_vllm_omni(
            outputs=[SimpleNamespace(request_output=SimpleNamespace(images=[]))],
        )
        provider = VLLMProvider("Qwen/Qwen-Image-2512")

        with self.assertRaises(ImageGenerationError):
            provider.generate("A missing image")


if __name__ == "__main__":
    unittest.main()
