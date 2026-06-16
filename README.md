image-generator
===============

Python library for generating images with OpenAI, Google Gemini, HuggingFace, or vLLM-Omni.

Providers follow the **Factory Pattern**: obtain a provider instance via `get_provider` and call `.generate()` on it.

Install
-------

```bash
pip install git+https://github.com/paulovsantanas/image-generator.git
```

Provider dependencies are optional. Install only the extras you need:

```bash
pip install "image-generator[openai] @ git+https://github.com/paulovsantanas/image-generator.git"
pip install "image-generator[google] @ git+https://github.com/paulovsantanas/image-generator.git"
pip install "image-generator[huggingface] @ git+https://github.com/paulovsantanas/image-generator.git"
pip install "image-generator[vllm] @ git+https://github.com/paulovsantanas/image-generator.git"
pip install "image-generator[all] @ git+https://github.com/paulovsantanas/image-generator.git"
```

Environment
-----------

Set the API keys in your environment before using API-based providers:

```bash
export OPENAI_API_KEY=your_openai_key
export GEMINI_API_KEY=your_gemini_key
```

Gemini also accepts `GOOGLE_API_KEY` if you prefer that naming.

> **HuggingFace** loads models **locally** via `diffusers`. No API key is required by default. The model is downloaded on first use and cached on disk by the `diffusers` library.
>
> **vLLM-Omni** loads models **locally** and is best suited for GPU environments. Install the optional dependency group before using it:
>
> ```bash
> pip install "image-generator[vllm]"
> ```

Usage
-----

### OpenAI

```python
from image_generator import Provider, get_provider

provider = get_provider(Provider.OPENAI, model="gpt-image-2")
image_bytes = provider.generate(
    prompt="A minimalist poster of a banana in a gallery",
    generation_params={"quality": "low", "size": "1024x1024"},
)

with open("out.png", "wb") as f:
    f.write(image_bytes)
```

### Google Gemini

```python
from image_generator import Provider, get_provider

provider = get_provider(Provider.GOOGLE, model="gemini-2.5-flash-image")
image_bytes = provider.generate(
    prompt="A minimalist poster of a banana in a gallery",
    generation_params={
        "generationConfig": {
            "imageConfig": {
                "aspect_ratio": "16:9",
            },
        }
    },
)

with open("out.png", "wb") as f:
    f.write(image_bytes)
```

### HuggingFace (local)

```python
from image_generator import Provider, get_provider

provider = get_provider(Provider.HUGGINGFACE, model="Qwen/Qwen-Image-2512")

image_bytes = provider.generate(
    prompt="A minimalist poster of a banana in a gallery",
    generation_params={
        "width": 1664,
        "height": 928,
        "num_inference_steps": 50,
    },
)

with open("out.png", "wb") as f:
    f.write(image_bytes)

# The same provider instance can be reused without reloading the model.
image_bytes_2 = provider.generate("A cat on the moon")
```

### vLLM-Omni (local)

```python
from image_generator import Provider, get_provider

provider = get_provider(Provider.VLLM, model="Qwen/Qwen-Image-2512")

image_bytes = provider.generate(
    prompt="A minimalist poster of a banana in a gallery",
    generation_params={
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "cfg_scale": 4.0,
        "guidance_scale": 4.0,
        "seed": 142,
    },
)

with open("out.png", "wb") as f:
    f.write(image_bytes)

# Reuse the same provider instance to avoid reloading the model.
image_bytes_2 = provider.generate("A cat on the moon")
```

Supported Models
----------------

### OPENAI

- `gpt-image-1`
- `gpt-image-1-mini`
- `gpt-image-1.5`
- `gpt-image-2`
- `chatgpt-image-latest`
- `dall-e-2`
- `dall-e-3`

Param details: https://developers.openai.com/api/reference/python/resources/images/methods/generate

### GOOGLE

- `gemini-2.5-flash-image` (Nano Banana)
- `gemini-3-pro-image-preview` (Nano Banana Pro)
- `gemini-3.1-flash-image-preview` (Nano Banana 2)

Param details: https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_images

### HUGGINGFACE

- `Qwen/Qwen-Image-2512`

Other diffusers-compatible models may work. Pass any HuggingFace model ID as the `model` argument.

### VLLM

- `Qwen/Qwen-Image-2512`

Other vLLM-Omni text-to-image models may work. Pass any supported model ID as the `model` argument.

HuggingFace Generation Parameters
---------------------------------

The HuggingFace provider accepts the following fields inside `generation_params`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | `1024` | Output width in pixels |
| `height` | `1024` | Output height in pixels |
| `num_inference_steps` | `50` | Number of denoising steps |
| `guidance_scale` | `None` | Guidance scale (CFG) |
| `negative_prompt` | `None` | Negative prompt |
| `seed` | `None` | Random seed for reproducibility |

Any additional keys are passed through to the `DiffusionPipeline.__call__`.

vLLM-Omni Generation Parameters
-------------------------------

The vLLM-Omni provider accepts the following fields inside `generation_params`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | `1024` | Output width in pixels |
| `height` | `1024` | Output height in pixels |
| `num_inference_steps` | `50` | Number of denoising steps |
| `guidance_scale` | `4.0` | Classifier-free guidance scale |
| `cfg_scale` | `4.0` | True classifier-free guidance scale used by Qwen Image |
| `negative_prompt` | `None` | Negative prompt |
| `seed` | `None` | Random seed for reproducibility |
| `num_images_per_prompt` | `1` | Number of images to generate for the prompt |
| `guidance_scale_2` | `None` | Secondary guidance scale for models that support it |
| `use_system_prompt` | `None` | vLLM-Omni system prompt preset |
| `system_prompt` | `None` | Custom system prompt |
| `timesteps_shift` | provider default | Advanced sampler parameter |
| `cfg_schedule` | provider default | Advanced CFG schedule parameter |
| `use_norm` | provider default | Advanced normalization parameter |

Any additional keys are passed through to `OmniDiffusionSamplingParams.extra_args`.

Reusing a Provider
------------------

`get_provider` returns an `ImageProvider` instance. Reuse the **same instance** to avoid extra overhead:

- For **OpenAI** and **Gemini**, this avoids recreating the HTTP client each time.
- For **HuggingFace**, this is critical: the model is loaded into GPU/CPU memory on instantiation. Reusing the instance avoids reloading the entire diffusion pipeline.
- For **vLLM-Omni**, this is also critical: the model is loaded locally on instantiation. Reusing the instance avoids reloading the vLLM-Omni pipeline.

Notes
-----

- Obtain a provider with `get_provider(Provider.<NAME>, model="...")`.
- `model` is a free string for versioning (e.g., `gpt-image-2`, `gemini-2.5-flash-image`, `Qwen/Qwen-Image-2512`).
- `generation_params` is passed through to the provider-specific request and can include advanced fields.
