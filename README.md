image-generator
===============

Python library for generating images with Google Gemini or OpenAI.

Install
-------

```bash
uv sync
```

Environment
-----------

Set the API keys in your environment before importing the library:

```bash
export GEMINI_API_KEY=your_gemini_key
export OPENAI_API_KEY=your_openai_key
```

Gemini also accepts `GOOGLE_API_KEY` if you prefer that naming.

Usage
-----

```python
from image_generator import Provider, generate_image

image_bytes = generate_image(
    prompt="A minimalist poster of a banana in a gallery",
    provider=Provider.GOOGLE,
    model_version="gemini-2.5-flash-image",
    generation_params={
        "generationConfig": {
            "imageConfig": {
                "aspect_ratio": "16:9",
            },
        }
    }
)

with open("out.png", "wb") as f:
    f.write(image_bytes)
```

```python
from image_generator import Provider, generate_image

image_bytes = generate_image(
    prompt="A minimalist poster of a banana in a gallery",
    provider=Provider.OPENAI,
    model_version="gpt-image-2",
    generation_params={
        "quality": "low",
        "size": "1024x1024",
        # "response_format": "b64_json", # Required for dall-e family
    },
)

with open("out.png", "wb") as f:
    f.write(image_bytes)
```

Supported Models
----------------

GOOGLE

- `gemini-2.5-flash-image` (Nano Banana)
- `gemini-3-pro-image-preview` (Nano Banana Pro)
- `gemini-3.1-flash-image-preview` (Nano Banana 2)

Param details: https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_images

OPENAI

- `gpt-image-1`
- `gpt-image-1-mini`
- `gpt-image-1.5`
- `gpt-image-2`
- `chatgpt-image-latest`
- `dall-e-2`
- `dall-e-3`

Param details: https://developers.openai.com/api/reference/python/resources/images/methods/generate

Notes
-----

- `provider` is an enum (`Provider.GOOGLE`, `Provider.OPENAI`).
- `model_version` is a free string for versioning (e.g., `gpt-image-1`, `gemini-2.5-flash-image`).
- `generation_params` is passed through to the provider-specific request and can include advanced fields.
