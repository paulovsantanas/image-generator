from image_generator import Provider, generate_image

if __name__ == "__main__":
    provider = Provider.OPENAI
    model_version = "gpt-image-2"
    # model_version = "dall-e-2"
    params = {
        "quality": "low",
        "size": "1024x1024",
        # "response_format": "b64_json", # Necessário para dall-e models
    }

    # provider = Provider.GOOGLE
    # model_version = "gemini-2.5-flash-image"
    # model_version = "gemini-3-pro-image-preview"
    # params = {
    #     "image_config": {
    #         "aspect_ratio": "16:9"
    #     },
    # },

    image_bytes = generate_image(
        prompt="A minimalist poster of a banana in a gallery",
        provider=provider,
        model_version=model_version,
        generation_params=params
    )

    with open(f"out_{model_version}.png", "wb") as f:
    # with open(f"out_imagen-4.0-generate-001.png", "wb") as f:
        f.write(image_bytes)