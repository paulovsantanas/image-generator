from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
