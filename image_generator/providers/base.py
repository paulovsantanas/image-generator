from abc import ABC, abstractmethod
from typing import Any


class ImageProvider(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, generation_params: dict[str, Any] | None = None) -> bytes: ...
