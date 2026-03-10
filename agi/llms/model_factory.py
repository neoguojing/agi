from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Union

from langchain_core.runnables import Runnable

from agi.config import log
from agi.llms.base import CustomerLLM
from agi.llms.image2image import Image2Image
from agi.llms.multimodel import MultiModel
from agi.llms.speech2text import Speech2Text
from agi.llms.text2image import Text2Image
from agi.llms.tts import TextToSpeech

ModelInstance = Union[CustomerLLM, Runnable]


class ModelFactory:
    """Thread-safe model factory with in-memory singleton caching."""

    _instances: OrderedDict[str, ModelInstance] = OrderedDict()
    _lock = threading.Lock()

    _registry: dict[str, Callable[[], ModelInstance]] = {
        "text2image": Text2Image,
        "image2image": Image2Image,
        "speech2text": Speech2Text,
        "text2speech": TextToSpeech,
        "multimodel": MultiModel,
    }

    @staticmethod
    def get_model(model_type: str) -> ModelInstance:
        """Retrieve and cache a model instance by model type."""
        with ModelFactory._lock:
            if model_type not in ModelFactory._instances:
                ModelFactory._instances[model_type] = ModelFactory._load_model(model_type)
            return ModelFactory._instances[model_type]

    @staticmethod
    def _load_model(model_type: str) -> ModelInstance:
        factory = ModelFactory._registry.get(model_type)
        if factory is None:
            raise ValueError(f"Invalid model type: {model_type}")

        log.info(f"create model instance: {model_type}")
        return factory()

    @staticmethod
    def reset() -> None:
        """Destroy and clear all cached model instances."""
        with ModelFactory._lock:
            for key, instance in list(ModelFactory._instances.items()):
                destroy = getattr(instance, "destroy", None)
                if callable(destroy):
                    try:
                        destroy()
                    except Exception as exc:  # noqa: BLE001
                        log.warning(f"Error destroying model {key}: {exc}")
            ModelFactory._instances.clear()
