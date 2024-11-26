import random
from typing import Tuple, Dict

import numpy as np
import torch
from pydantic import BaseModel, Field, model_serializer

from configuration.audio_config import _AudioConfig
from configuration.rawboost_config import _RawboostConfig
from df_logger import main_logger


class _BaseConfig(BaseModel):
    """
    Base configuration class for audio preprocessing.
    This class can be inherited to create more specific configurations.
    """

    partition_ratio: Tuple[float, float] = Field(default=(0.8, 0.1), description="Train/validation split ratio")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    rawboost_config: _RawboostConfig = Field(description="Rawboost configuration")
    audio_config: _AudioConfig = Field(default=_AudioConfig(), description="Audio processing parameters")
    device: torch.device = Field(default=torch.device("cpu"), description="Device to use for processing")

    def __init__(self, **data):
        super().__init__(**data)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        main_logger.info(f"Using device: {self.device}")
        self.set_global_seed()

    class Config:
        """Pydantic configuration"""
        extra = "allow"  # Allows child classes to add extra fields
        arbitrary_types_allowed = True

    @model_serializer
    def serialize_model(self) -> Dict:

        result = {key: getattr(self, key) for key in self.model_fields if key != "audio_config"}

        result.update(self.audio_config.model_dump())
        return result

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print(_BaseConfig().model_dump())
