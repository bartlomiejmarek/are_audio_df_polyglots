from pathlib import Path
from typing import Union

from pydantic import Field

from configuration.audio_config import _AudioConfig
from configuration.base_config import _BaseConfig
from configuration.whisper_audio_config import _WhisperAudioConfig


class ModelConfig(_BaseConfig):
    whisper_model_weights_path: Union[str, Path] = Field(
        description="Path to the model Whisper weights",
    )
    w2v_model_path: Union[str, Path] = Field(
        description="Path to the W2V model",
    )

    whisper_audio_config: _WhisperAudioConfig = Field(
        default=_WhisperAudioConfig(),
        description="Audio configuration for Whisper model"
    )
    audio_config: _AudioConfig = Field(
        default=_AudioConfig(),
        description="Audio configuration"
    )
