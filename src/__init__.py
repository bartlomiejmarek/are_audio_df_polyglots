"""
Source: https://github.com/piotrkawa/deepfake-whisper-features
Author: Piotr Kawa (https://github.com/piotrkawa)

"""
from configuration.audio_config import _AudioConfig
from configuration.models_config import ModelConfig
from configuration.rawboost_config import _RawboostConfig
from dotenv import load_dotenv
from pathlib import Path 

load_dotenv()

model_configuration = ModelConfig(
    whisper_model_weights_path=Path('ASSET_PATH') / "whisper_large-v2.pth",
    w2v_model_path='models/XLR_300M/xlsr2_300m.pt',
    audio_config=_AudioConfig(),
    rawboost_config=_RawboostConfig(algo_id=4)
)
