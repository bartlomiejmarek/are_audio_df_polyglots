from collections import OrderedDict
from pathlib import Path

import torch
import whisper

WHISPER_MODEL_WEIGHTS_PATH = "src/models/assets"

models = [
    # "tiny",
    # "small",
    # "medium",
    "large-v2",
]
def download_whisper(whisper_size):
    model = whisper.load_model(whisper_size)
    return model


def extract_and_save_encoder(model, save_path):
    model_ckpt = OrderedDict()

    model_ckpt['model_state_dict'] = OrderedDict()

    for key, value in model.encoder.state_dict().items():
        model_ckpt['model_state_dict'][f'encoder.{key}'] = value

    model_ckpt['dims'] = model.dims
    torch.save(model_ckpt, save_path)

for m in models:
    print(f"Downloading 'whisper-{m}' model!")
    model = download_whisper(m)
    print(f"'whisper-{m}' downloaded!")
    save_path = Path(WHISPER_MODEL_WEIGHTS_PATH) / f"whisper_{m}.pth"
    extract_and_save_encoder(model, save_path)
    print(f"Saved encoder at '{save_path}'")