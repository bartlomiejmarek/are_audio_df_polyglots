from os.path import isdir
from pathlib import Path
from typing import Union

import torch

from df_logger import main_logger


def save_model(
        model: torch.nn.Module,
        model_dir: Union[Path, str],
        file_name: str,
) -> Union[str, Path]:
    # change type to Path
    model_dir = Path(model_dir)
    if not isdir(model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), f"{model_dir}/{file_name}")
    main_logger.info(f"Model saved at location '{model_dir}'!")
    return f"{model_dir}/{file_name}"
