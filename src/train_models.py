from os.path import join, normpath, dirname, exists
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import yaml
from torch.utils.data import DataLoader

from configuration.df_ft_config import DF_FT_Config
from src.models import get_model
from src.trainer import GDTrainer

from df_logger import main_logger
from src.utils import save_model


def train_nn(
        data_train: DataLoader,
        data_test: DataLoader,
        config: DF_FT_Config,
        device: Union[torch.device, str],
        model_config: Dict,
        out_dir: Union[str, Path]
) -> Tuple[str, str]:
    # load model architecture
    model_name, model_parameters = model_config["model"]["name"], model_config["model"]["parameters"]

    optimizer_config = config.trainer_config.optimizer_parameters
    main_logger.info(f"Optimizer config: {optimizer_config}, device: {device}")
    current_model = get_model(
        model_name=model_name,
        config=model_parameters,
        device=device
    )

    # If provided weights, apply corresponding ones (from an appropriate fold)
    checkpoint_path = model_config.get("model", {}).get("checkpoint_path", None)

    if checkpoint_path is not None and exists(checkpoint_path):
        current_model.load_state_dict(torch.load(checkpoint_path), strict=False)
        main_logger.info(f"Loaded weights from '{checkpoint_path}'.")
    else:
        main_logger.info(f"Training '{model_name}' model on {len(data_train)} audio files.")
    current_model = current_model.to(device)

    use_scheduler = "rawnet3" in model_name.lower()

    current_model = GDTrainer(
        device=device,
        config=config.trainer_config,
        use_scheduler=use_scheduler,
        checkpoint_path=out_dir / "checkpoints"  # type: ignore

    ).train(
        train_loader=data_train,
        model=current_model,
        test_loader=data_test,
    )

    checkpoint_path = save_model(
        model=current_model,
        model_dir=out_dir,
        file_name="weights.pth",
    )

    model_config['model']["checkpoint_path"] = checkpoint_path
    model_config["model"]["optimizer"] = optimizer_config
    config_name = "configuration.yaml"
    with open(join(dirname(checkpoint_path), config_name), mode="w+") as f:
        yaml.dump(model_config, f, default_flow_style=False)
    main_logger.info(f"Test configuration {config_name} saved at location '{out_dir}'!")
    del current_model

    return out_dir, checkpoint_path
