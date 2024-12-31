import copy
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from yaml import safe_load

from configuration.df_train_config import DF_Train_Config
from configuration.rawboost_config import _RawboostConfig
from configuration.train_config import _TrainerConfig
from df_logger import main_logger
from parser import parse_arguments
from src.datasets.asvspoof2019_dataset import ASVspoof2019Dataset
from src.train_models import train_nn
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    # set up the configuration
    train_config = DF_Train_Config(
        seed=42,
        # set up the training configuration
        trainer_config=_TrainerConfig(
            optimizer=Adam,
            batch_size=32,
            num_epochs=25,
            early_stopping=False,
            early_stopping_patience=5,
            optimizer_parameters={
                # "lr": 1e-3,
                # "weight_decay": 5e-5, # 0.00005
                "lr": 5e-6,
                "weight_decay": 5e-7
            },
            criterion=BCEWithLogitsLoss,
        ),
        root_dir=Path(os.getenv("PATH_TO_ASV")), 
        root_path_to_protocol=Path(os.getenv("PATH_TO_ASV_PROTOCOL")), 
        rawboost_config=_RawboostConfig(algo_id=0),
    )

    args = parse_arguments()

    if args.batch_size is not None:
        train_config.trainer_config.batch_size = args.batch_size
    if args.epochs is not None:
        train_config.trainer_config.num_epochs = args.epochs
    if args.lr is not None:
        train_config.trainer_config.optimizer_parameters["lr"] = args.lr
    if args.weight_decay is not None:
        train_config.trainer_config.optimizer_parameters["weight_decay"] = args.weight_decay
    if args.model_out_dir is not None:
        train_config.out_model_dir = args.model_out_dir
    if args.early_stopping is not None:
        train_config.trainer_config.early_stopping = args.early_stopping
    if args.early_stopping_patience is not None:
        train_config.trainer_config.early_stopping_patience = args.early_stopping_patience
    if args.dataset_dir is not None:
        train_config.root_dir = args.dataset_dir
    if args.rawboost_algo is not None:
        train_config.rawboost_config = _RawboostConfig(algo_id=args.rawboost_algo)


    with open(args.config, mode="r") as f:
        model_config = safe_load(f)

    model_name = model_config["model"]["name"]
    model_config["model"]['rawboost_algo'] = train_config.rawboost_config.algo_id

    train_dataset = ASVspoof2019Dataset(
        config=train_config,
        root_path=train_config.root_dir,
        protocol_dir=train_config.root_path_to_protocol,
        subset="train"
    )
    val_dataset = ASVspoof2019Dataset(
        config=train_config,
        root_path=train_config.root_dir,
        protocol_dir=train_config.root_path_to_protocol,
        subset="val"
    )

    test_dataset = ASVspoof2019Dataset(
        config=train_config,
        root_path=train_config.root_dir,
        protocol_dir=train_config.root_path_to_protocol,
        subset="test"
    )

    # merge datasets into train_dataset
    train_dataset.samples_df = pd.concat([
        train_dataset.samples_df,
        test_dataset.samples_df,
        val_dataset.samples_df
        ]
    ).reset_index(drop=True)

    # delete val_dataset and test_dataset
    del val_dataset, test_dataset

    # split the dataset into train and validation
    train_df, val_df = train_test_split(
        train_dataset.samples_df,
        test_size=0.2,
        random_state=train_config.seed,
    )
    val_dataset = copy.deepcopy(train_dataset)
    train_dataset.samples_df = train_df
    val_dataset.samples_df = val_df
    main_logger.info(f"Dataset sizes train: {len(train_dataset)}, val: {len(val_dataset)}")

    assert len((set(train_dataset.samples_df['path']))& set(val_dataset.samples_df['path']))==0, "Train and validation datasets have common samples"
    # create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.trainer_config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.trainer_config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )
    out_model_dir = Path(train_config.out_model_dir) / f"{model_name}_lr_{train_config.trainer_config.optimizer_parameters['lr']}_wd_{train_config.trainer_config.optimizer_parameters['weight_decay']}"
    out_model_dir.mkdir(parents=True, exist_ok=True)

    # create the trainer
    config_save_path, checkpoint_path = train_nn(
        data_train=train_loader,
        data_test=val_loader,
        config=train_config,
        model_config=model_config,
        out_dir=out_model_dir,
        device=train_config.device,
    )

    main_logger.info(f"Model has been trained. Configuration saved at {config_save_path}")

    return config_save_path


if __name__ == "__main__":
    main()
