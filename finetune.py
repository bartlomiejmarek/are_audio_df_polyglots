import copy
from pathlib import Path

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
from src.datasets.mlaad_dataset import MLAADDataset
from src.train_models import train_nn
import os 
from dotenv import load_dotenv

load_dotenv()


def main():
    ft_config = DF_Train_Config(
        seed=42,
        # set up the training configuration
        trainer_config=_TrainerConfig(
            optimizer=Adam,
            batch_size=32,
            num_epochs=10,
            early_stopping=False,
            early_stopping_patience=5,
            optimizer_parameters={
                "lr": 1e-6,
                "weight_decay": 1e-7
            },
            criterion=BCEWithLogitsLoss,
        ),
        root_path_to_protocol=Path(os.getenv('PATH_TO_MLAAD')),
        rawboost_config=_RawboostConfig(algo_id=0),
    )

    args = parse_arguments()
    if args.output_file is not None:
        ft_config.evaluate_output_file = args.output_file
    if args.batch_size is not None:
        ft_config.trainer_config.batch_size = args.batch_size
    if args.epochs is not None:
        ft_config.trainer_config.num_epochs = args.epochs
    if args.lr is not None:
        ft_config.trainer_config.optimizer_parameters["lr"] = args.lr
    if args.weight_decay is not None:
        ft_config.trainer_config.optimizer_parameters["weight_decay"] = args.weight_decay
    if args.model_out_dir is not None:
        ft_config.out_model_dir = args.model_out_dir
    if args.early_stopping is not None:
        ft_config.trainer_config.early_stopping = args.early_stopping
    if args.early_stopping_patience is not None:
        ft_config.trainer_config.early_stopping_patience = args.early_stopping_patience
    if args.dataset_dir is not None:
        ft_config.root_dir = args.dataset_dir
    if args.rawboost_algo is not None:
        ft_config.rawboost_config = _RawboostConfig(algo_id=args.rawboost_algo)
        
    with open(args.config, mode="r") as f:
        model_config = safe_load(f)

    model_name = model_config.get("model", {}).get("name")
    model_config["model"]["fine_tune"] = args.ft_languages
    model_config["model"]['rawboost_algo'] = ft_config.rawboost_config.algo_id
    main_logger.info(f"Running {model_name} fine-tuned with {args.ft_languages}.")
    ft_filter = ("architecture", ['griffin_lim', 'vits'], "include")
    # ft_filter = ("architecture", ['griffin_lim', 'vits'], "include")
    model_config["model"]['filter_strategy'] = {
        "column_name": ft_filter[0],
        "values": ft_filter[1],
        "strategy": ft_filter[2]
    }
       
    train_dataset = MLAADDataset(
        config=ft_config,
        root_path=ft_config.root_path_to_protocol,
        languages=args.ft_languages,
        predefined_column_and_values=ft_filter,  # type: ignore
        split_languages_separately=True,
    )

    # split the dataset into train and validation
    train_df, val_df = train_test_split(
        train_dataset.samples_df,
        test_size=0.2,
        random_state=42,
    )

    val_dataset = copy.deepcopy(train_dataset)
    train_dataset.samples_df = train_df
    val_dataset.samples_df = val_df
    main_logger.info(f"Dataset sizes train: {len(train_dataset)}, val: {len(val_dataset)}")

    # create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=ft_config.trainer_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ft_config.trainer_config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4
    )
    out_model_dir = Path(ft_config.out_model_dir) / (f'{model_name}_ft_{"_".join(args.ft_languages)}_{ft_filter[0]}_{"".join(ft_filter[1])}_{ft_filter[2]}')
    out_model_dir.mkdir(parents=True, exist_ok=True)
    
    main_logger.info(out_model_dir)
    # create the trainer
    config_save_path, checkpoint_path = train_nn(
        data_train=train_loader,
        data_test=val_loader,
        config=ft_config,
        model_config=model_config,
        out_dir=out_model_dir,
        device=ft_config.device,
    )

    main_logger.info(f"Model has been fine-tuned. Configuration saved at {config_save_path}")
    return config_save_path


if __name__ == "__main__":
    main()
