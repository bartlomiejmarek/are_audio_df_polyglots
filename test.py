from copy import deepcopy

from pathlib import Path

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from yaml import safe_load

from configuration.df_ft_config import DF_FT_Config
from configuration.rawboost_config import _RawboostConfig
from configuration.train_config import _TrainerConfig
from df_logger import main_logger
from parser import parse_arguments
from src.datasets.mlaad_dataset import MLAADDataset
from src.evaluate import evaluate_nn
from utils.utils import append_to_csv
import os
from dotenv import load_dotenv
load_dotenv()

def main():

    test_config = DF_FT_Config(
        seed=42,
        # set up the training configuration
        trainer_config=_TrainerConfig(
            optimizer=Adam,
            batch_size=32,
            num_epochs=25,
            early_stopping=False,
            early_stopping_patience=5,
            optimizer_parameters={
                "lr": 1.0e-05,
                "weight_decay": 0.0001
            },
            criterion=BCEWithLogitsLoss,
        ),
        rawboost_config=_RawboostConfig(algo_id=0),
        root_path_to_protocol=Path(os.getenv('PATH_TO_MLAAD')),
        evaluate_output_file="results/evaluation_results.csv",
    )

    args = parse_arguments()
    if args.output_file is not None:
        test_config.evaluate_output_file = args.output_file
    if args.batch_size is not None:
        test_config.trainer_config.batch_size = args.batch_size
    if args.epochs is not None:
        test_config.trainer_config.num_epochs = args.epochs
    if args.lr is not None:
        test_config.trainer_config.optimizer_parameters["lr"] = args.lr
    if args.model_out_dir is not None:
        test_config.out_model_dir = args.model_out_dir
    if args.early_stopping is not None:
        test_config.trainer_config.early_stopping = args.early_stopping
    if args.early_stopping_patience is not None:
        test_config.trainer_config.early_stopping_patience = args.early_stopping_patience
    if args.dataset_dir is not None:
        test_config.root_dir = args.dataset_dir
    if args.rawboost_algo is not None:
        test_config.rawboost_config = _RawboostConfig(algo_id=args.rawboost_algo)

    if "uk" in args.eval_languages:
        test_filter = ("architecture", ['griffin_lim', 'vits'], "exclude")
    else:
        test_filter = ("architecture", ['xtts_v1.1', 'xtts_v2'], "include")

    with open(args.config, mode="r") as f:
        model_config = safe_load(f)

    # define the dataset
    test_dataset = MLAADDataset(
        config=test_config,
        root_path=test_config.root_path_to_protocol,
        languages=args.eval_languages,
        predefined_column_and_values=test_filter,
        split_languages_separately=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config.trainer_config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    main_logger.info(f'Test dataset {len(test_dataset.samples_df)}')

    y, y_pred, y_pred_label = evaluate_nn(
            model_path=model_config["model"]["checkpoint_path"],
            test_loader=test_loader,
            model_config=model_config["model"],
            device=test_config.device,
    )

    eval_results = {
        "y": y.cpu().numpy().tolist(),
        "y_pred": y_pred.cpu().numpy().tolist(),
        "y_pred_label": y_pred_label.cpu().numpy().tolist(),
        'model': model_config["model"]["name"],
        'model_path': model_config["model"]["checkpoint_path"],
        "language balance": dict(test_dataset.samples_df['language'].value_counts()),
        "label balance": dict(test_dataset.samples_df['label'].value_counts()),
        'fine-tuned languages': model_config["model"].get("fine_tune", []),
        'evaluated languages': args.eval_languages,
        'rawboost_algo': model_config["model"].get("rawboost_algo", 0)
    }
    append_to_csv(test_config.evaluate_output_file, list(eval_results.keys()), list(eval_results.values()))


if __name__ == "__main__":
    main()
