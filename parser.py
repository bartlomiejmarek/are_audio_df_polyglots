import argparse
from os.path import isdir

from configuration.df_ft_config import DF_FT_Config

def parse_arguments(
        description: str = "Run script to train and evaluate model",
):
    parser = argparse.ArgumentParser(description=description)

    # General argument
    parser.add_argument(
        "--dataset_dir",
        type=lambda x: x if isdir(x) else parser.error(f"{x} does not exist."),
        help="Path to the MLAAD dataset directory",
    )

    parser.add_argument(
        "--rawboost_algo",
        type=int,
        help="Rawboost algorithm id",
        choices=range(0, 10)
    )

    # Fine-tuning / Training arguments
    ft_group = parser.add_argument_group("Fine-tuning / Training arguments")

    ft_group.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file"
    )

    ft_group.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train the model",
    )

    ft_group.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training the model"
    )

    ft_group.add_argument(
        "--lr",
        type=float,
        help="Learning rate for the optimizer"
    )

    ft_group.add_argument(
        "--weight_decay",
        type=float,
        help="Learning rate for the optimizer"
    )

    ft_group.add_argument(
        "--model_out_dir",
        "-o",
        type=str,
        help="Path to output directory where the trained or fine-tuned model will be saved"
    )

    ft_group.add_argument(
        "--ft_languages",
        nargs="+",
        type=str,
        help="List of language codes for fine-tuning",
        choices=['de', 'en', 'es', 'pl', 'uk', 'fr', 'it', 'ru'],
    )

    es_group = parser.add_argument_group("Early Stopping arguments")

    es_group.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping during training"
    )

    es_group.add_argument(
        "--early_stopping_patience",
        type=int,
        help="Number of epochs to wait for improvement before stopping early. Required if early stopping is enabled.",
    )

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation arguments")

    eval_group.add_argument(
        "--eval_languages",
        nargs="+",
        type=str,
        help="List of language codes for evaluation",
        choices=['de', 'en', 'es', 'pl', 'uk', 'fr', 'it', 'ru'],
    )

    eval_group.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file where the evaluation results will be saved",
    )

    return parser.parse_args()


if __name__ == "__main__":
    config = DF_FT_Config()
    args = parse_arguments()
    print(args.eval_languages)
