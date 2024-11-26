from os import makedirs
from pathlib import Path
from typing import Dict, Tuple, Union

from pydantic import model_serializer, Field

from configuration.base_config import _BaseConfig
from configuration.train_config import _TrainerConfig


class DF_FT_Config(_BaseConfig):
    bona_fide_label: Tuple = Field(default=("bonafide", 1), description="Label for bonafide samples in the dataset")
    spoof_label: Tuple = Field(default=("spoof", 0), description="Label for spoof samples in the dataset")
    label_column: str = Field(default="label", description="Label column name")
    trainer_config: _TrainerConfig = Field(default=_TrainerConfig(), description="Training parameters")

    root_path_to_protocol: Union[Path, str] = Field(
        default=None,
        description="Path to the protocol file for the dataset"
    )

    root_dir: Union[Path, str] = Field(
        default=None,
        description="Root directory for the dataset"
    )

    ft_out_dir: Union[Path, str] = Field(
        default="fine_tuned_models",
        description="Path to save the configuration file"
    )

    evaluate_output_file: Union[Path, str] = Field(
        default="evaluation_results.csv",
        description="Path to save the evaluation results"
    )

    path_to_test_samples: Union[Path, str] = Field(
        default="data/mlaad/testing_recordings.csv",
        description="Path to the file with the test samples"
    )

    base_models_dir: Union[Path, str] = Field(
        default="all_models",
        description="Path to the directory with the base models"
    )

    k_folds: int = Field(
        default=5,
        description="Number of folds for the k-fold cross-validation"
    )


    @model_serializer
    def serialize_model(self) -> Dict:
        results = super().serialize_model()
        results.update({
            "bonafide_label": self.bonafide_label,
            "spoof_label": self.spoof_label,
            "label_column": self.label_column
        })
        return results

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        makedirs(self.ft_out_dir, exist_ok=True)


if __name__ == '__main__':
    print(DF_FT_Config().model_dump())
