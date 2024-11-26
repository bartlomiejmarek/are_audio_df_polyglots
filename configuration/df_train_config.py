from os import makedirs
from pathlib import Path
from typing import Dict, Tuple, Union

from pydantic import model_serializer, Field

from configuration.base_config import _BaseConfig
from configuration.train_config import _TrainerConfig


class DF_Train_Config(_BaseConfig):
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

    out_model_dir: Union[Path, str] = Field(
        default="all_models",
        description="Path to the directory with the base models"
    )

    config_dir: Union[Path, str] = Field(
        default="model_configs/training",
        description="Path to the directory with base .yaml configuration files"
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
        makedirs(self.out_model_dir, exist_ok=True)


if __name__ == '__main__':
    print(DF_Train_Config().model_dump())
