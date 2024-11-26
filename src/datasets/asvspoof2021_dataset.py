import os
from pathlib import Path
from typing import Dict, Union, Literal

import pandas as pd

from configuration.df_ft_config import DF_FT_Config
from src.datasets.dataset_sampler import DFDataset
from src.datasets.utils import find_files_mapping


class ASVspoof2021Dataset(DFDataset):
    def __init__(
            self,
            samples_df: pd.DataFrame = None,
            return_label=True,
            config: DF_FT_Config = None,
            processor=None,
            path_to_protocol: Union[Path, str] = None,
            full_paths_mapping: Dict = None,
            root_path: Union[Path, str] = None,
            subset: Literal['train', 'val', 'test'] = None,

    ):

        super().__init__(
            samples_df=samples_df,
            return_label=return_label,
            config=config,
            processor=processor,
            subset=subset
        )
        self.full_paths_mapping = full_paths_mapping
        self.root_path = root_path
        self.samples_df = self.get_dataframe_from_protocol(path_to_protocol)


    def get_dataframe_from_protocol(self, path_to_protocol: str) -> pd.DataFrame:
        """Read the protocol file for ASVspoof2021 and return a DataFrame with the samples."""
        samples = {
            "sample_name": [],
            "label": [],
            "path": [],
            "attack_type": [],
        }
        with open(path_to_protocol, mode="r") as file:
            for i, line in enumerate(file):
                self.process_line(line, samples, self.full_paths_mapping)
        return pd.DataFrame(samples).astype(str).reset_index(drop=True)

    def process_line(self, line: str, samples: Dict[str, list], paths: Dict[str, str]):
        """Process a line from the protocol file and update the samples dictionary."""
        sample_name, attack_type, label = line.strip().split(" ")[1], line.strip().split(" ")[4], \
            line.strip().split(" ")[5]
        # Check if the sample name is in the paths
        if sample_name not in paths.keys():
            return
        # Add the sample to the samples dictionary
        samples["sample_name"].append(sample_name)
        samples["label"].append(label)
        samples["attack_type"].append(attack_type)
        sample_path = paths.get(sample_name, None)
        # Check if the sample path exists
        assert sample_path is not None and (
                self.root_path / sample_path).exists(), f"Sample path ({sample_path}) does not exist!"
        samples["path"].append(str(self.root_path / sample_path))


if __name__ == "__main__":
    mapping = find_files_mapping(Path(os.path.join(
        "/media/bartek/ELITE SE880/Datasets/ASVspoof2021_DF/")),
        "*.flac"
    )

    train_dataset = ASVspoof2021Dataset(
        config=DF_FT_Config(),
        root_path=Path('/media/bartek/ELITE SE880/Datasets/ASVspoof2021_DF/'),
        path_to_protocol="/media/bartek/ELITE SE880/Datasets/ASVspoof2021_DF/keys/CM/trial_metadata.txt",
        full_paths_mapping=mapping,
        subset="train"
    )
    val_dataset = ASVspoof2021Dataset(
        config=DF_FT_Config(),
        root_path=Path('/media/bartek/ELITE SE880/Datasets/ASVspoof2021_DF/'),
        path_to_protocol="/media/bartek/ELITE SE880/Datasets/ASVspoof2021_DF/keys/CM/trial_metadata.txt",
        full_paths_mapping=mapping,
        subset="val"
    )

    test_dataset = ASVspoof2021Dataset(
        config=DF_FT_Config(),
        root_path=Path('/media/bartek/ELITE SE880/Datasets/ASVspoof2021_DF/'),
        path_to_protocol="/media/bartek/ELITE SE880/Datasets/ASVspoof2021_DF/keys/CM/trial_metadata.txt",
        full_paths_mapping=mapping,
        subset="test"
    )

    print(bool(set(train_dataset.samples_df['path']) & set(val_dataset.samples_df['path']) & set(
        test_dataset.samples_df['path'])))
