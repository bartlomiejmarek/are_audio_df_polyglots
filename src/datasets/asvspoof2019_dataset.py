import copy
from os.path import join
from pathlib import Path
from typing import Dict, Union, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from configuration.df_ft_config import DF_FT_Config
from configuration.rawboost_config import _RawboostConfig
from src.datasets.dataset_sampler import DFDataset
from src.datasets.utils import find_files_mapping


class ASVspoof2019Dataset(DFDataset):
    name_mapping = {
        "train": "train",
        "val": "dev",
        "test": "eval"
    }

    def __init__(
            self,
            samples_df: pd.DataFrame = None,
            return_label=True,
            config: DF_FT_Config = None,
            processor=None,
            protocol_dir: Union[Path, str] = None,
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

        self.root_path = join(root_path, f"ASVspoof2019_LA_{self.name_mapping[subset]}")
        self.full_paths_mapping = find_files_mapping(
            Path(self.root_path),
            "*.flac",
            path_as_str=False
        )
        self.samples_df = self.get_dataframe_from_protocol(
                join(
                    protocol_dir,
                    f"ASVspoof2019.LA.cm.{self.name_mapping[subset]}.tr{'n' if subset == 'train' else 'l'}.txt"
                )
            )

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
        sample_name, attack_type, label = line.strip().split(" ")[1], line.strip().split(" ")[3], \
            line.strip().split(" ")[4]
        # Check if the sample name is in the paths
        if sample_name not in paths.keys():
            return
        # Add the sample to the samples dictionary
        samples["sample_name"].append(sample_name)
        samples["label"].append(label)
        samples["attack_type"].append(attack_type)
        sample_path = paths.get(sample_name, None)
        # Check if the sample path exists
        full_path = (self.root_path / sample_path).resolve()
        assert sample_path is not None and (
                full_path).exists(), f"Sample path ({full_path}) does not exist!"
        samples["path"].append(str(full_path))


if __name__ == "__main__":

    train_dataset = ASVspoof2019Dataset(
        config=DF_FT_Config(rawboost_config=_RawboostConfig(algo_id=0)),
        root_path='/Volumes/SSD_Bartek/Datasets/LA',
        protocol_dir=Path("/Volumes/SSD_Bartek/Datasets/LA/ASVspoof2019_LA_cm_protocols"),
        subset="train"
    )

    val_dataset = ASVspoof2019Dataset(
        config=DF_FT_Config(rawboost_config=_RawboostConfig(algo_id=0)),
        root_path='/Volumes/SSD_Bartek/Datasets/LA',
        protocol_dir=Path("/Volumes/SSD_Bartek/Datasets/LA/ASVspoof2019_LA_cm_protocols"),
        subset="val"
    )

    test_dataset = ASVspoof2019Dataset(
        config=DF_FT_Config(rawboost_config=_RawboostConfig(algo_id=0)),
        root_path='/Volumes/SSD_Bartek/Datasets/LA',
        protocol_dir=Path("/Volumes/SSD_Bartek/Datasets/LA/ASVspoof2019_LA_cm_protocols"),
        subset="test"
    )

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    print(bool(set(train_dataset.samples_df['path']) & set(val_dataset.samples_df['path']) & set(
        test_dataset.samples_df['path'])))

    train_dataset.samples_df = pd.concat([
        train_dataset.samples_df,
        test_dataset.samples_df,
        val_dataset.samples_df
        ]
    ).reset_index(drop=True)

    del val_dataset, test_dataset

    # split the dataset into train and validation
    train_df, val_df = train_test_split(
        train_dataset.samples_df,
        test_size=0.2,
        random_state=42,
    )
    val_dataset = copy.deepcopy(train_dataset)
    train_dataset.samples_df = train_df
    val_dataset.samples_df = val_df
    print(f"Dataset sizes train: {len(train_dataset)}, val: {len(val_dataset)}")

