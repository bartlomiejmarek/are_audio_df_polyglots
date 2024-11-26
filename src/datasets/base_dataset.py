from os.path import isfile
from typing import Union, Tuple, List, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset

import torchaudio
from configuration.base_config import _BaseConfig
from src.datasets.audio_utils import apply_preprocessing
from src.datasets.rawboost import process_rawboost_feature


class BaseDataset(Dataset):
    def __init__(
            self,
            samples_df: pd.DataFrame,
            return_label: bool = True,
            config: _BaseConfig = None,
            processor=None,  # for HF datasets
            subset: Literal['train', 'val', 'test'] = None,

    ):
        self.return_label = return_label
        self.config = config
        self.processor = processor
        self.subset = subset
        self.samples_df = samples_df

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        """
        return len(self.samples_df)

    def limit_dataset(self, reduced_number: int) -> pd.DataFrame:
        """
        Limit the dataset to a certain number of samples.
        """
        return self.samples_df.sample(reduced_number, random_state=self.config.seed)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple]:
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            A tuple containing the processed audio data
        """
        sample = self.samples_df.iloc[index]

        assert isfile(sample["path"]), f"File {sample['path']} does not exist!"

        waveform, audio_sample_rate = torchaudio.load(sample["path"], normalize=self.config.audio_config.normalize)

        if self.subset == 'train':
            # apply rawboost
            waveform = process_rawboost_feature(
                waveform,
                audio_sample_rate,
                self.config.rawboost_config
            )
            waveform = waveform.unsqueeze(0) if isinstance(waveform, torch.Tensor) else torch.from_numpy(waveform).unsqueeze(0)

        waveform, sample_rate = apply_preprocessing(waveform, audio_sample_rate, **self.config.audio_config.model_dump())

        # for hugging face processors
        if self.processor is not None:
            waveform = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

        return_data: List[Union[torch.Tensor, int, Tuple]] = [waveform]

        if self.return_label:
            return_data.append(self.config.bona_fide_label[1] if sample["label"] == self.config.bona_fide_label[0] else
                               self.config.spoof_label[1])

        return tuple(return_data) if len(return_data) > 1 else return_data[0]
