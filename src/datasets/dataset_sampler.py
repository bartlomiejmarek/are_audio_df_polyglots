from typing import Literal, List

import pandas as pd

from configuration.df_ft_config import DF_FT_Config
from src.datasets.base_dataset import BaseDataset


class DFDataset(BaseDataset):
    def __init__(
            self,
            samples_df: pd.DataFrame,
            config: DF_FT_Config,
            return_label: bool = True,
            processor=None,
            method: Literal['over', 'under'] = None,
            subset: Literal['train', 'val', 'test'] = None,
    ):
        super().__init__(
            samples_df=samples_df,
            return_label=return_label,
            processor=processor,
            subset=subset,
            config=config
        )

        if method is not None:
            self._resample_dataset(method)
        self._grouped_samples = None

    @property
    def grouped_samples(self):
        if self._grouped_samples is None:
            self._grouped_samples = self.samples_df.groupby(self.config.label_column)
        return self._grouped_samples

    def _resample_dataset(self, method: Literal['over', 'under']) -> None:
        bona_length = len(self.grouped_samples.get_group(self.config.bonafide_label[0]))
        spoof_length = len(self.grouped_samples.get_group(self.config.spoof_label[0]))

        assert any(["over" == method.lower(),
                    "under" == method.lower()]), f"Method must be either 'over' or 'under'. Got {method}."
        if method == 'over':
            target_length = max(bona_length, spoof_length)
        else:  # under
            target_length = min(bona_length, spoof_length)

        resampled = [
            group.sample(target_length, replace=(method == 'over'))
            for _, group in self.grouped_samples
        ]

        self.samples_df = pd.concat(resampled, ignore_index=True)
        self._grouped_samples = None  # Reset cached groupby

    def _get_samples_by_label(self, label: str) -> pd.DataFrame:
        """Return samples for a specific label."""
        return self.grouped_samples.get_group(label)

    def _get_bonafide_only(self) -> pd.DataFrame:
        """Return only the bonafide samples."""
        return self._get_samples_by_label(self.config.bonafide_label[0])

    def _get_spoof_only(self) -> pd.DataFrame:
        """Return only the spoof samples."""
        return self._get_samples_by_label(self.config.spoof_label[0])