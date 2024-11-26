from os.path import join, normpath
from pathlib import Path
from re import split
from typing import Union, Tuple, Literal, Optional, List

import pandas as pd
from sklearn.utils import resample
from configuration.df_ft_config import DF_FT_Config
from configuration.rawboost_config import _RawboostConfig
from df_logger import main_logger
from src.datasets.dataset_sampler import DFDataset


class MLAADDataset(DFDataset):
    def __init__(
            self,
            samples_df: pd.DataFrame = None,
            return_label=True,
            config: DF_FT_Config = None,
            processor=None,
            root_path: Union[Path, str] = None,
            subset: Literal['train', 'val', 'test'] = None,
            languages: Optional[List[str]] = None,
            predefined_column_and_values: Optional[Tuple[str, List[str], Literal["exclude", "include"]]] = None,
            split_languages_separately: bool = True,
            # e.g. ("path", ["path1", "path2", ...])

    ):
        super().__init__(
            samples_df=samples_df,
            return_label=return_label,
            config=config,
            processor=processor,
            subset=subset
        )
        self.root_path = root_path
        self.languages = languages

        self.predefined_column_and_values = predefined_column_and_values
        self.split_languages_separately = split_languages_separately

        # get the samples from the protocol files
        self.samples_df = self.get_dataframe_from_protocol(file_prefix="meta.csv")
        # use predefined list of paths if provided
        if self.predefined_column_and_values is not None:
            self.samples_df = self._use_predefined_list_of_samples(
                column_name=self.predefined_column_and_values[0],
                list_of_values=self.predefined_column_and_values[1],
                mode=self.predefined_column_and_values[2]
            )
            assert len(self.samples_df) > 0, "No samples found in the dataset. Check the values in the predefined list."
        # split subset
        else:
            self.samples_df = self.split_dataframe(
                self.samples_df
            )
        self.samples_df = self._extract_bona_fide_samples( self.samples_df)
        main_logger.info(f"Dataset loaded. Number of samples: {len(self)}.")

    
    def _read_and_concatenate_csv(
            self,
            start_path: Union[str, Path],
            file_prefix: str = "meta.csv",
            csv_separator: str = "|"
    ) -> pd.DataFrame:
        """
        Walks through a directory, looking for CSV files that start with a given prefix,
        reads them into Pandas DataFrames, and concatenates them into one large DataFrame.
        """

        start_path = Path(start_path)

        # get all the files that start with the given prefix (meta files)
        dataframes = [
            pd.read_csv(file, sep=csv_separator)
            for file in start_path.rglob(file_prefix)
        ]

        # concatenate all the dataframes into one large dataframe (in the 'path' column should be path to spoof audio
        df = pd.concat(dataframes, ignore_index=True)

        df['path'] = df['path'].apply(lambda x: normpath(join(self.root_path, x)))

        # set the label to 'spoof'
        df['label'] = self.config.spoof_label[0]

        return df

    
    def _extract_bona_fide_samples(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Separate the bona-fide samples from the spoof samples in the dataset.
        :param dataframe:
        :return: concatenated DataFrame with spoof and bona-fide samples
        """

        main_logger.info("Separating bona-fide samples from the spoof samples.")
        df_copy = dataframe.copy().reset_index(drop=True)
        # Step 1: Replace 'path' with 'original_file' values
        df_copy['path'] = df_copy['original_file'].apply(lambda x: normpath(join(self.root_path, "bonafide", x)))

        # Step 2: Replace 'original_file' with '-'
        df_copy['original_file'] = '-'

        # Step 3: Set 'label' as 'bonafide'
        df_copy['label'] = self.config.bona_fide_label[0]

        df = pd.concat(
            [dataframe.set_index('path'), df_copy.set_index('path')]
        ).sample(frac=1, random_state=self.config.seed).reset_index()

        main_logger.info("Concatenated spoof and bona-fide samples.")

        return df.loc[~df['path'].duplicated(keep='first')]


    def _equalize_samples(self, dataframe, n=1000):
        # Separate spoof and non-spoof samples
        spoof_df = dataframe[dataframe['label'] == 'spoof']
       
        # Initialize an empty list to store balanced spoof samples
        balanced_spoof_list = []
        
        # Loop through each unique architecture
        for architecture in spoof_df['architecture'].unique():
            # Loop through each unique language
            for language in spoof_df['language'].unique():
                # Filter for current architecture and language
                df = spoof_df[(spoof_df['architecture'] == architecture) & (spoof_df['language'] == language)]
                if len(df) == 0:
                    continue
                # If there are more samples than needed, downsample
                if len(df) > n:
                    df = df.sample(n=n, random_state=42)
                # If there are fewer samples than needed, upsample
                elif len(df) < n:
                    df = resample(df, replace=True, n_samples=n, random_state=42)
                
                # Append balanced data to the list
                balanced_spoof_list.append(df)

        return pd.concat(balanced_spoof_list)
        

        
    
    def get_dataframe_from_protocol(self, file_prefix: str) -> pd.DataFrame:
        """Read the protocol file for MLAAD and return a DataFrame with the samples."""

        main_logger.info(f"Reading protocol files. Looking for: '{file_prefix}' in the {self.root_path} directory.")

        dataframe = self._read_and_concatenate_csv(
            start_path=self.root_path,
            file_prefix=file_prefix,
            csv_separator="|"
        )
        dataframe = self._equalize_samples(dataframe, n=1000)
        # separate bona-fide samples (to exstract bona-fide samples to the 'path' column)


        if self.languages is not None:
            dataframe = dataframe.loc[dataframe['language'].isin(self.languages)]

        dataframe['user_id'] = dataframe.apply(
            lambda row: split(r"mix/|female/|male/", row['original_file'])[-1].split('/')[0], axis=1
        )
        return dataframe

    
    def split_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Split the dataset into train, validation, and test sets.

        :param dataframe: DataFrame with samples to split
        :param split_languages_separately:
            If True: Each language will be split separately (e.g. 70% of English, 70% of French, etc.)
            If False: The whole dataset will be split according to the split_ratio
        """
        main_logger.info("Splitting the dataset into train, validation, and test sets.")
        if self.languages is not None and self.subset is not None:
            main_logger.info(f"Splitting languages {self.languages} together.")
            return dataframe[dataframe['language'].isin(self.languages)].reset_index(drop=True)
        elif self.languages is not None:
            return dataframe[dataframe['language'].isin(self.languages)].reset_index(drop=True)
        return dataframe


if __name__ == "__main__":
    langs = ['ru', 'en', 'es', 'fr', 'de', 'it', 'pl', 'uk']
    
    test_filter = ("architecture", ['vits', 'griffin_lim'], "include")
    train_dataset = MLAADDataset(
            config=DF_FT_Config(rawboost_config=_RawboostConfig(algo_id=0)),
            root_path=Path("/storage1/bartekmarek/mlin/"),
            languages=langs,
            predefined_column_and_values=test_filter,  # type: ignore
            split_languages_separately=True,

        )

    print(train_dataset.samples_df.columns)
