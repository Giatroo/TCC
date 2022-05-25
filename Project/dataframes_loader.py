from typing import Tuple

import pandas as pd
from pandas import DataFrame

import utils


class DataFramesLoader:
    def _assert_answer_labels_are_valid(self, df: DataFrame) -> None:
        """Asserts that the answer labels are valid.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to assert.
        """
        assert all(
            (df["answer1_label"] == 0) | (df["answer1_label"] == 1)
        ), "answer1_label must be 0 or 1"
        assert all(
            (df["answer2_label"] == 0) | (df["answer2_label"] == 1)
        ), "answer2_label must be 0 or 1"

    def _assert_answer_labels_are_different(self, df: DataFrame) -> None:
        """Assert that the answer labels are different.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to assert.
        """
        assert all(
            df["answer1_label"] != df["answer2_label"]
        ), "answer1_label and answer2_label must be different"

    def _preprocess_df(self, df: DataFrame) -> DataFrame:
        """Preprocesses the DataFrame.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to preprocess.

        Returns
        -------
        DataFrame
            The preprocessed DataFrame.
        """
        df = df.dropna()

        return df

    def _assert_df_values(self, df: DataFrame) -> None:
        """Asserts that the DataFrame values are valid.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to assert.
        """
        self._assert_answer_labels_are_valid(df)
        self._assert_answer_labels_are_different(df)

    def _read_dataset(self, dataset_path: str) -> DataFrame:
        """Reads and returns the dataset.

        Parameters
        ----------
        dataset_path : str
            The path to the dataset.

        Returns
        -------
        DataFrame
            The dataset.
        """
        df = pd.read_pickle(dataset_path)
        return df

    def get_dataset(self, dataset_path: str) -> DataFrame:
        """Returns a dataset. It asserts the values in the dataset are correct
        and preprocesses it.

        Parameters
        ----------
        dataset_path : str
            The path to the dataset.

        Returns
        -------
        DataFrame
            The dataset.
        """
        df = self._read_dataset(dataset_path)
        self._assert_df_values(df)
        df = self._preprocess_df(df)
        return df

    def get_datasets(
        self, preloaded: bool = True
    ) -> Tuple[DataFrame, DataFrame]:
        """Returns tha train and test datasets.

        Returns
        -------
        train_df, test_df : Tuple[DataFrame, DataFrame]
            The train and test datasets.
        """
        prefix = "preloaded_" if preloaded else "balanced_unhashed_"

        dataset_path = utils.get_global_vars()["dataset_path"]
        train_df = self.get_dataset(f"{dataset_path}{prefix}train_df.pkl")
        test_df = self.get_dataset(f"{dataset_path}{prefix}test_df.pkl")
        return train_df, test_df
