# Python Standard Libraries
import typing

# Third Party Libraries
from pandas import DataFrame
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project Libraries
import utils


class ModelTrainer:
    def _create_dataloader(
        self,
        df: DataFrame,
        shuffle: bool = True,
        batch_size: int = 4,
        verbose=False,
    ) -> DataLoader:
        """Receives a dataset and creates a dataloader for it.

        Parameters
        ----------
        df : DataFrame
            The dataset.
        shuffle : bool, default=True
            Whether to shuffle the dataset or not.
        batch_size : int, default=4
            The batch size for the dataloader.
        verbose : bool, default=False
            Whether to print the progress or not.

        Returns
        -------
        DataLoader
            The dataloader.
        """
        examples = list()
        for _, row in tqdm(
            list(df.iterrows()), desc="Creating dataloader", disable=not verbose
        ):
            comment, answer1, label1 = utils.get_example_input(row, 1)
            _, answer2, label2 = utils.get_example_input(row, 2)

            example = InputExample(texts=[comment, answer1], label=label1)
            examples.append(example)
            example = InputExample(texts=[comment, answer2], label=label2)
            examples.append(example)

        dataloader = DataLoader(
            examples, shuffle=shuffle, batch_size=batch_size
        )
        return dataloader

    @staticmethod
    def get_deberta() -> CrossEncoder:
        """Returns the DeBERTa model."""
        model = CrossEncoder("microsoft/deberta-base", num_labels=2)
        return model

    @staticmethod
    def get_bert() -> CrossEncoder:
        """Returns the BERT model."""
        model = CrossEncoder("bert-base-uncased", num_labels=2)
        return model

    GetModelFunction = typing.Callable[[], CrossEncoder]

    def train_model(
        self,
        dataset: DataFrame,
        get_model: GetModelFunction,
        epochs: int = 5,
        warmup_steps: int = 100,
        batch_size: int = 8,
        verbose: bool = True,
    ) -> CrossEncoder:
        """Trains the model and returns it.

        Parameters
        ----------
        dataset : DataFrame
            The dataset to train the model on.
        get_model : GetModelFunction
            A function that returns the model.
        epochs : int, default=5
            The number of epochs.
        warmup_steps : int, default=100
            The number of warmup steps.
        batch_size : int, default=8
            The batch size for the dataloader.
        verbose : bool, default=True
            Whether to print the progress or not.
        Returns
        -------
        model : CrossEncoder
            The model after training.
        """
        model = get_model()
        train_dataloader = self._create_dataloader(
            dataset, batch_size=batch_size, verbose=verbose
        )

        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=verbose,
        )
        return model
