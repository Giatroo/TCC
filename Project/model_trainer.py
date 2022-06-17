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
        verbose : bool, default=True
            Whether to print the progress or not.
        Returns
        -------
        model : CrossEncoder
            The model after training.
        """
        model = get_model()
        train_dataloader = self._create_dataloader(dataset, verbose=verbose)

        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=verbose,
        )
        return model

    def save_model(self, model: CrossEncoder, model_path: str) -> None:
        """Saves a model.

        Parameters
        ----------
        model : CrossEncoder
            The model to save.
        model_path : str
            The path to save the model. It must be a folder name.
        """
        model.save(model_path)

    def load_model(self, model_path: str) -> CrossEncoder:
        """Loads a model.

        Parameters
        ----------
        model_path : str
            The path to the model. It must be a folder name.

        Returns
        -------
        model : CrossEncoder
            The pre-trained model.
        """
        model = CrossEncoder(model_path)
        return model


def parse_input():
    # Python Standard Libraries
    import argparse

    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script trains a model on the train dataset."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the model to save.",
    )
    parser.add_argument(
        "--preloaded_data",
        action="store_true",
        default=False,
        help="Whether to use the preloaded data or not.",
    )
    parser.add_argument(
        "--bert",
        action="store_true",
        default=False,
        help="Whether to use BERT or DeBERTa. If not specified, DeBERTa is used.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=5,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--warmup_steps",
        "-ws",
        type=int,
        default=100,
        help="The number of warmup steps for the model.",
    )
    parser.add_argument(
        "--preloaded_model",
        action="store_true",
        default=False,
        help="If specified, it'll assume the model name is an already used model and will train it again.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Whether to print the progress or not.",
    )

    args = parser.parse_args()
    return args


def main(
    model_name: str,
    preloaded_data: bool,
    use_bert: bool,
    epochs: int,
    warmup_steps: int,
    preloaded_model: bool,
    verbose: bool,
) -> None:
    """Trains a model and saves it.

    Parameters
    ----------
    model_name : str
        The name of the model to save.
    preloaded_data : bool
        Whether to use the preloaded data or not.
    use_bert : bool
        Whether to use BERT or DeBERTa. If not specified, DeBERTa is used.
    epochs : int
        The number of epochs to train the model.
    warmup_steps : int
        The number of warmup steps for the model.
    preloaded_model : bool
        If specified, it'll assume the model name is an already used model and
        will train it again.
    verbose : bool
        Whether to print the progress or not.
    """
    # Project Libraries
    from dataframes_loader import DataFramesLoader
    import utils

    df_loader = DataFramesLoader()
    train_df, _ = df_loader.get_datasets(preloaded_data)
    models_path = utils.get_global_vars()["models_path"]

    if preloaded_model:
        model = CrossEncoder(f"{models_path}{model_name}")
        get_model = lambda: model
    else:
        get_model = (
            ModelTrainer.get_bert if use_bert else ModelTrainer.get_deberta
        )

    trainer = ModelTrainer()
    model = trainer.train_model(
        train_df,
        get_model,
        epochs=epochs,
        warmup_steps=warmup_steps,
        verbose=verbose,
    )

    if verbose:
        print(f"Saving model to {models_path}{model_name}")
    trainer.save_model(model, f"{models_path}{model_name}")


if __name__ == "__main__":
    args = parse_input()

    main(
        model_name=args.model_name,
        preloaded_data=args.preloaded_data,
        use_bert=args.bert,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        preloaded_model=args.preloaded_model,
        verbose=args.verbose,
    )
