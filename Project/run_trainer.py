"""Module that can be runned as script and used to train a model with certain
arguments."""
# Python Standard Libraries
import argparse
import os

# Third Party Libraries
from sentence_transformers import CrossEncoder

# Project Libraries
from dataframes_loader import DataFramesLoader
import model_io
from model_trainer import ModelTrainer
import utils


def check_if_model_folder_already_exists(
    model_path: str, preloaded_model: bool
) -> None:
    """Checks if the model folder already exists and, if so, asks the user whether to overwrite it or not.

    If the user don't want to overwrite it, it'll exit the script.

    Parameters
    ----------
    model_path : str
        The path to the model. It must be a folder name.
    preloaded_model : bool
        Whether the model is preloaded or not.
    """
    if os.path.exists(model_path) and not preloaded_model:
        usr_input = input(
            f"The folder {model_path} already exists and you didn't specify you want to retrain the model. Do you want to delete the previous trained model and train it again from scratch (y/[n])? "
        )

        if usr_input == "y":
            print(f"Deleting {model_path} folder...")
            os.system(f"rm -rf {model_path}")
        else:
            print("Exiting...")
            exit(-1)


def parse_input():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script trains a model on the train dataset and saves it."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the model to save. This is the only required argument.",
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
        help="The number of epochs to train the model. The default is 5.",
    )
    parser.add_argument(
        "--warmup_steps",
        "-ws",
        type=int,
        default=100,
        help="The number of warmup steps for the model. The default is 100.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size for the dataloader. The default is 4.",
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
    batch_size: int,
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
    batch_size : int
        The batch size for the dataloader.
    preloaded_model : bool
        If specified, it'll assume the model name is an already used model and
        will train it again.
    verbose : bool
        Whether to print the progress or not.
    """
    df_loader = DataFramesLoader()
    train_df, _ = df_loader.get_datasets(preloaded_data)
    models_path = utils.get_global_vars()["models_path"]
    model_path = f"{models_path}{model_name}"

    check_if_model_folder_already_exists(model_path, preloaded_model)

    if preloaded_model:
        model = model_io.load_model(model_path)
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
        batch_size=batch_size,
        verbose=verbose,
    )

    if verbose:
        print(f"Saving model to {model_path}")
    model_io.save_model(model, model_path)


if __name__ == "__main__":
    args = parse_input()

    main(
        model_name=args.model_name,
        preloaded_data=args.preloaded_data,
        use_bert=args.bert,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        preloaded_model=args.preloaded_model,
        verbose=args.verbose,
    )
