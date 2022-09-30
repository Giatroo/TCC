# Python Standard Libraries
import argparse
import math
from typing import Tuple

# Third Party Libraries
import ktrain
from ktrain import text
import pandas as pd
from sklearn.model_selection import train_test_split

# Project Libraries
from dataframes_loader import DataFramesLoader
from model_io import ModelIO


def main(
    preloaded_data: bool = False,
    training_instances: int = int(1e20),
    validation_perc: float = 0.15,
    model_name="microsoft/deberta-base",
    model_save_name="deberta_base",
    batch_size: int = 8,
    epochs: int = 2,
    learning_rate: float = 5e-5,
    verbose: bool = False,
):
    model_io = ModelIO(verbose=verbose)

    print(training_instances)

    train_df = get_train_df(preloaded_data, training_instances)
    train_df, validation_df = get_train_validation_df(train_df, validation_perc)

    t = text.Transformer(
        model_name,
        maxlen=512,
        class_names=train_df["answer_label"].unique().tolist(),
    )

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(validation_df)

    trn = t.preprocess_train(X_train, y_train)
    val = t.preprocess_train(X_val, y_val)

    model = t.get_classifier()
    learner = ktrain.get_learner(
        model, train_data=trn, val_data=val, batch_size=batch_size
    )

    learner.autofit(learning_rate, epochs)

    predictor = ktrain.get_predictor(learner.model, preproc=t)
    model_io.save_model(predictor, model_save_name)


def transform_two_answers_per_row_to_one_answer_per_row(df):
    dfs = list()
    for i in range(1, 3):
        columns = ["comment_text", f"answer{i}_text", f"answer{i}_label"]
        map_columns = {
            f"answer{i}_text": "answer_text",
            f"answer{i}_label": "answer_label",
        }
        dfs.append(df[columns].rename(columns=map_columns))

    return pd.concat(dfs).reset_index(drop=True)


def get_X_y(df: pd.DataFrame):
    X = df[["comment_text", "answer_text"]].values
    X = list(map(tuple, X))

    y = df["answer_label"].values

    return X, y


def get_train_df(preloaded: bool, training_instances: int) -> pd.DataFrame:
    df_loader = DataFramesLoader()
    train_df, _ = df_loader.get_datasets(preloaded=preloaded)
    train_df = transform_two_answers_per_row_to_one_answer_per_row(train_df)
    train_df = train_df[:training_instances]
    return train_df


def get_train_validation_df(
    train_df: pd.DataFrame,
    validation_perc: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    size_split = int(len(train_df) * validation_perc)
    train_df, validation_df = train_test_split(
        train_df, test_size=size_split, random_state=42  # type: ignore
    )
    train_df.reset_index(inplace=True, drop=True)  # type: ignore
    validation_df.reset_index(inplace=True, drop=True)  # type: ignore
    return train_df, validation_df  # type: ignore


def parse_input():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="This script trains a model on the train dataset and saves it."
    )
    parser.add_argument(
        "model_save_name",
        type=str,
        help="The name of the model to save. This is the only required argument.",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="microsoft/deberta-base",
        help='The name of the model used by the HuggingFace library. The default is "microsoft/deberta-base"',
    )
    parser.add_argument(
        "--preloaded_data",
        action="store_true",
        default=False,
        help="Whether to use the preloaded data or not.",
    )
    parser.add_argument(
        "--training_instances",
        type=int,
        default=int(1e20),
        help="The number of instances to train on. If this number is above the number of rows in the training dataset, the whole dataset will be used. The default is infinity.",
    )
    parser.add_argument(
        "--validation_perc",
        type=float,
        default=0.15,
        help="The percentage of the training data to be used on the validation set. The default is 0.15.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=2,
        help="The number of epochs to train the model. The default is 2.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="The batch size for the training process. The default is 8.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=5e-5,
        help="The learning rate for the model. The default is 5e-5.",
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


if __name__ == "__main__":
    args = parse_input()

    main(
        preloaded_data=args.preloaded_data,
        training_instances=args.training_instances,
        validation_perc=args.validation_perc,
        model_name=args.model_name,
        model_save_name=args.model_save_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
    )
